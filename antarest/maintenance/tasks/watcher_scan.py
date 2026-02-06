# Copyright (c) 2026, RTE (https://www.rte-france.com)
#
# See AUTHORS.txt
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0
#
# This file is part of the Antares project.

"""Watcher scan task for discovering studies on disk."""

import logging
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path
from queue import Queue
from typing import List

from antarest.core.config import Config
from antarest.core.utils.fastapi_sqlalchemy import db
from antarest.core.utils.lock import LockNotAcquired, create_lock
from antarest.login.model import Group
from antarest.maintenance.tasks.common import BackGroundTaskStatus, LockId, WatcherScanTaskResult
from antarest.study.model import DEFAULT_WORKSPACE_NAME, StudyFolder
from antarest.study.service import StudyService

logger = logging.getLogger(__name__)


def _collect_studies(config: Config) -> List[StudyFolder]:
    """
    Collect studies using GNU find for C-native filesystem traversal,
    with post-filtering in Python. Falls back to Python-based scanning
    if find is unavailable.
    """
    workspace_roots: list[str] = []
    workspace_map: dict[str, tuple[str, list[Group], list[re.Pattern], list[re.Pattern]]] = {}

    for name, workspace in config.storage.workspaces.items():
        if name != DEFAULT_WORKSPACE_NAME:
            root_path = Path(workspace.path)
            groups = [Group(id=escape(g), name=escape(g)) for g in workspace.groups]
            compiled_in = [re.compile(r) for r in workspace.filter_in]
            compiled_out = [re.compile(r) for r in workspace.filter_out]

            # Check root name against filters (same as original behavior)
            root_name = root_path.name
            if not any(p.search(root_name) for p in compiled_in):
                continue
            if any(p.search(root_name) for p in compiled_out):
                continue

            root_str = str(root_path)
            workspace_roots.append(root_str)
            workspace_map[root_str] = (name, groups, compiled_in, compiled_out)

    if not workspace_roots:
        return []

    # Single find command: C-native traversal, prune temp dirs, tag output
    # fmt: off
    cmd = ["find"] + workspace_roots + [
        "(", "-name", "~*.thermal_timeseries_gen.tmp", "-type", "d", "-prune", ")",
        "-o",
        "(", "-name", "~*.upgrade.tmp", "-type", "d", "-prune", ")",
        "-o",
        "(", "-name", "study.antares", "-printf", "S:%h\\n", ")",
        "-o",
        "(", "-name", "AW_NO_SCAN", "-printf", "N:%h\\n", ")",
    ]
    # fmt: on

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300  # noqa: S603
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.warning(f"find command failed, falling back to Python scan: {e}")
        return _collect_studies_python(config)

    if result.returncode > 1:
        logger.warning(f"find exited with code {result.returncode}, falling back to Python scan")
        return _collect_studies_python(config)

    # Parse tagged output
    study_paths: list[str] = []
    noscan_dirs: set[str] = set()

    for line in result.stdout.splitlines():
        if line.startswith("S:"):
            study_paths.append(line[2:])
        elif line.startswith("N:"):
            noscan_dirs.add(line[2:])

    # Sort workspace roots longest-first so nested workspaces match correctly
    sorted_roots = sorted(workspace_map.keys(), key=len, reverse=True)

    studies: list[StudyFolder] = []
    for study_path in study_paths:
        # Check if under a no-scan directory
        if noscan_dirs and any(study_path == ns or study_path.startswith(ns + "/") for ns in noscan_dirs):
            continue

        # Map to workspace and apply filter_in/filter_out
        for root in sorted_roots:
            if study_path == root or study_path.startswith(root + "/"):
                ws_name, groups, c_in, c_out = workspace_map[root]
                relative = study_path[len(root) :].lstrip("/")
                if not relative or _passes_path_filters(relative, c_in, c_out):
                    studies.append(StudyFolder(Path(study_path), ws_name, groups))
                break

    return studies


def _passes_path_filters(
    relative_path: str,
    compiled_filter_in: list[re.Pattern],
    compiled_filter_out: list[re.Pattern],
) -> bool:
    """Check that every path component passes filter_in/filter_out."""
    for part in relative_path.split("/"):
        if not any(p.search(part) for p in compiled_filter_in):
            return False
        if any(p.search(part) for p in compiled_filter_out):
            return False
        if part.startswith("~") and (
            part.endswith(".thermal_timeseries_gen.tmp") or part.endswith(".upgrade.tmp")
        ):
            return False
    return True


# ---------------------------------------------------------------------------
# Python fallback (used when GNU find is not available)
# ---------------------------------------------------------------------------


def _collect_studies_python(config: Config) -> List[StudyFolder]:
    """Queue-based parallel scanning fallback."""
    scan_tasks: List[tuple[Path, str, List[Group], List[re.Pattern], List[re.Pattern]]] = []
    for name, workspace in config.storage.workspaces.items():
        if name != DEFAULT_WORKSPACE_NAME:
            path = Path(workspace.path)
            groups = [Group(id=escape(g), name=escape(g)) for g in workspace.groups]
            compiled_in = [re.compile(r) for r in workspace.filter_in]
            compiled_out = [re.compile(r) for r in workspace.filter_out]
            scan_tasks.append((path, name, groups, compiled_in, compiled_out))

    studies: List[StudyFolder] = []
    results_queue: Queue = Queue()
    outstanding = 0

    with ThreadPoolExecutor(max_workers=32) as executor:
        for root_path, workspace_name, groups, compiled_in, compiled_out in scan_tasks:
            future = executor.submit(_scan_single_dir, root_path, compiled_in, compiled_out)
            metadata = (root_path, workspace_name, groups, compiled_in, compiled_out)
            future.add_done_callback(lambda f, m=metadata: results_queue.put((f, m)))
            outstanding += 1

        while outstanding > 0:
            done_future, (dir_path, ws_name, grps, c_in, c_out) = results_queue.get()
            outstanding -= 1

            try:
                result = done_future.result()
                if result == "study":
                    studies.append(StudyFolder(dir_path, ws_name, grps))
                elif isinstance(result, list):
                    for subdir in result:
                        f = executor.submit(_scan_single_dir, subdir, c_in, c_out)
                        meta = (subdir, ws_name, grps, c_in, c_out)
                        f.add_done_callback(lambda f2, m=meta: results_queue.put((f2, m)))
                        outstanding += 1
            except Exception as e:
                logger.error(f"Failed to scan dir {dir_path}", exc_info=e)

    return studies


def _scan_single_dir(
    path: Path,
    compiled_filter_in: List[re.Pattern],
    compiled_filter_out: List[re.Pattern],
) -> None | str | List[Path]:
    """
    Scan a single directory in a single os.scandir() pass.
    Returns None (ignored), "study" (found), or list of subdirectory Paths.
    """
    try:
        name = path.name

        if not any(p.search(name) for p in compiled_filter_in):
            return None
        if any(p.search(name) for p in compiled_filter_out):
            return None

        if name.startswith("~") and (
            name.endswith(".thermal_timeseries_gen.tmp") or name.endswith(".upgrade.tmp")
        ):
            return None

        subdirs = []
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.name == "study.antares":
                    return "study"
                if entry.name == "AW_NO_SCAN":
                    return None
                try:
                    if entry.is_dir():
                        subdirs.append(Path(entry.path))
                except (PermissionError, OSError):
                    pass
        return subdirs

    except Exception as e:
        logger.error(f"Error scanning {path}: {e}")
        return None


def scan_workspaces(
    config: Config,
    study_service: StudyService,
    dry_run: bool = False,
) -> WatcherScanTaskResult:
    """
    Scan all workspaces for studies and sync with database.

    Args:
        config: Application configuration.
        study_service: Study service for database synchronization.
        dry_run: If True, only scan without syncing to database.

    Returns:
        WatcherScanTaskResult with scan statistics.
    """
    start_time = time.time()
    studies_found = 0

    logger.info(f"Starting watcher scan (dry_run={dry_run})")

    try:
        with db():
            with create_lock(db.session, lock_id=LockId.WATCHER_SCAN):
                studies = _collect_studies(config)
                studies_found = len(studies)
                logger.info(f"Found {studies_found} studies across all workspaces")

                if not dry_run:
                    study_service.sync_studies_on_disk(studies, None, True)

    except LockNotAcquired:
        logger.warning("Could not acquire lock, another watcher scan is probably running")
        return WatcherScanTaskResult(
            status=BackGroundTaskStatus.SKIPPED,
            reason="lock_not_acquired",
            studies_found=0,
            duration_seconds=time.time() - start_time,
            dry_run=dry_run,
        )
    except Exception as e:
        logger.error("Watcher scan failed", exc_info=e)
        return WatcherScanTaskResult(
            status=BackGroundTaskStatus.ERROR,
            error=str(e),
            studies_found=0,
            duration_seconds=time.time() - start_time,
            dry_run=dry_run,
        )

    duration = time.time() - start_time
    logger.info(f"Watcher scan done in {duration:.1f}s: {studies_found} studies found")

    return WatcherScanTaskResult(
        status=BackGroundTaskStatus.SUCCESS,
        studies_found=studies_found,
        duration_seconds=duration,
        dry_run=dry_run,
    )
