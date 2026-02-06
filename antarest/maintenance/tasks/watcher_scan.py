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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape
from pathlib import Path
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
    Collect studies from all workspaces using parallel os.walk per subtree.

    Strategy:
    1. Scan workspace roots to find first-level subdirectories
    2. Pre-filter subdirectories by name
    3. Distribute filtered subtrees across threads, each running os.walk
    """
    scan_tasks: List[tuple[Path, str, List[Group], List[re.Pattern], List[re.Pattern]]] = []
    for name, workspace in config.storage.workspaces.items():
        if name != DEFAULT_WORKSPACE_NAME:
            path = Path(workspace.path)
            groups = [Group(id=escape(g), name=escape(g)) for g in workspace.groups]
            compiled_in = [re.compile(r) for r in workspace.filter_in]
            compiled_out = [re.compile(r) for r in workspace.filter_out]
            scan_tasks.append((path, name, groups, compiled_in, compiled_out))

    # Phase 1: Scan workspace roots to collect first-level subdirectories
    subtree_tasks: list[tuple[Path, str, list[Group], list[re.Pattern], list[re.Pattern]]] = []
    studies: List[StudyFolder] = []

    for root_path, ws_name, groups, c_in, c_out in scan_tasks:
        root_result = _scan_single_dir(root_path, c_in, c_out)
        if root_result == "study":
            studies.append(StudyFolder(root_path, ws_name, groups))
        elif isinstance(root_result, list):
            for subdir in root_result:
                if _should_descend(subdir.name, c_in, c_out):
                    subtree_tasks.append((subdir, ws_name, groups, c_in, c_out))

    # Phase 2: Parallel os.walk on each first-level subtree
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {
            executor.submit(_walk_subtree, subdir, c_in, c_out): (ws_name, groups)
            for subdir, ws_name, groups, c_in, c_out in subtree_tasks
        }
        for future in as_completed(futures):
            ws_name, groups = futures[future]
            try:
                for study_path in future.result():
                    studies.append(StudyFolder(study_path, ws_name, groups))
            except Exception as e:
                logger.error("Failed to scan subtree", exc_info=e)

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


def _walk_subtree(
    root: Path,
    compiled_filter_in: List[re.Pattern],
    compiled_filter_out: List[re.Pattern],
) -> List[Path]:
    """
    Walk a subtree using os.walk. Marker files (study.antares, AW_NO_SCAN)
    are detected from the filenames list — no extra I/O, already listed
    by os.walk's internal scandir call.
    """
    studies: List[Path] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dir_name = os.path.basename(dirpath)

        if not any(p.search(dir_name) for p in compiled_filter_in):
            dirnames.clear()
            continue
        if any(p.search(dir_name) for p in compiled_filter_out):
            dirnames.clear()
            continue
        if dir_name.startswith("~") and (
            dir_name.endswith(".thermal_timeseries_gen.tmp")
            or dir_name.endswith(".upgrade.tmp")
        ):
            dirnames.clear()
            continue

        if "AW_NO_SCAN" in filenames:
            dirnames.clear()
            continue
        if "study.antares" in filenames:
            studies.append(Path(dirpath))
            dirnames.clear()
            continue

        # Pre-prune: filter dirnames before os.walk descends into them
        dirnames[:] = [d for d in dirnames if _should_descend(d, compiled_filter_in, compiled_filter_out)]

    return studies


def _should_descend(
    name: str,
    compiled_filter_in: List[re.Pattern],
    compiled_filter_out: List[re.Pattern],
) -> bool:
    """Check if a subdirectory should be descended into based on name filters."""
    if not any(p.search(name) for p in compiled_filter_in):
        return False
    if any(p.search(name) for p in compiled_filter_out):
        return False
    if name.startswith("~") and (
        name.endswith(".thermal_timeseries_gen.tmp") or name.endswith(".upgrade.tmp")
    ):
        return False
    return True


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
