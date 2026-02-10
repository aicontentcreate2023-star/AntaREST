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

"""Watcher scan task for discovering studies on disk.

Parallel BFS with mtime caching: first run does a full os.scandir() of every
directory, subsequent runs skip unchanged directories (same mtime → cached result).
"""

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape
from pathlib import Path
from typing import Any, List

from antarest.core.config import Config
from antarest.core.utils.fastapi_sqlalchemy import db
from antarest.core.utils.lock import LockNotAcquired, create_lock
from antarest.login.model import Group
from antarest.maintenance.tasks.common import BackGroundTaskStatus, LockId, WatcherScanTaskResult
from antarest.study.model import DEFAULT_WORKSPACE_NAME, StudyFolder
from antarest.study.service import StudyService

logger = logging.getLogger(__name__)

# dir path -> (mtime_ns, scan result). Avoids re-scanning unchanged directories.
_dir_cache: dict[str, Any] = {}


def collect_studies(
    config: Config,
    workspace_name: str | None = None,
    root_path: Path | None = None,
    max_depth: int | None = None,
) -> List[StudyFolder]:
    """Parallel BFS scan with mtime caching for incremental speedup.

    Args:
        config: Application configuration.
        workspace_name: If provided with root_path, scan only this workspace starting from root_path.
        root_path: Starting directory for a partial scan (requires workspace_name).
        max_depth: Maximum BFS depth. None means unlimited, 1 means direct children only.
    """
    studies: List[StudyFolder] = []

    # to_scan entries: (path, ws_name, groups, filter_in, filter_out, depth)
    to_scan: list[tuple[Path, str, list[Group], list[re.Pattern[str]], list[re.Pattern[str]], int]] = []

    if workspace_name is not None and root_path is not None:
        # Partial scan: single workspace from a specific root
        workspace = config.storage.workspaces[workspace_name]
        groups = [Group(id=escape(g), name=escape(g)) for g in workspace.groups]
        filter_in = [re.compile(r) for r in workspace.filter_in]
        filter_out = [re.compile(r) for r in workspace.filter_out]
        to_scan.append((root_path, workspace_name, groups, filter_in, filter_out, 0))
    else:
        # Full scan: all workspaces (except default)
        for name, workspace in config.storage.workspaces.items():
            if name == DEFAULT_WORKSPACE_NAME:
                continue
            root = Path(workspace.path)
            groups = [Group(id=escape(g), name=escape(g)) for g in workspace.groups]
            filter_in = [re.compile(r) for r in workspace.filter_in]
            filter_out = [re.compile(r) for r in workspace.filter_out]
            to_scan.append((root, name, groups, filter_in, filter_out, 0))

    with ThreadPoolExecutor(max_workers=32) as executor:
        while to_scan:
            futures = {}
            for path, ws_name, groups, f_in, f_out, depth in to_scan:
                f = executor.submit(_scan_dir, path, f_in, f_out)
                futures[f] = (path, ws_name, groups, f_in, f_out, depth)

            to_scan = []
            for future in as_completed(futures):
                path, ws_name, groups, f_in, f_out, depth = futures[future]
                try:
                    result = future.result()
                    if result == "study":
                        studies.append(StudyFolder(path, ws_name, groups))
                    elif isinstance(result, list):
                        if max_depth is None or depth + 1 <= max_depth:
                            to_scan.extend((sub, ws_name, groups, f_in, f_out, depth + 1) for sub in result)
                except Exception as e:
                    logger.error(f"Failed to scan dir {path}", exc_info=e)

    return studies


def _scan_dir(path: Path, filter_in: List[re.Pattern[str]], filter_out: List[re.Pattern[str]]) -> Any:
    """Scan one directory: check filters, use mtime cache, or full scandir."""
    try:
        name = path.name
        if not any(p.search(name) for p in filter_in):
            return None
        if any(p.search(name) for p in filter_out):
            return None
        if name.startswith("~") and name.endswith((".thermal_timeseries_gen.tmp", ".upgrade.tmp")):
            return None

        key = str(path)
        try:
            mtime = os.stat(path).st_mtime_ns
        except (PermissionError, OSError):
            _dir_cache.pop(key, None)
            return None

        cached = _dir_cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]

        subdirs = []
        is_study = False
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.name == "AW_NO_SCAN":
                    _dir_cache[key] = (mtime, None)
                    return None
                if entry.name == "study.antares":
                    is_study = True
                    continue
                try:
                    if entry.is_dir():
                        subdirs.append(Path(entry.path))
                except (PermissionError, OSError):
                    pass

        if is_study:
            _dir_cache[key] = (mtime, "study")
            return "study"

        _dir_cache[key] = (mtime, subdirs)
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
                studies = collect_studies(config)
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
