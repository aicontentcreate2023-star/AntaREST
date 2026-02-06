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

Uses queue-based parallel scanning with mtime caching for incremental scans.
On the first invocation, every directory is fully scanned with os.scandir().
On subsequent invocations, only directories whose mtime has changed are
re-scanned; others return their cached result via a cheap os.stat() call.
"""

import logging
import os
import re
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

# ---------------------------------------------------------------------------
# Incremental mtime cache
# ---------------------------------------------------------------------------
# Maps directory path (str) -> (mtime_ns, scan_result)
# scan_result is: None (filtered/ignored), "study", or list of subdirectory
# path strings.  Thread-safe under CPython GIL (dict __setitem__/__getitem__).
_dir_cache: dict[str, tuple[int, None | str | list[str]]] = {}


def _collect_studies(config: Config) -> List[StudyFolder]:
    """Queue-based parallel scan with mtime caching for incremental speedup."""
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
    Scan a single directory with mtime-based caching.

    1. Apply name filters (cheap, no I/O).
    2. stat() the directory to get mtime_ns.
    3. If mtime matches cache → return cached result (no readdir).
    4. Otherwise → full os.scandir(), update cache.
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

        path_str = str(path)

        # Check mtime to decide if we can use cached result
        try:
            current_mtime = os.stat(path).st_mtime_ns
        except (PermissionError, OSError):
            _dir_cache.pop(path_str, None)
            return None

        cached = _dir_cache.get(path_str)
        if cached is not None and cached[0] == current_mtime:
            # Cache hit: convert cached string paths back to Path objects
            cached_result = cached[1]
            if isinstance(cached_result, list):
                return [Path(p) for p in cached_result]
            return cached_result

        # Cache miss: full scandir
        subdirs = []
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.name == "study.antares":
                    _dir_cache[path_str] = (current_mtime, "study")
                    return "study"
                if entry.name == "AW_NO_SCAN":
                    _dir_cache[path_str] = (current_mtime, None)
                    return None
                try:
                    if entry.is_dir():
                        subdirs.append(Path(entry.path))
                except (PermissionError, OSError):
                    pass

        # Cache the subdirectory list as strings (smaller, picklable)
        _dir_cache[path_str] = (current_mtime, [str(s) for s in subdirs])
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
