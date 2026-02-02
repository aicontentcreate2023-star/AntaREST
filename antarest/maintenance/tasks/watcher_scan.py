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
    Collect studies from all workspaces (except default) using parallel scanning.

    Args:
        config: Application configuration.

    Returns:
        List of StudyFolder found in all workspaces.
    """
    from antarest.study.storage.utils import should_ignore_folder_for_scan

    # Collect all root paths to scan
    scan_tasks: List[tuple[Path, str, List[Group], List[str], List[str]]] = []
    for name, workspace in config.storage.workspaces.items():
        if name != DEFAULT_WORKSPACE_NAME:
            path = Path(workspace.path)
            groups = [Group(id=escape(g), name=escape(g)) for g in workspace.groups]
            scan_tasks.append((path, name, groups, workspace.filter_in, workspace.filter_out))

    studies: List[StudyFolder] = []
    total_dirs_scanned = 0

    # Use a thread pool to parallelize I/O-bound scanning
    # For I/O-bound NFS operations, more workers = better throughput
    max_workers = 64
    logger.info(f"[PROFILE] Starting parallel scan with {max_workers} workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for root_path, workspace, groups, filter_in, filter_out in scan_tasks:
            result, count = _parallel_scan(
                executor, root_path, workspace, groups, filter_in, filter_out, should_ignore_folder_for_scan
            )
            studies += result
            total_dirs_scanned += count

    logger.info(f"[PROFILE] Total directories scanned: {total_dirs_scanned}")
    return studies


def _parallel_scan(
    executor: ThreadPoolExecutor,
    root_path: Path,
    workspace: str,
    groups: List[Group],
    filter_in: List[str],
    filter_out: List[str],
    should_ignore_fn,
) -> tuple[List[StudyFolder], int]:
    """
    Scan directories in parallel using a thread pool.

    Strategy: BFS with parallel processing of each level's subdirectories.
    """
    studies: List[StudyFolder] = []
    dirs_scanned = 0

    # Queue of directories to process: (path, depth)
    to_process = [(root_path, 0)]

    while to_process:
        # Process current batch in parallel
        futures = {}
        for path, depth in to_process:
            future = executor.submit(_scan_single_dir, path, workspace, groups, filter_in, filter_out, should_ignore_fn)
            futures[future] = (path, depth)

        to_process = []
        for future in as_completed(futures):
            path, depth = futures[future]
            dirs_scanned += 1
            try:
                result = future.result()
                if result is None:
                    # Directory was ignored
                    continue
                if result == "study":
                    # Found a study - don't descend further
                    studies.append(StudyFolder(path, workspace, groups))
                elif isinstance(result, list):
                    # List of subdirectories to process
                    for subdir in result:
                        to_process.append((subdir, depth + 1))
            except Exception as e:
                logger.error(f"Failed to scan dir {path}", exc_info=e)

    return studies, dirs_scanned


def _scan_single_dir(
    path: Path,
    workspace: str,
    groups: List[Group],
    filter_in: List[str],
    filter_out: List[str],
    should_ignore_fn,
) -> None | str | List[Path]:
    """
    Scan a single directory and return:
    - None if directory should be ignored
    - "study" if this is a study directory
    - List of subdirectory paths to scan further
    """
    try:
        if should_ignore_fn(path, filter_in, filter_out):
            return None

        if (path / "study.antares").exists():
            return "study"

        # Collect subdirectories
        subdirs = []
        with os.scandir(path) as entries:
            for entry in entries:
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
                t0 = time.time()
                studies = _collect_studies(config)
                studies_found = len(studies)
                t1 = time.time()
                logger.info(f"[PROFILE] _collect_studies: {t1 - t0:.1f}s - Found {studies_found} studies")

                if not dry_run:
                    study_service.sync_studies_on_disk(studies, None, True)
                    t2 = time.time()
                    logger.info(f"[PROFILE] sync_studies_on_disk: {t2 - t1:.1f}s")

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
