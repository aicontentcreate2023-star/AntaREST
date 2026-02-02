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
    Collect studies from all workspaces (except default).

    Args:
        config: Application configuration.

    Returns:
        List of StudyFolder found in all workspaces.
    """
    studies: List[StudyFolder] = []
    total_dirs_scanned = 0
    for name, workspace in config.storage.workspaces.items():
        if name != DEFAULT_WORKSPACE_NAME:
            path = Path(workspace.path)
            groups = [Group(id=escape(g), name=escape(g)) for g in workspace.groups]
            result, dirs_scanned = rec_scan_for_studies_with_count(
                path, name, groups, workspace.filter_in, workspace.filter_out
            )
            studies += result
            total_dirs_scanned += dirs_scanned
    logger.info(f"[PROFILE] Total directories scanned: {total_dirs_scanned}")
    return studies


def rec_scan_for_studies_with_count(
    path: Path,
    workspace: str,
    groups: List[Group],
    filter_in: List[str],
    filter_out: List[str],
    max_depth: int | None = None,
) -> tuple[List[StudyFolder], int]:
    """Wrapper that counts directories scanned for profiling."""
    from antarest.study.storage.utils import should_ignore_folder_for_scan

    dirs_scanned = 1  # Count this directory

    try:
        if should_ignore_folder_for_scan(path, filter_in, filter_out):
            return [], dirs_scanned

        if (path / "study.antares").exists():
            return [StudyFolder(path, workspace, groups)], dirs_scanned

        if max_depth is not None and max_depth <= 0:
            return [], dirs_scanned

        folders: List[StudyFolder] = []
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_dir():
                    child_max_depth = max_depth - 1 if max_depth is not None else None
                    try:
                        result, child_count = rec_scan_for_studies_with_count(
                            Path(entry.path), workspace, groups, filter_in, filter_out, child_max_depth
                        )
                        folders += result
                        dirs_scanned += child_count
                    except Exception as e:
                        logger.error(f"Failed to scan dir {entry.path}", exc_info=e)
        return folders, dirs_scanned
    except Exception as e:
        logger.error(f"Failed to scan dir {path}", exc_info=e)
        return [], dirs_scanned


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
