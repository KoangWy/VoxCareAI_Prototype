"""Embedded PostgreSQL (``storage/embedded_pg.py``): start, crash recovery, sharing, clean stop.

Runs only where the ``localdb`` group is installed (pgserver ships the PostgreSQL binaries) and not as root
(initdb refuses root). On Windows this exercises the hidden-console start that keeps Ctrl+C away from the
server and the log file outside the data directory.
"""

from __future__ import annotations

import importlib.util
import os
import time

import pytest

pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("pgserver") is None, reason="pgserver (localdb group) not installed"),
    pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="initdb refuses to run as root"),
]


def _server(base):
    from voxcare.storage.embedded_pg import EmbeddedPostgres

    return EmbeddedPostgres(data_dir=base / "pg", log_file=base / "pg.log")


def _query(dsn, sql):
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        cursor = conn.execute(sql)
        return cursor.fetchone() if cursor.description else None


def test_start_crash_recovery_sharing_and_clean_stop(tmp_path):
    import psutil

    first = _server(tmp_path)
    dsn = first.start()
    try:
        _query(dsn, "CREATE TABLE kept (x int)")
        _query(dsn, "INSERT INTO kept VALUES (42)")
        assert _query(dsn, "SHOW timezone") is not None
        assert (tmp_path / "pg.log").exists() and not (tmp_path / "pg" / "log").exists()

        # what Ctrl+C used to do: the server dies without a shutdown checkpoint
        pid = int((tmp_path / "pg" / "postmaster.pid").read_text().splitlines()[0])
        psutil.Process(pid).kill()
        deadline = time.monotonic() + 15
        while first.running_port() is not None and time.monotonic() < deadline:
            time.sleep(0.25)

        second = _server(tmp_path)
        dsn = second.start()  # crash recovery, within the start timeout
        assert _query(dsn, "SELECT x FROM kept") == (42,)
        assert second.owned

        shared = _server(tmp_path)
        assert shared.start() == dsn and not shared.owned  # its owner (this process) is alive
        shared.stop()
        assert second.running_port() == second.port  # a shared handle never stops the server

        second.stop()
        assert second.running_port() is None
        assert not (tmp_path / "pg.owner").exists()
    finally:  # never leave a server behind, whatever failed
        leftover = _server(tmp_path)
        if leftover.running_port() is not None:
            leftover.owned, leftover.port = True, leftover.running_port()
            leftover.stop()
