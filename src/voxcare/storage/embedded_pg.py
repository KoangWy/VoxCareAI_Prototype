"""Embedded PostgreSQL for machines without Docker or a native install (``uv sync --group localdb``).

Uses the PostgreSQL binaries shipped in the ``pgserver`` wheel but manages the server itself, because
pgserver's own start-up breaks on Windows in three ways seen in practice:

1. the server shares the terminal's console, so Ctrl+C on uvicorn also kills PostgreSQL mid-write
   (exit 0xC000013A) and the next start needs crash recovery;
2. its server log lives *inside* the data directory; during crash recovery PostgreSQL fsyncs every file
   there, the log is held open by the starting server, and Windows answers with a sharing violation and
   a 30-second retry;
3. it waits only 10 s for ``pg_ctl start``, so a recovering server is reported as a failure.

Here the server runs in its own hidden console and process group (Ctrl+C in the terminal no longer
reaches it), writes its log next to the data directory (``data/pg.log``), gets a start timeout long
enough for crash recovery, and is stopped cleanly (``pg_ctl stop -m fast``) when the app shuts down.
A server already running on the same data directory is reused; it is stopped at exit only when the
process that started it is gone (a leftover from an earlier run), never while another app uses it.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("voxcare.storage")

START_TIMEOUT_S = 120  # crash recovery on a slow or antivirus-scanned disk can take tens of seconds
STOP_TIMEOUT_S = 60
SUPERUSER = "postgres"
_WINDOWS = os.name == "nt"
# Own process group + hidden console: console Ctrl+C events never reach the server.
_CREATION_FLAGS = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if _WINDOWS else 0


class EmbeddedPostgresError(RuntimeError):
    """The embedded server could not be initialised or started; the message includes the log tail."""


def binaries_dir() -> Path:
    try:
        import pgserver  # type: ignore[import-not-found]
    except ImportError:
        raise EmbeddedPostgresError(
            "DATABASE_URL is not set and the embedded PostgreSQL package is not installed. Start PostgreSQL "
            "(docker compose up -d) and set DATABASE_URL, run `uv sync --group localdb`, or set "
            "VOXCARE_DATA_BACKEND=memory for the no-database quick start."
        ) from None
    return Path(pgserver.__file__).resolve().parent / "pginstall" / "bin"


def _exe(name: str) -> str:
    return str(binaries_dir() / (f"{name}.exe" if _WINDOWS else name))


def _free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def _pid_alive(pid: int) -> bool:
    """A running process (a killed process that is not yet reaped, a zombie, counts as gone)."""
    if pid <= 0:
        return False
    try:
        import psutil  # a pgserver dependency

        try:
            return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False
    except ImportError:  # pragma: no cover - psutil ships with pgserver
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True


def _pid_present(pid: int) -> bool:
    """The PID still exists in the process table, zombies included (PostgreSQL's own lock-file test)."""
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except ImportError:  # pragma: no cover
        return _pid_alive(pid)


@dataclass
class EmbeddedPostgres:
    data_dir: Path
    log_file: Path
    host: str = "127.0.0.1"
    start_timeout_s: int = START_TIMEOUT_S
    port: int | None = None
    owned: bool = False
    _stopped: bool = field(default=False, repr=False)

    @property
    def owner_file(self) -> Path:
        return self.data_dir.parent / f"{self.data_dir.name}.owner"

    @property
    def dsn(self) -> str:
        return f"postgresql://{SUPERUSER}@{self.host}:{self.port}/postgres"

    # -- commands --

    def _run(self, args: list[str], *, timeout: float) -> str:
        with tempfile.TemporaryFile("w+") as out:  # files, not pipes: pg_ctl's children inherit the handles
            try:
                subprocess.run(  # noqa: S603 - fixed argv of the bundled PostgreSQL binaries
                    args,
                    check=True,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout,
                    creationflags=_CREATION_FLAGS,
                )
            except subprocess.CalledProcessError as exc:
                out.seek(0)
                raise EmbeddedPostgresError(f"{Path(args[0]).name} failed: {out.read().strip()}") from exc
            out.seek(0)
            return out.read()

    def _log_tail(self, lines: int = 25) -> str:
        try:
            text = self.log_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "(no server log)"
        return "\n".join(text.splitlines()[-lines:])

    # -- lifecycle --

    def initialised(self) -> bool:
        return (self.data_dir / "PG_VERSION").exists()

    def init(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("initialising embedded PostgreSQL in %s", self.data_dir)
        self._run(
            [
                _exe("initdb"),
                "-D",
                str(self.data_dir),
                "-U",
                SUPERUSER,
                "--auth=trust",
                "--auth-local=trust",
                "--encoding=utf8",
            ],
            timeout=300,
        )

    def running_port(self) -> int | None:
        """The port of a live server on this data directory (from ``postmaster.pid``), else ``None``."""
        try:
            lines = (self.data_dir / "postmaster.pid").read_text(encoding="utf-8", errors="replace").splitlines()
            pid, port = int(lines[0]), int(lines[3])
        except (OSError, ValueError, IndexError):
            return None
        if not _pid_alive(pid):
            return None
        try:
            self._run([_exe("pg_ctl"), "-D", str(self.data_dir), "status"], timeout=30)
        except (EmbeddedPostgresError, subprocess.TimeoutExpired):
            return None
        try:  # a stale pid file can name a reused PID: only a listening server counts
            with socket.create_connection((self.host, port), timeout=1.0):
                pass
        except OSError:
            return None
        return port

    def _wait_for_stale_postmaster(self, timeout_s: float = 15.0) -> None:
        """After a crash the dead postmaster's PID can linger briefly (unreaped); PostgreSQL refuses the lock
        file until it is gone. Windows has no zombies, so this returns at once there."""
        try:
            pid = int((self.data_dir / "postmaster.pid").read_text(encoding="utf-8").splitlines()[0])
        except (OSError, ValueError, IndexError):
            return
        deadline = time.monotonic() + timeout_s
        while _pid_present(pid) and not _pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.25)

    def _owner_alive(self) -> bool:
        try:
            return _pid_alive(int(self.owner_file.read_text(encoding="utf-8").strip()))
        except (OSError, ValueError):
            return False

    def start(self) -> str:
        if not self.initialised():
            self.init()
        existing = self.running_port()
        if existing is not None and self._owner_alive():
            # Another live process (the app, or init_db.py next to it) started it: share, never stop it.
            self.port, self.owned = existing, False
            logger.info("sharing the embedded PostgreSQL already running on port %s", existing)
            return self.dsn
        if existing is not None:
            # A leftover from an earlier run, possibly still tied to an old terminal (and its Ctrl+C):
            # stop it cleanly and start it again in its own hidden console.
            logger.info("restarting the embedded PostgreSQL left running by an earlier run")
            with contextlib.suppress(EmbeddedPostgresError, subprocess.TimeoutExpired):
                self._run(
                    [_exe("pg_ctl"), "-D", str(self.data_dir), "-w", "-m", "fast", "stop"], timeout=STOP_TIMEOUT_S
                )
        self._wait_for_stale_postmaster()
        self.port = _free_port(self.host)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info("starting embedded PostgreSQL on %s:%s (log: %s)", self.host, self.port, self.log_file)
        try:
            self._run(
                [
                    _exe("pg_ctl"),
                    "-D",
                    str(self.data_dir),
                    "-w",
                    "-t",
                    str(self.start_timeout_s),
                    "-o",
                    f'-h "{self.host}"',
                    "-o",
                    f"-p {self.port}",
                    "-l",
                    str(self.log_file),
                    "start",
                ],
                timeout=self.start_timeout_s + 15,
            )
        except (EmbeddedPostgresError, subprocess.TimeoutExpired) as exc:
            raise EmbeddedPostgresError(
                f"the embedded PostgreSQL did not start within {self.start_timeout_s} s. "
                f"Last lines of {self.log_file}:\n{self._log_tail()}"
            ) from exc
        self.owned = True
        self.owner_file.write_text(str(os.getpid()), encoding="utf-8")
        atexit.register(self.stop)
        return self.dsn

    def stop(self) -> None:
        """Fast, clean shutdown (``pg_ctl stop -m fast``) of a server this process owns; idempotent."""
        if self._stopped or not self.owned:
            return
        self._stopped = True
        if self.port is not None and self.running_port() == self.port:
            try:
                self._run(
                    [_exe("pg_ctl"), "-D", str(self.data_dir), "-w", "-m", "fast", "stop"], timeout=STOP_TIMEOUT_S
                )
                logger.info("embedded PostgreSQL stopped")
            except (EmbeddedPostgresError, subprocess.TimeoutExpired) as exc:
                logger.warning("could not stop the embedded PostgreSQL cleanly: %s", exc)
        with contextlib.suppress(OSError, ValueError):
            if int(self.owner_file.read_text(encoding="utf-8").strip()) == os.getpid():
                self.owner_file.unlink()

    # Runtime.aclose() and the old pgserver handle both call ``cleanup``.
    cleanup = stop


__all__ = ["START_TIMEOUT_S", "EmbeddedPostgres", "EmbeddedPostgresError", "binaries_dir"]
