"""Injected clock. Domain code never reads the wall clock (P1 build guide §1).

The demo anchor is 2026-10-14 in Asia/Ho_Chi_Minh: "tomorrow" = 2026-10-15,
"12 days ago" = 2026-10-02, "next Monday" = 2026-10-19 (D§13). The anchor time
of day, 09:42 local, is the one the handoff fixture id encodes
(``HND-20261014-0942-8821``); that time is fixture filler.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

DEMO_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")
DEMO_ANCHOR = datetime(2026, 10, 14, 9, 42, tzinfo=DEMO_TIMEZONE)


class Clock(Protocol):
    def now(self) -> datetime:
        """Timezone-aware current instant."""
        ...


class FixedClock:
    """A clock pinned to one instant (tests and the demo). ``advance`` moves it forward."""

    def __init__(self, instant: datetime = DEMO_ANCHOR) -> None:
        if instant.tzinfo is None:
            raise ValueError("FixedClock needs a timezone-aware instant")
        self._instant = instant

    def now(self) -> datetime:
        return self._instant

    def advance(self, **delta: float) -> None:
        self._instant += timedelta(**delta)


class SystemClock:
    """The wall clock, for deployments that leave the demo anchor (``VOXCARE_CLOCK=system``)."""

    def now(self) -> datetime:
        return datetime.now(DEMO_TIMEZONE)


def local_today(clock: Clock) -> date:
    """The calendar date in Asia/Ho_Chi_Minh."""
    return clock.now().astimezone(DEMO_TIMEZONE).date()


__all__ = ["DEMO_ANCHOR", "DEMO_TIMEZONE", "Clock", "FixedClock", "SystemClock", "local_today"]
