"""No speculative writes, ever (guide §10.4): no write tool name can reach the speculative path."""

from __future__ import annotations

import pytest

from voxcare.middleware.registry import FUTURE_WRITE_TOOLS, PHASE1_REGISTRY
from voxcare.session.speculative import (
    SPECULATIVE_TOOLS,
    SpeculativeCache,
    SpeculativeWriteForbidden,
    assert_speculative_allowed,
)


@pytest.mark.parametrize("name", sorted(FUTURE_WRITE_TOOLS | (PHASE1_REGISTRY - SPECULATIVE_TOOLS)))
def test_only_the_order_read_may_run_speculatively(name):
    with pytest.raises(SpeculativeWriteForbidden):
        assert_speculative_allowed(name)
    with pytest.raises(SpeculativeWriteForbidden):
        SpeculativeCache().put("S", 7, name, "ORD-1002", {"outcome": "SUCCESS"})


def test_speculative_set_is_read_only():
    assert SPECULATIVE_TOOLS == {"lookup_order_details"}
    assert not SPECULATIVE_TOOLS & FUTURE_WRITE_TOOLS


def test_cache_injects_only_on_a_matching_commit_and_evicts_mismatches():
    cache = SpeculativeCache()
    assert cache.code_in("can you check ORD-1002 for") == "ORD-1002"
    cache.put("S", 7, "lookup_order_details", "ORD-1002", {"order": "x"})
    assert cache.take("S", 7, "Can you check ORD-1003?") is None  # mismatch: evicted, real read issued
    assert cache.entries == {}
    cache.put("S", 8, "lookup_order_details", "ORD-1002", {"order": "y"})
    assert cache.take("S", 8, "Can you check ORD-1002 please") == {"order": "y"}
