"""LLM failover (guide §8.3): two consecutive qualifying failures switch the session to the fallback for good."""

from __future__ import annotations

import httpx
import pytest
from openai import APIStatusError, APITimeoutError
from support import P, boundary_for, make_runtime

from voxcare.providers.llm import CascadeDialogueProvider


class _Adapter:
    def __init__(self, label, model, outcomes):
        self.label, self.model = label, model
        self.outcomes = list(outcomes)
        self.calls = 0

    async def plan_turn(self, **_):
        self.calls += 1
        item = self.outcomes.pop(0) if self.outcomes else []
        if isinstance(item, BaseException):
            raise item
        return item


def _status(code):
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return APIStatusError("boom", response=httpx.Response(code, request=request), body=None)


def _timeout():
    return APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))


CALL = {"user_text": "x", "history": (), "available_tools": ()}


async def test_two_consecutive_qualifying_failures_switch_for_the_rest_of_the_session():
    primary = _Adapter("OpenAI Responses", "gpt-6-luna", [_timeout(), _status(503), [], []])
    fallback = _Adapter("Anthropic Messages", "claude-haiku-5-5", [[P("list_customer_orders", {})], [], []])
    cascade = CascadeDialogueProvider(primary, fallback)
    with pytest.raises(APITimeoutError):
        await cascade.plan_turn(**CALL)
    second = await cascade.plan_turn(**CALL)  # the second failure switches and the fallback answers this turn
    assert second == [P("list_customer_orders", {})]
    await cascade.plan_turn(**CALL)
    await cascade.plan_turn(**CALL)
    assert primary.calls == 2 and fallback.calls == 3  # never switches back
    events = cascade.drain_events()
    assert len(events) == 1 and events[0]["primary_model"] == "gpt-6-luna"
    assert events[0]["active_model"] == "claude-haiku-5-5" and events[0]["failure_count"] == "2"


async def test_a_success_resets_the_counter_and_4xx_does_not_qualify():
    primary = _Adapter("p", "gpt-6-luna", [_timeout(), [], _timeout(), [], _status(400), _status(400)])
    fallback = _Adapter("f", "claude-haiku-5-5", [])
    cascade = CascadeDialogueProvider(primary, fallback)
    for expected in (APITimeoutError, None, APITimeoutError, None, APIStatusError, APIStatusError):
        if expected is None:
            await cascade.plan_turn(**CALL)
        else:
            with pytest.raises(expected):
                await cascade.plan_turn(**CALL)
    assert fallback.calls == 0 and cascade.drain_events() == ()


async def test_failover_is_audited_in_the_design_shape():
    runtime = await make_runtime("memory")
    primary = _Adapter("OpenAI Responses", "gpt-6-luna", [_status(500), _status(502)])
    fallback = _Adapter("Anthropic Messages", "claude-haiku-5-5", [[P("list_customer_orders", {})]])
    cascade = CascadeDialogueProvider(primary, fallback)
    boundary = boundary_for(runtime, cascade)
    session = await boundary.create_session()
    first = await boundary.handle_turn(session.session_id, "What did I buy before?")
    assert first.outcome == "DEPENDENCY_UNAVAILABLE"
    second = await boundary.handle_turn(session.session_id, "What did I buy before?")
    assert second.outcome == "ANSWERED"
    row = next(
        r
        for r in await runtime.repository.list_audit(session.session_id)
        if r["action_name"] == "LLM_PROVIDER_FAILOVER"
    )
    assert (row["actor"], row["action_status"], row["outcome_status"], row["mutation_idempotency_key"]) == (
        "System",
        "Success",
        "SUCCESS",
        None,
    )
    assert row["request_payload_json"] == {"primary_model": "gpt-6-luna", "consecutive_failures": 2}
    assert row["response_payload_json"] == {"active_model": "claude-haiku-5-5", "session_scope": "remainder_of_session"}
    record = await runtime.store.get_session(session.session_id)
    assert record["llm_active_provider"] == "claude-haiku-5-5"


async def test_a_primary_slower_than_the_ttft_budget_counts_as_a_timeout():
    import asyncio

    class Slow(_Adapter):
        async def plan_turn(self, **_):
            self.calls += 1
            await asyncio.sleep(0.2)
            return []

    primary = Slow("p", "gpt-6-luna", [])
    fallback = _Adapter("f", "claude-haiku-5-5", [[P("list_customer_orders", {})]])
    cascade = CascadeDialogueProvider(primary, fallback, primary_timeout_s=0.01)
    with pytest.raises(TimeoutError):
        await cascade.plan_turn(**CALL)
    assert await cascade.plan_turn(**CALL) == [P("list_customer_orders", {})]
    assert CascadeDialogueProvider(primary, None, primary_timeout_s=0.01).primary_timeout_s is None
