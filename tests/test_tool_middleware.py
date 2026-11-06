"""Tool middleware pipeline (guide §6.2) and pre-invocation validator (§6.3)."""

from __future__ import annotations

import asyncio

import pytest
from support import make_runtime

from voxcare.middleware import validator
from voxcare.middleware.registry import FUTURE_WRITE_TOOLS, PHASE1_REGISTRY, TOOLS
from voxcare.middleware.tool_middleware import ToolContext

CTX = ToolContext(session_id="SES-MW", customer_id="CUST-001")


@pytest.fixture
async def runtime():
    return await make_runtime("memory")


def test_registry_is_the_phase_1_set_with_strict_schemas():
    assert PHASE1_REGISTRY == {
        "lookup_order_details",
        "track_shipment_status",
        "list_customer_orders",
        "retrieve_policy_clauses",
        "search_catalog",
        "compare_products",
        "apply_voucher",
        "escalate_to_human",
    }
    assert not PHASE1_REGISTRY & FUTURE_WRITE_TOOLS
    for tool in TOOLS:
        schema = tool["parameters"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])  # OpenAI strict mode
        assert "session_customer_id" not in schema["properties"]
    policy = next(tool for tool in TOOLS if tool["name"] == "retrieve_policy_clauses")
    assert set(policy["parameters"]["properties"]) == {"query", "category"}  # the whole signature (D§4.5)


@pytest.mark.parametrize(
    ("tool", "args", "reason"),
    [
        ("lookup_order_details", {"order_code": ""}, "order_code missing"),
        ("lookup_order_details", {"order_code": "   "}, "order_code missing"),
        ("track_shipment_status", {"order_code": None, "tracking_code": None}, "order_code and tracking_code missing"),
        ("retrieve_policy_clauses", {"query": " ", "category": "ALL"}, "query empty"),
        ("retrieve_policy_clauses", {"query": "x", "category": "FEES"}, "category outside enum"),
        (
            "search_catalog",
            {"category": "PHONES", "brand": None, "max_price_vnd": None, "keyword": None},
            "category outside enum",
        ),
        (
            "search_catalog",
            {"category": "TV_AV", "brand": None, "max_price_vnd": "15m", "keyword": None},
            "max_price_vnd has the wrong type",
        ),
        ("compare_products", {"skus": ["TV-SAM-55"]}, "skus must hold 2-3 codes"),
        ("compare_products", {"skus": ["A", "B", "C", "D"]}, "skus must hold 2-3 codes"),
        ("apply_voucher", {"voucher_code": "", "order_value_vnd": 1, "category": "TV_AV"}, "voucher_code empty"),
        (
            "apply_voucher",
            {"voucher_code": "X", "order_value_vnd": True, "category": "TV_AV"},
            "order_value_vnd has the wrong type",
        ),
        ("escalate_to_human", {"trigger_reason": "ANGRY", "priority": "NORMAL"}, "trigger_reason outside enum"),
        ("lookup_order_details", {"order_code": "ORD-1001", "extra": 1}, "unknown argument extra"),
    ],
)
def test_validator_rejects_with_invalid_input(tool, args, reason):
    envelope = validator.check(tool, args)
    assert envelope["error_code"] == "INVALID_INPUT" and envelope["http"] == 400
    assert envelope["_reason"] == reason


async def test_validator_rejection_never_reaches_the_store(runtime):
    runtime.repository.read_log.clear()
    result = await runtime.middleware.handle(CTX, "track_shipment_status", {"order_code": None, "tracking_code": ""})
    assert result.error_code == "INVALID_INPUT" and not result.invoked
    assert runtime.repository.read_log == []
    assert await runtime.repository.list_audit("SES-MW") == []


@pytest.mark.parametrize("tool", sorted(FUTURE_WRITE_TOOLS | {"drop_tables", "lookup_customer"}))
async def test_unlisted_tools_are_audited_and_never_executed(runtime, tool):
    before = await runtime.repository.get_order_by_code("ORD-1001")
    result = await runtime.middleware.handle(CTX, tool, {"order_code": "ORD-1001"})
    assert result.error_code == "SERVICE_ERROR" and not result.invoked
    assert await runtime.repository.get_order_by_code("ORD-1001") == before
    row = (await runtime.repository.list_audit("SES-MW"))[-1]
    assert (row["action_name"], row["action_status"], row["request_payload_json"]["tool_name"]) == (
        "TOOL_INVOCATION",
        "Failed",
        tool,
    )


async def test_model_supplied_session_customer_id_is_stripped(runtime, caplog):
    result = await runtime.middleware.handle(
        CTX, "lookup_order_details", {"order_code": "ORD-2002", "session_customer_id": "CUST-002"}
    )
    assert result.stripped_untrusted
    assert result.error_code == "OUT_OF_SCOPE"  # the server-side scope (CUST-001) is what counts
    assert "untrusted_input_stripped" in caplog.text


async def test_timeout_is_retried_once_with_the_same_request(runtime):
    calls = []
    original = runtime.repository.get_order_by_code

    async def slow(code):
        calls.append(code)
        await asyncio.sleep(0.05)
        return await original(code)

    runtime.repository.get_order_by_code = slow
    runtime.middleware.orders.timeout_s = 0.01
    result = await runtime.middleware.handle(CTX, "lookup_order_details", {"order_code": "ORD-1001"})
    assert result.error_code == "SERVICE_TIMEOUT" and result.envelope["http"] == 504
    assert result.attempts == 2 and calls == ["ORD-1001", "ORD-1001"]
    row = (await runtime.repository.list_audit("SES-MW"))[-1]
    assert row["request_payload_json"]["attempts"] == 2 and row["outcome_status"] == "FAILED"


async def test_service_error_is_not_retried(runtime):
    runtime.repository.fail_with = RuntimeError("index down")
    result = await runtime.middleware.handle(
        CTX, "search_catalog", {"category": "TV_AV", "brand": None, "max_price_vnd": None, "keyword": None}
    )
    runtime.repository.fail_with = None
    assert result.error_code == "SERVICE_ERROR" and result.attempts == 1 and result.envelope["http"] == 500


async def test_stock_quantity_never_leaves_the_catalog(runtime):
    await runtime.repository.set_stock("TV-LG-55", 0)
    search = await runtime.middleware.handle(
        CTX, "search_catalog", {"category": "TV_AV", "brand": "LG", "max_price_vnd": None, "keyword": None}
    )
    assert search.error_code == "STATE_CONFLICT" and search.envelope["http"] == 409
    assert "stock_quantity" not in str(search.internal)
    both = await runtime.middleware.handle(
        CTX, "search_catalog", {"category": "TV_AV", "brand": None, "max_price_vnd": None, "keyword": None}
    )
    assert {row["sku"]: row["availability"] for row in both.envelope["results"]} == {
        "TV-LG-55": "out_of_stock",
        "TV-SAM-55": "in_stock",
    }


async def test_compare_rejects_mixed_categories_and_unknown_skus(runtime):
    mixed = await runtime.middleware.handle(CTX, "compare_products", {"skus": ["TV-SAM-55", "WM-PAN-09"]})
    assert mixed.error_code == "INVALID_INPUT"
    unknown = await runtime.middleware.handle(CTX, "compare_products", {"skus": ["TV-SAM-55", "TV-XXXX-55"]})
    assert unknown.error_code == "NOT_FOUND" and unknown.internal["_missing"] == "TV-XXXX-55"


async def test_lookup_returns_no_shipment_fields_and_track_handles_no_manifest(runtime):
    lookup = await runtime.middleware.handle(CTX, "lookup_order_details", {"order_code": "ORD-1002"})
    assert not {"tracking_code", "carrier_name", "estimated_delivery_date"} & set(lookup.envelope["order"])
    track = await runtime.middleware.handle(
        CTX, "track_shipment_status", {"order_code": "ORD-1001", "tracking_code": None}
    )
    shipment = track.envelope["shipment"]
    assert shipment["order_code"] == "ORD-1001"  # G-2: every other field null, no synthetic row
    assert all(value is None for key, value in shipment.items() if key != "order_code")
    assert track.envelope["events"] == []
    by_tracking = await runtime.middleware.handle(
        CTX, "track_shipment_status", {"order_code": None, "tracking_code": "GHN-VN-882910"}
    )
    assert by_tracking.envelope["shipment"]["carrier_name"] == "GHN"
    missing = await runtime.middleware.handle(
        CTX, "track_shipment_status", {"order_code": None, "tracking_code": "GHN-VN-000000"}
    )
    assert missing.error_code == "NOT_FOUND"
