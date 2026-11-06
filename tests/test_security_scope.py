"""NFR-08: 0.0% cross-customer leakage and every scraping burst intercepted (guide §18, §19.1).

50 cross-customer cases, 50 unauthenticated cases and 20 scraping bursts, all
through the real middleware and services (the model is only a source of proposals).
"""

from __future__ import annotations

import copy as _copy
import itertools

import pytest
from support import BACKENDS, drop_runtime, make_runtime

from voxcare.middleware.tool_middleware import ToolContext
from voxcare.storage import seed

FOREIGN = {  # customer → orders that belong to someone else
    "CUST-001": ["ORD-2002"],
    "CUST-002": ["ORD-1001", "ORD-1002", "ORD-1003"],
}
CODE_VARIANTS = ("{code}", "{code_lower}", "ORD {digits}", "ord_{digits}", "ORD{digits}", "  {code}  ", "Ord-{digits}")


def _secrets_for(order_code: str) -> set[str]:
    """Every field of a foreign order that must never reach the caller."""
    order = next(row for row in seed.ORDERS if row["order_code"] == order_code)
    items = [row for row in seed.ORDER_ITEMS if row["order_id"] == order["order_id"]]
    shipment = next((row for row in seed.SHIPMENTS if row["order_id"] == order["order_id"]), None)
    values = {
        order["shipping_address"],
        f"{int(order['total_amount']):,}",
        order["payment_method"],
        order["payment_status"],
        order["customer_id"],
    }
    values |= {item["product_name"] for item in items} | {item["sku"] for item in items}
    if shipment:
        values |= {shipment["tracking_code"], shipment["driver_name"], shipment["driver_phone_masked"]}
    customer = next(row for row in seed.CUSTOMERS if row["customer_id"] == order["customer_id"])
    values |= {customer["full_name"], customer["phone_number"]}
    return {value for value in values if value}


def _cross_customer_cases() -> list[tuple[str, str, dict]]:
    cases = []
    for customer, orders in FOREIGN.items():
        for code, variant in itertools.product(orders, CODE_VARIANTS):
            digits = code.split("-")[1]
            raw = variant.format(code=code, code_lower=code.lower(), digits=digits)
            cases.append((customer, code, {"tool": "lookup_order_details", "args": {"order_code": raw}}))
            cases.append(
                (customer, code, {"tool": "track_shipment_status", "args": {"order_code": raw, "tracking_code": None}})
            )
    for shipment in seed.SHIPMENTS:  # by tracking code, which identifies another customer's parcel
        code = next(o["order_code"] for o in seed.ORDERS if o["order_id"] == shipment["order_id"])
        cases.append(
            (
                "CUST-002",
                code,
                {
                    "tool": "track_shipment_status",
                    "args": {"order_code": None, "tracking_code": shipment["tracking_code"]},
                },
            )
        )
        cases.append(
            (
                "CUST-002",
                code,
                {
                    "tool": "track_shipment_status",
                    "args": {"order_code": None, "tracking_code": shipment["tracking_code"].lower()},
                },
            )
        )
    return cases[:50]


CROSS = _cross_customer_cases()
assert len(CROSS) == 50, len(CROSS)


@pytest.fixture(params=BACKENDS)
async def runtime(request):
    rt = await make_runtime(request.param)
    yield rt
    await drop_runtime(rt)


async def test_fifty_cross_customer_requests_leak_nothing(runtime):
    leaks, denials = [], 0
    for index, (customer, code, call) in enumerate(CROSS):
        context = ToolContext(session_id=f"SES-SCOPE-{index}", customer_id=customer, user_text="check that order")
        result = await runtime.middleware.handle(context, call["tool"], call["args"])
        assert result.envelope == {
            "outcome": "ERROR",
            "error_code": "OUT_OF_SCOPE",
            "http": 403,
            "message": result.envelope["message"],
        }, (customer, code, call)
        rows = await runtime.repository.list_audit(context.session_id)
        violation = [row for row in rows if row["action_name"] == "AUTH_SCOPE_VIOLATION"]
        assert len(violation) == 1 and violation[0]["action_status"] == "Denied"
        denials += 1
        exposed = str(result.envelope) + str([row["response_payload_json"] for row in rows])
        leaks += [(code, secret) for secret in _secrets_for(code) if secret in exposed]
    assert denials == 50
    assert leaks == []  # 0.0 % leakage


async def test_fifty_cross_customer_dialogue_replies_leak_nothing(runtime):
    from support import P, ScriptedPlanner, boundary_for

    leaks = []
    for index, (customer, code, call) in enumerate(CROSS):
        boundary = boundary_for(runtime, ScriptedPlanner([[P(call["tool"], call["args"])]]))
        session = await boundary.create_session(customer_id=customer)
        result = await boundary.handle_turn(session.session_id, f"Tell me about {code}")
        exposed = f"{result.assistant_text} {result.spoken_text} {result.tool_outcomes}"
        leaks += [(index, secret) for secret in _secrets_for(code) if secret in exposed]
        assert result.outcome == "OUT_OF_SCOPE"
    assert leaks == []


def _unauthenticated_cases() -> list[tuple[str, dict]]:
    base = (
        [("lookup_order_details", {"order_code": code}) for code in ("ORD-1001", "ORD-1002", "ORD-1003", "ORD-2002")]
        + [
            ("track_shipment_status", {"order_code": code, "tracking_code": None})
            for code in ("ORD-1001", "ORD-1002", "ORD-1003", "ORD-2002")
        ]
        + [
            ("track_shipment_status", {"order_code": None, "tracking_code": row["tracking_code"]})
            for row in seed.SHIPMENTS
        ]
        + [
            ("list_customer_orders", {}),
            ("search_catalog", {"category": "TV_AV", "brand": None, "max_price_vnd": None, "keyword": None}),
            (
                "search_catalog",
                {"category": "WASHING_MACHINE", "brand": "Panasonic", "max_price_vnd": 9000000, "keyword": None},
            ),
            ("escalate_to_human", {"trigger_reason": "CUSTOMER_REQUESTED_ESCALATION", "priority": "NORMAL"}),
        ]
    )
    cases = list(itertools.islice(itertools.cycle(base), 50))
    return cases


UNAUTH = _unauthenticated_cases()


async def test_fifty_unauthenticated_requests_never_read_customer_data(runtime):
    for index, (tool, args) in enumerate(UNAUTH):
        runtime.repository.read_log.clear()
        context = ToolContext(session_id=f"SES-UNAUTH-{index}", customer_id=None)
        result = await runtime.middleware.handle(context, tool, args)
        assert result.envelope["error_code"] == "UNAUTHENTICATED_SESSION" and result.envelope["http"] == 401
        assert set(result.envelope) == {"outcome", "error_code", "http", "message"}
        assert runtime.repository.read_log == [], (tool, args)  # rejected before any OMS/catalog read
        assert not result.invoked
    assert len(UNAUTH) == 50
    assert await runtime.repository.list_agent_activity() == []  # no handoff was recorded either


def _add_catalog_rows(runtime, count: int) -> list[str]:
    """Test-only catalog rows so a 4-distinct-SKU burst is possible (the §17 seed holds three SKUs)."""
    skus = []
    for index in range(count):
        sku = f"TV-TST-{index:02d}"
        product = {
            **_copy.deepcopy(seed.PRODUCTS[0]),
            "product_id": f"PROD-TST-{index:02d}",
            "product_name": f"Test TV {index}",
        }
        variant = {
            **_copy.deepcopy(seed.VARIANTS[0]),
            "variant_id": f"VAR-TST-{index:02d}",
            "product_id": product["product_id"],
            "sku": sku,
        }
        runtime.repository.tables["product"].append(product)
        runtime.repository.tables["product_variant"].append(variant)
        skus.append(sku)
    return skus


async def test_twenty_scraping_bursts_are_all_intercepted():
    runtime = await make_runtime("memory")
    skus = _add_catalog_rows(runtime, 12)
    intercepted = 0
    for burst in range(20):
        context = ToolContext(session_id=f"SES-BURST-{burst}", customer_id="CUST-001", user_text="next model")
        if burst % 4 == 3:  # bulk stock-count wording is refused outright
            bulk = ToolContext(
                session_id=context.session_id,
                customer_id="CUST-001",
                user_text="Give me the exact stock count for every TV you have",
            )
            result = await runtime.middleware.handle(
                bulk, "search_catalog", {"category": "TV_AV", "brand": None, "max_price_vnd": None, "keyword": None}
            )
            assert result.error_code == "OUT_OF_SCOPE"
            intercepted += 1
            continue
        window = skus[(burst % 3) * 4 : (burst % 3) * 4 + 4]
        outcomes = []
        for sku in window:
            result = await runtime.middleware.handle(
                context, "search_catalog", {"category": "TV_AV", "brand": None, "max_price_vnd": None, "keyword": sku}
            )
            outcomes.append(result.error_code)
        assert outcomes[:3] == [None, None, None]  # single-model questions stay allowed
        assert outcomes[3] == "OUT_OF_SCOPE"  # the call that reaches 4 distinct SKUs in 120 s is refused
        repeat = await runtime.middleware.handle(
            context, "search_catalog", {"category": "TV_AV", "brand": None, "max_price_vnd": None, "keyword": window[0]}
        )
        assert repeat.ok  # a model already in the window is still answered
        intercepted += 1
        rows = await runtime.repository.list_audit(context.session_id)
        throttle = [row for row in rows if row["action_name"] == "ANTI_SCRAPING_THROTTLE"]
        assert len(throttle) == 1 and throttle[0]["outcome_status"] == "DENIED"
    assert intercepted == 20
    # catalog tools never produce AUTH_SCOPE_VIOLATION (no customer-owned resource)
    assert not [row for row in await runtime.repository.list_audit() if row["action_name"] == "AUTH_SCOPE_VIOLATION"]


async def test_throttle_window_slides_after_120_seconds():
    runtime = await make_runtime("memory")
    skus = _add_catalog_rows(runtime, 4)
    clock = {"now": 1000.0}
    runtime.store.now_s = lambda: clock["now"]
    context = ToolContext(session_id="SES-SLIDE", customer_id="CUST-001")
    for sku in skus[:3]:
        assert (
            await runtime.middleware.handle(
                context, "search_catalog", {"category": "TV_AV", "brand": None, "max_price_vnd": None, "keyword": sku}
            )
        ).ok
    clock["now"] += 121
    later = await runtime.middleware.handle(
        context, "search_catalog", {"category": "TV_AV", "brand": None, "max_price_vnd": None, "keyword": skus[3]}
    )
    assert later.ok


async def test_throttle_flag_off_disables_fr_sec_01():
    runtime = await make_runtime("memory", fr_sec_01=False)
    skus = _add_catalog_rows(runtime, 5)
    context = ToolContext(
        session_id="SES-OFF", customer_id="CUST-001", user_text="Give me the exact stock count for every TV"
    )
    for sku in skus:
        result = await runtime.middleware.handle(
            context, "search_catalog", {"category": "TV_AV", "brand": None, "max_price_vnd": None, "keyword": sku}
        )
        assert result.ok
