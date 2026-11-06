"""Tool middleware (guide §6.2): the only path from a model's tool proposal to a domain service.

Pipeline for one proposed call:

1. allowlist (``PHASE1_REGISTRY``): anything else, future-only writes included, is audited and never executed;
2. strip ``session_customer_id`` if the model supplied it (untrusted input, logged);
3. pre-invocation validator (no I/O);
4. verified-session gate for ``REQUIRES_VERIFIED_CUSTOMER`` → UNAUTHENTICATED_SESSION 401, before any read;
5. FR-SEC-01 throttle for ``search_catalog`` (bulk-stock wording, ≥4 distinct SKUs in 120 s);
6. the service, with ``session_customer_id`` injected server-side, retried once on SERVICE_TIMEOUT only;
7. audit: ``TOOL_INVOCATION`` with ``tool_name`` in the request payload (masked), plus
   ``AUTH_SCOPE_VIOLATION`` (System, Denied/DENIED, details concealed) on OUT_OF_SCOPE.

The model has zero database connectivity: a call is only a proposal.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..domain.clock import Clock, local_today
from ..domain.enums import ActionName, ActionStatus, Actor, AuditOutcome, ErrorCode
from ..domain.services.catalog import CatalogService, PromotionService
from ..domain.services.envelope import error, public
from ..domain.services.handoff import HandoffContext, HandoffService
from ..domain.services.orders import CustomerOrdersService, OrderService, ShipmentService
from ..domain.services.policy import PolicyService
from ..observability.audit import AuditWriter
from ..observability.otel import tracer
from . import validator
from .registry import PHASE1_REGISTRY, REQUIRES_VERIFIED_CUSTOMER
from .throttle import CatalogThrottle

logger = logging.getLogger("voxcare.middleware")

TOOL_RETRY_MAX = 1
_ORDER_SCOPED_TOOLS = frozenset({"lookup_order_details", "track_shipment_status"})


@dataclass
class ToolContext:
    """Server-side session facts the middleware injects; never visible in a tool schema."""

    session_id: str
    customer_id: str | None
    user_text: str = ""
    turn_epoch: int = 0
    fixture_session: bool = False
    active_order_code: str | None = None
    transcript: Sequence[Mapping[str, str]] = ()
    symptom_history: Sequence[Mapping[str, Any]] = ()
    recommendation: str | None = None
    summary: str | None = None


@dataclass
class ToolResult:
    name: str
    arguments: dict[str, Any]
    envelope: dict[str, Any]  # public: what the dialogue layer and the model may see
    internal: dict[str, Any] = field(default_factory=dict)  # includes ``_`` keys for rendering and audit
    invoked: bool = False
    attempts: int = 0
    stripped_untrusted: bool = False
    # Each pipeline stage the proposal went through, in order: {"check", "result": pass|fail|note, "detail"?}.
    # Read by the session activity log; never sent to the model.
    checks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.envelope.get("outcome") == "SUCCESS"

    @property
    def error_code(self) -> str | None:
        return self.envelope.get("error_code")


class ToolMiddleware:
    def __init__(
        self,
        *,
        orders: OrderService,
        shipments: ShipmentService,
        customer_orders: CustomerOrdersService,
        policy: PolicyService,
        catalog: CatalogService,
        promotions: PromotionService,
        handoff: HandoffService,
        throttle: CatalogThrottle,
        audit: AuditWriter,
        clock: Clock,
        auth_required: bool = True,
    ) -> None:
        self.audit = audit
        self.throttle = throttle
        self.clock = clock
        self.auth_required = auth_required
        self.orders, self.shipments, self.customer_orders = orders, shipments, customer_orders
        self.policy, self.catalog, self.promotions, self.handoff = policy, catalog, promotions, handoff
        self._services: dict[str, Callable[[dict[str, Any], ToolContext], Awaitable[dict[str, Any]]]] = {
            "lookup_order_details": lambda a, c: orders.lookup_order(a["order_code"], c.customer_id or ""),
            "track_shipment_status": lambda a, c: shipments.track_shipment(
                a.get("order_code"), a.get("tracking_code"), c.customer_id or ""
            ),
            "list_customer_orders": lambda a, c: customer_orders.list_orders(c.customer_id or ""),
            "retrieve_policy_clauses": lambda a, c: policy.retrieve(a["query"], a["category"]),
            "search_catalog": lambda a, c: catalog.search(
                a["category"], a.get("brand"), a.get("max_price_vnd"), a.get("keyword")
            ),
            "compare_products": lambda a, c: catalog.compare(a["skus"]),
            "apply_voucher": lambda a, c: promotions.apply_voucher(
                a["voucher_code"], a["order_value_vnd"], a["category"], local_today(self.clock)
            ),
            "escalate_to_human": lambda a, c: handoff.escalate(
                a["trigger_reason"],
                a["priority"],
                HandoffContext(
                    session_id=c.session_id,
                    customer_id=c.customer_id or "",
                    active_order_code=c.active_order_code,
                    transcript=c.transcript,
                    symptom_history=c.symptom_history,
                    recommendation=c.recommendation,
                    summary=c.summary,
                    fixture_session=c.fixture_session,
                ),
            ),
        }

    async def handle(self, context: ToolContext, name: str, arguments: Mapping[str, Any] | None) -> ToolResult:
        with tracer().start_as_current_span(
            f"tool:{name}", attributes={"session_id": context.session_id, "turn_epoch": context.turn_epoch}
        ) as span:
            result = await self._handle(context, name, dict(arguments or {}))
            span.set_attribute("tool.outcome", result.envelope.get("outcome", ""))
            if result.error_code:
                span.set_attribute("tool.error_code", result.error_code)
            span.set_attribute("tool.invoked", result.invoked)
            return result

    async def _handle(self, context: ToolContext, name: str, args: dict[str, Any]) -> ToolResult:
        checks: list[dict[str, Any]] = []
        result = await self._run(context, name, args, checks)
        result.checks = checks
        return result

    async def _run(
        self, context: ToolContext, name: str, args: dict[str, Any], checks: list[dict[str, Any]]
    ) -> ToolResult:
        def mark(check: str, result: str, detail: str | None = None) -> None:
            checks.append({"check": check, "result": result, **({"detail": detail} if detail else {})})

        if name not in PHASE1_REGISTRY:
            mark("allowlist", "fail", "not in the Phase 1 tool list; never executed")
            envelope = error(ErrorCode.SERVICE_ERROR, "That tool is not available.")
            await self._audit_tool(
                context,
                name,
                {},
                envelope,
                ActionStatus.FAILED,
                AuditOutcome.FAILED,
                extra={"rejected": "not_allowlisted"},
            )
            logger.warning("tool proposal rejected: %s is not allowlisted", name)
            return ToolResult(name, {}, envelope, envelope)
        mark("allowlist", "pass")

        stripped = False
        if "session_customer_id" in args:
            args.pop("session_customer_id")
            stripped = True
            mark("untrusted_input", "note", "removed a session_customer_id the model supplied")
            logger.warning("untrusted_input_stripped: session_customer_id supplied by the model for %s", name)

        rejection = validator.check(name, args)
        if rejection is not None:
            mark("arguments", "fail", str(rejection.get("_reason") or "invalid arguments"))
            return ToolResult(name, args, public(rejection), rejection, stripped_untrusted=stripped)
        mark("arguments", "pass")

        if name in REQUIRES_VERIFIED_CUSTOMER and self.auth_required:
            if context.customer_id is None:
                mark("verified_session", "fail", "caller not verified; no data was read")
                envelope = error(ErrorCode.UNAUTHENTICATED_SESSION, "The call session is not verified.")
                await self._audit_tool(context, name, args, envelope, ActionStatus.DENIED, AuditOutcome.DENIED)
                return ToolResult(name, args, envelope, envelope, stripped_untrusted=stripped)
            mark("verified_session", "pass", context.customer_id)

        if name == "search_catalog" and self.throttle.enabled:
            if self.throttle.bulk_request(args.get("keyword"), context.user_text):
                mark("throttle", "fail", "bulk stock request")
                return await self._throttled(context, name, args, reason="bulk_stock_request", stripped=stripped)

        started = time.perf_counter()
        service = self._services[name]
        internal = await service(args, context)
        attempts = 1
        while internal.get("error_code") == ErrorCode.SERVICE_TIMEOUT and attempts <= TOOL_RETRY_MAX:
            logger.warning("%s timed out; retrying once with the same request", name)
            internal = await service(args, context)
            attempts += 1
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        notes = [] if internal.get("outcome") == "SUCCESS" else [f"returned {internal.get('error_code')}"]
        if attempts > 1:
            notes.append("retried once after a timeout")
        failed = internal.get("error_code") in (ErrorCode.SERVICE_TIMEOUT, ErrorCode.SERVICE_ERROR)
        mark("service", "fail" if failed else "pass", "; ".join(notes) or None)

        if name == "search_catalog" and internal.get("outcome") == "SUCCESS":
            skus = [row["sku"] for row in internal.get("results", [])]
            if not await self.throttle.admit(context.session_id, skus):
                mark("throttle", "fail", f"more than {self.throttle.distinct_skus} distinct SKUs in the window")
                return await self._throttled(
                    context, name, args, reason="distinct_sku_window", stripped=stripped, skus=skus
                )
            if self.throttle.enabled:
                mark("throttle", "pass")
        if name == "compare_products" and internal.get("outcome") == "SUCCESS":
            await self.throttle.record(context.session_id, args["skus"])

        if internal.get("_scope_violation"):
            mark("customer_scope", "fail", "belongs to another customer; no details returned")
            await self.audit.write(
                session_id=context.session_id,
                action_name=ActionName.AUTH_SCOPE_VIOLATION,
                actor=Actor.SYSTEM,
                action_status=ActionStatus.DENIED,
                outcome_status=AuditOutcome.DENIED,
                order_id=internal.get("_order_id"),
                request={
                    "tool_name": name,
                    **{key: value for key, value in args.items() if value is not None},
                    "session_customer_id": context.customer_id,
                },
                response={"error_code": "AUTH_SCOPE_VIOLATION", "details_concealed": True},
            )
        elif name in _ORDER_SCOPED_TOOLS and internal.get("outcome") == "SUCCESS":
            mark("customer_scope", "pass", "on the verified account")
        elif name == "list_customer_orders" and internal.get("outcome") == "SUCCESS":
            mark("customer_scope", "pass", "only the verified customer's orders")
        envelope = public(internal)
        if name != "escalate_to_human":  # the handoff service writes HANDOFF_REQUEST_RECORDED itself
            status, outcome = _audit_status(envelope)
            await self._audit_tool(
                context,
                name,
                args,
                envelope,
                status,
                outcome,
                order_id=internal.get("_order_id"),
                extra={"attempts": attempts, "latency_ms": latency_ms} if attempts > 1 else None,
            )
        return ToolResult(name, args, envelope, internal, invoked=True, attempts=attempts, stripped_untrusted=stripped)

    async def _throttled(
        self,
        context: ToolContext,
        name: str,
        args: dict[str, Any],
        *,
        reason: str,
        stripped: bool,
        skus: Sequence[str] = (),
    ) -> ToolResult:
        envelope = error(ErrorCode.OUT_OF_SCOPE, "Bulk catalog requests are throttled.")
        await self.audit.write(
            session_id=context.session_id,
            action_name=ActionName.ANTI_SCRAPING_THROTTLE,
            actor=Actor.SYSTEM,
            action_status=ActionStatus.DENIED,
            outcome_status=AuditOutcome.DENIED,
            request={"tool_name": name, **{k: v for k, v in args.items() if v is not None}, "reason": reason},
            response={
                "error_code": "OUT_OF_SCOPE",
                "throttled": True,
                "distinct_sku_limit": self.throttle.distinct_skus,
                **({"skus_in_request": list(skus)} if skus else {}),
            },
        )
        internal = {**envelope, "_throttled": True}
        return ToolResult(
            name, args, envelope, internal, invoked=reason != "bulk_stock_request", stripped_untrusted=stripped
        )

    async def _audit_tool(
        self,
        context: ToolContext,
        name: str,
        args: Mapping[str, Any],
        envelope: Mapping[str, Any],
        status: ActionStatus,
        outcome: AuditOutcome,
        *,
        order_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            await self.audit.write(
                session_id=context.session_id,
                action_name=ActionName.TOOL_INVOCATION,
                actor=Actor.BOT,
                action_status=status,
                outcome_status=outcome,
                order_id=order_id,
                request={"tool_name": name, **{k: v for k, v in args.items() if v is not None}, **(extra or {})},
                response=audit_summary(name, envelope),
            )
        except Exception as exc:  # a read result is already decided; the failure is logged, never hidden
            logger.error("audit write failed for %s: %s", name, type(exc).__name__)


def _audit_status(envelope: Mapping[str, Any]) -> tuple[ActionStatus, AuditOutcome]:
    if envelope.get("outcome") == "SUCCESS":
        return ActionStatus.SUCCESS, AuditOutcome.SUCCESS
    code = envelope.get("error_code")
    if code in (ErrorCode.OUT_OF_SCOPE, ErrorCode.UNAUTHENTICATED_SESSION):
        return ActionStatus.DENIED, AuditOutcome.DENIED
    if code in (ErrorCode.SERVICE_TIMEOUT, ErrorCode.SERVICE_ERROR):
        return ActionStatus.FAILED, AuditOutcome.FAILED
    return ActionStatus.SUCCESS, AuditOutcome.FAILED  # the call ran and returned a business error


def audit_summary(name: str, envelope: Mapping[str, Any]) -> dict[str, Any]:
    """A compact, masked-safe response snapshot: identifiers and states, never full clause text or PII rows."""
    if envelope.get("outcome") != "SUCCESS":
        summary = {key: envelope.get(key) for key in ("outcome", "error_code", "http") if key in envelope}
        for key in ("highest_similarity_score", "threshold_required", "valid", "discount_vnd"):
            if key in envelope:
                summary[key] = envelope[key]
        if envelope.get("error_code") == "OUT_OF_SCOPE":
            summary["details_concealed"] = True
        return summary
    if name == "lookup_order_details":
        order = envelope["order"]
        return {"order_status": order["order_status"], "version": order["version"], "item_count": len(order["items"])}
    if name == "track_shipment_status":
        shipment = envelope["shipment"]
        return {
            "shipment_status": shipment["shipment_status"],
            "estimated_delivery_date": shipment["estimated_delivery_date"],
            "event_count": len(envelope["events"]),
        }
    if name == "list_customer_orders":
        return {"order_codes": [order["order_code"] for order in envelope["orders"]]}
    if name == "retrieve_policy_clauses":
        return {
            "clauses": [[clause["clause_id"], clause["similarity_score"]] for clause in envelope["clauses"]],
            "corpus_id": envelope["corpus_id"],
            "is_fictional": envelope["is_fictional"],
        }
    if name == "search_catalog":
        return {"skus": [row["sku"] for row in envelope["results"]]}
    if name == "compare_products":
        return {"skus": [row["sku"] for row in envelope["products"]], "not_recorded": envelope.get("not_recorded")}
    if name == "apply_voucher":
        return {key: envelope[key] for key in ("valid", "discount_vnd", "capped_at_vnd")}
    return {"outcome": envelope.get("outcome")}


__all__ = ["TOOL_RETRY_MAX", "ToolContext", "ToolMiddleware", "ToolResult", "audit_summary"]
