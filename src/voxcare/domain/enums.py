"""Canonical enums (P1 build guide §5, design D§3.5): defined once, imported everywhere.

Do not add members. Three traps the guide names:

- ``FAILED_NO_COMMIT`` is an OMS response value stored in
  ``response_payload_json.oms_status``; it is not a member of any enum here.
- ``PendingActionStatus.CANCELLED`` and ``OrderStatus.CANCELLED`` belong to two
  separate machines.
- ``AUTH_SCOPE_VIOLATION`` is an audit action; the caller-facing tool error for
  the same event is ``ErrorCode.OUT_OF_SCOPE``.
"""

from __future__ import annotations

from enum import StrEnum


class OrderStatus(StrEnum):
    PENDING_CONFIRMATION = "pending_confirmation"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    READY_FOR_PICKUP = "ready_for_pickup"
    SHIPPED = "shipped"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    DELIVERY_FAILED = "delivery_failed"
    RETURN_REQUESTED = "return_requested"
    RETURNED = "returned"
    CANCELLED = "cancelled"


class ShipmentStatus(StrEnum):
    LABEL_CREATED = "label_created"
    IN_TRANSIT = "in_transit"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    DELIVERY_FAILED = "delivery_failed"
    RETURNED = "returned"


class ShipmentEventType(StrEnum):
    MANIFEST_CREATED = "MANIFEST_CREATED"
    ORIGIN_PICKUP = "ORIGIN_PICKUP"
    HUB_DEPARTURE = "HUB_DEPARTURE"
    OUT_FOR_DELIVERY = "OUT_FOR_DELIVERY"
    DELIVERED = "DELIVERED"
    FAILED_ATTEMPT = "FAILED_ATTEMPT"
    RETURNED_TO_SENDER = "RETURNED_TO_SENDER"


class InstallationStatus(StrEnum):
    PENDING = "Pending"
    ASSIGNED = "Assigned"
    COMPLETED = "Completed"
    RESCHEDULED = "Rescheduled"


class RmaStatus(StrEnum):
    REQUESTED = "requested"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"


class PendingActionStatus(StrEnum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    EXECUTING = "executing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ActionType(StrEnum):
    CANCEL = "Cancel"
    RESCHEDULE = "Reschedule"
    ADDRESS_CHANGE = "AddressChange"
    REVERSAL = "Reversal"


class PolicyCategory(StrEnum):
    """Stored clause topics. ``ALL`` is a retrieval parameter only (``POLICY_QUERY_ALL``)."""

    SHIPPING = "SHIPPING"
    INSTALLATION = "INSTALLATION"
    WARRANTY = "WARRANTY"
    RETURNS = "RETURNS"


POLICY_QUERY_ALL = "ALL"


class ProductCategory(StrEnum):
    TV_AV = "TV_AV"
    REFRIGERATOR = "REFRIGERATOR"
    WASHING_MACHINE = "WASHING_MACHINE"
    AIR_CONDITIONER = "AIR_CONDITIONER"
    KITCHEN = "KITCHEN"


class Actor(StrEnum):
    CALLER = "Caller"
    BOT = "Bot"
    STAFF = "Staff"
    SYSTEM = "System"


class ActionStatus(StrEnum):
    SUCCESS = "Success"
    FAILED = "Failed"
    ABORTED = "Aborted"
    DENIED = "Denied"


class AuditOutcome(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    DENIED = "DENIED"
    ABORTED = "ABORTED"
    PENDING = "PENDING"
    UNKNOWN = "UNKNOWN"


class ActionName(StrEnum):
    TOOL_INVOCATION = "TOOL_INVOCATION"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    DELIVERY_RESCHEDULED = "DELIVERY_RESCHEDULED"
    RMA_CREATED = "RMA_CREATED"
    AUTH_SCOPE_VIOLATION = "AUTH_SCOPE_VIOLATION"
    ACTION_ABORTED_PRE_SUBMIT = "ACTION_ABORTED_PRE_SUBMIT"
    ACTION_ABORTED_POST_SUBMIT = "ACTION_ABORTED_POST_SUBMIT"
    WITHDRAWAL_CONFIRMED = "WITHDRAWAL_CONFIRMED"
    EXECUTION_PENDING = "EXECUTION_PENDING"
    MUTATION_OUTCOME_UNKNOWN = "MUTATION_OUTCOME_UNKNOWN"
    HANDOFF_REQUEST_RECORDED = "HANDOFF_REQUEST_RECORDED"
    ANTI_SCRAPING_THROTTLE = "ANTI_SCRAPING_THROTTLE"
    REVERSAL_REQUESTED = "REVERSAL_REQUESTED"
    REVERSAL_REFUSED = "REVERSAL_REFUSED"
    LLM_PROVIDER_FAILOVER = "LLM_PROVIDER_FAILOVER"
    TURN_COMPLETION_DEGRADED = "TURN_COMPLETION_DEGRADED"
    TURN_COMPLETION_FALLBACK_RESOLVED = "TURN_COMPLETION_FALLBACK_RESOLVED"


class TriggerReason(StrEnum):
    CUSTOMER_REQUESTED_ESCALATION = "CUSTOMER_REQUESTED_ESCALATION"
    CLARIFICATION_LIMIT_EXCEEDED = "CLARIFICATION_LIMIT_EXCEEDED"
    LOW_CONFIDENCE_THRESHOLD = "LOW_CONFIDENCE_THRESHOLD"
    COMPLEX_EXCEPTION = "COMPLEX_EXCEPTION"


class Priority(StrEnum):
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    URGENT = "URGENT"


class FallbackResolution(StrEnum):
    MANUAL_FINISH = "MANUAL_FINISH"
    CALLER_RETRY = "CALLER_RETRY"
    CALLER_REQUESTED_HANDOFF = "CALLER_REQUESTED_HANDOFF"


class ErrorCode(StrEnum):
    INVALID_INPUT = "INVALID_INPUT"
    UNAUTHENTICATED_SESSION = "UNAUTHENTICATED_SESSION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    NOT_FOUND = "NOT_FOUND"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    STATE_CONFLICT = "STATE_CONFLICT"
    SERVICE_TIMEOUT = "SERVICE_TIMEOUT"
    SERVICE_ERROR = "SERVICE_ERROR"


# HTTP status carried in the error envelope for each code (D§4.1; INSUFFICIENT_EVIDENCE is 200 by design).
ERROR_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.INVALID_INPUT: 400,
    ErrorCode.UNAUTHENTICATED_SESSION: 401,
    ErrorCode.OUT_OF_SCOPE: 403,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.INSUFFICIENT_EVIDENCE: 200,
    ErrorCode.STATE_CONFLICT: 409,
    ErrorCode.SERVICE_TIMEOUT: 504,
    ErrorCode.SERVICE_ERROR: 500,
}

# Inline vocabularies of D§3.3.2 (not enums in the design, but closed sets enforced by the DDL).
PAYMENT_METHODS = ("COD", "VNPAY_QR", "DOMESTIC_ATM", "CREDIT_CARD")
PAYMENT_STATUSES = ("UNPAID", "PAID", "REFUND_PENDING", "REFUNDED")
CARRIERS = ("GHN", "GHTK", "ViettelPost")
SESSION_CHANNELS = ("VoiceWebRTC", "PhoneSIP")

__all__ = [
    "CARRIERS",
    "ERROR_HTTP_STATUS",
    "PAYMENT_METHODS",
    "PAYMENT_STATUSES",
    "POLICY_QUERY_ALL",
    "SESSION_CHANNELS",
    "ActionName",
    "ActionStatus",
    "ActionType",
    "Actor",
    "AuditOutcome",
    "ErrorCode",
    "FallbackResolution",
    "InstallationStatus",
    "OrderStatus",
    "PendingActionStatus",
    "PolicyCategory",
    "Priority",
    "ProductCategory",
    "RmaStatus",
    "ShipmentEventType",
    "ShipmentStatus",
    "TriggerReason",
]
