"""CatalogService (``search_catalog``, ``compare_products``) and PromotionService (``apply_voucher``).

Guide §7.4-7.6, design D§4.6, D§9. ``stock_quantity`` never leaves this module:
callers get the coarse availability state only. An unrecorded attribute is
reported as not recorded, never filled in. ``apply_voucher`` is a quote and never
mutates an order.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

from ...storage.db import Repository
from ..enums import ErrorCode
from ..rules import VOUCHER_CODE_SHAPE, evaluate_voucher
from .envelope import DEFAULT_TOOL_TIMEOUT_S, error, guarded, success

CATEGORY_LABELS = {  # the wording of the INVALID_INPUT line (D§9.2.3)
    "TV_AV": "TV and audio",
    "REFRIGERATOR": "refrigeration",
    "WASHING_MACHINE": "laundry",
    "AIR_CONDITIONER": "air conditioning",
    "KITCHEN": "kitchen",
}


def availability(row: dict[str, Any]) -> str:
    return "in_stock" if int(row["stock_quantity"]) > 0 else "out_of_stock"


def short_label(row: dict[str, Any]) -> str:
    """ "Samsung 55" for the Scenario 7 comparison line (brand plus screen size, else brand)."""
    size = (row.get("specifications_json") or {}).get("screen_size_in")
    return f"{row['brand']} {size}" if size else row["brand"]


class CatalogService:
    def __init__(self, repository: Repository, *, timeout_s: float = DEFAULT_TOOL_TIMEOUT_S) -> None:
        self.repository = repository
        self.timeout_s = timeout_s

    async def search(
        self, category: str, brand: str | None, max_price_vnd: int | None, keyword: str | None
    ) -> dict[str, Any]:
        return await guarded(
            lambda: self._search(category, brand, max_price_vnd, keyword),
            timeout_s=self.timeout_s,
            system="catalog system",
        )

    async def _search(
        self, category: str, brand: str | None, max_price_vnd: int | None, keyword: str | None
    ) -> dict[str, Any]:
        rows = await self.repository.search_products(category, brand, max_price_vnd, keyword)
        if not rows:
            asked = keyword or (f"{brand} {CATEGORY_LABELS[category]}" if brand else CATEGORY_LABELS[category])
            return {**error(ErrorCode.NOT_FOUND, "No catalog model matches the search."), "_missing": asked}
        results = [
            {
                "sku": row["sku"],
                "product_name": row["product_name"],
                "brand": row["brand"],
                "category": row["category"],
                "price_vnd": int(row["price"]),
                "availability": availability(row),
            }
            for row in rows
        ]
        if len(results) == 1 and results[0]["availability"] == "out_of_stock":
            return {**error(ErrorCode.STATE_CONFLICT, "The model is out of stock."), "_sku": results[0]["sku"]}
        labels = {row["sku"]: short_label(row) for row in rows}
        return {**success(results=results), "_labels": labels}

    async def compare(self, skus: Sequence[str]) -> dict[str, Any]:
        return await guarded(lambda: self._compare(skus), timeout_s=self.timeout_s, system="catalog")

    async def _compare(self, skus: Sequence[str]) -> dict[str, Any]:
        wanted = [sku.strip().upper() for sku in skus]
        rows = {row["sku"]: row for row in await self.repository.get_products_by_skus(wanted)}
        for sku in wanted:
            if sku not in rows:
                return {**error(ErrorCode.NOT_FOUND, "A requested SKU is not in the catalog."), "_missing": sku}
        categories = {rows[sku]["category"] for sku in wanted}
        if len(categories) != 1:
            return error(ErrorCode.INVALID_INPUT, "Models from different categories cannot be compared.")
        recorded = {sku: dict(rows[sku].get("specifications_json") or {}) for sku in wanted}
        every_attribute = sorted({key for attributes in recorded.values() for key in attributes})
        not_recorded = {
            sku: [key for key in every_attribute if key not in recorded[sku]]
            for sku in wanted
            if any(key not in recorded[sku] for key in every_attribute)
        }
        return {
            **success(
                category=next(iter(categories)),
                products=[
                    {
                        "sku": sku,
                        "product_name": rows[sku]["product_name"],
                        "brand": rows[sku]["brand"],
                        "price_vnd": int(rows[sku]["price"]),
                        "attributes": recorded[sku],
                    }
                    for sku in wanted
                ],
                not_recorded=not_recorded,
            ),
            "_labels": {sku: short_label(rows[sku]) for sku in wanted},
        }


class PromotionService:
    def __init__(self, repository: Repository, *, timeout_s: float = DEFAULT_TOOL_TIMEOUT_S) -> None:
        self.repository = repository
        self.timeout_s = timeout_s

    async def apply_voucher(
        self, voucher_code: str, order_value_vnd: int, category: str, today: date
    ) -> dict[str, Any]:
        return await guarded(
            lambda: self._apply(voucher_code, order_value_vnd, category, today),
            timeout_s=self.timeout_s,
            system="promotion system",
        )

    async def _apply(self, voucher_code: str, order_value_vnd: int, category: str, today: date) -> dict[str, Any]:
        code = voucher_code.strip().upper().replace(" ", "_")
        if not VOUCHER_CODE_SHAPE.fullmatch(code):
            return {
                **error(
                    ErrorCode.INVALID_INPUT,
                    "The voucher code is malformed.",
                    valid=False,
                    discount_vnd=0,
                    capped_at_vnd=0,
                ),
                "_reason": "MALFORMED",
            }
        promotion = await self.repository.get_promotion(code)
        quote = evaluate_voucher(promotion, code, order_value_vnd, category, today)
        if quote.valid:
            return {
                **success(
                    valid=True, discount_vnd=quote.discount_vnd, capped_at_vnd=quote.capped_at_vnd, is_fictional=True
                ),
                "_code": code,
                "_quoted_price_vnd": order_value_vnd - quote.discount_vnd,
            }
        return {
            **error(
                ErrorCode.INVALID_INPUT,
                "The voucher does not apply to this order value and category.",
                valid=False,
                discount_vnd=0,
                capped_at_vnd=quote.capped_at_vnd,
                is_fictional=True,
            ),
            "_reason": quote.reason,
            "_code": code,
            "_promotion": {
                "category": promotion["category"],
                "min_order_value": int(promotion["min_order_value"]),
            }
            if promotion
            else None,
        }


__all__ = ["CATEGORY_LABELS", "CatalogService", "PromotionService", "availability", "short_label"]
