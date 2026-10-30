"""Typed HTTP request schemas for the REST surface."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TextTurnRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2_000)
    source: str = "manual"


class HandoffRequest(BaseModel):
    reason: str = Field(default="caller_requested_support", min_length=1, max_length=200)


__all__ = ["HandoffRequest", "TextTurnRequest"]
