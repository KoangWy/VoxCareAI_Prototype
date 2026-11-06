"""Embedding models for the Chroma policy collection (the guide leaves the model open, §3.1 and G-10).

Two choices, selected by ``VOXCARE_POLICY_EMBEDDING``:

- ``minilm`` (default): ``all-MiniLM-L6-v2`` through Chroma's built-in ONNX
  embedding function. Chroma downloads the model (≈80 MB) to
  ``~/.cache/chroma/onnx_models`` on first use.
- ``hashing``: a deterministic, offline lexical embedding (signed feature
  hashing over normalised words, bigrams and character 4-grams, IDF-weighted by
  the 9-clause corpus). It needs no download, so tests and machines without
  network access use it.

The 0.70 threshold of the design belongs to an unnamed model (G-10). Each
choice carries its own calibrated default, measured with
``scripts/calibrate_policy_threshold.py`` and recorded in docs/POLICY_RETRIEVAL.md;
``VOXCARE_RAG_THRESHOLD`` overrides it.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import pairwise
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


# Calibrated thresholds on the labeled set in scripts/calibrate_policy_threshold.py (see docs/POLICY_RETRIEVAL.md).
CALIBRATED_THRESHOLDS = {"minilm": 0.32, "hashing": 0.09}
DESIGN_THRESHOLD = 0.70


class MiniLMEmbedder:
    """all-MiniLM-L6-v2 (Chroma default ONNX); mean pooling, L2-normalised."""

    name = "minilm"

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._function = DefaultEmbeddingFunction()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(map(float, vector)) for vector in self._function(list(texts))]


_STOP = frozenset(
    """a an and are as at be by can could do does for from have how i if in is it its me my of on or our please
    the this that to we what when where which who will with would you your voxcom about any there their them""".split()
)
_CANON = {
    "fridge": "refrigerator", "fridges": "refrigerator", "refrigerators": "refrigerator",
    "washer": "washing", "washers": "washing", "laundry": "washing",
    "tv": "television", "tvs": "television", "televisions": "television", "telly": "television",
    "screen": "display", "monitor": "display",
    "deliver": "delivery", "delivered": "delivery", "delivering": "delivery", "shipping": "delivery",
    "ship": "delivery", "shipped": "delivery", "courier": "delivery", "freight": "delivery",
    "install": "installation", "installing": "installation", "installer": "installation", "setup": "installation",
    "mount": "mounting", "mounted": "mounting", "bracket": "mounting",
    "guarantee": "warranty", "covered": "warranty", "coverage": "warranty",
    "refund": "return", "returning": "return", "returned": "return", "exchange": "replacement",
    "replace": "replacement", "swap": "replacement", "faulty": "defect", "defective": "defect", "broken": "defect",
    "cost": "fee", "charge": "fee", "price": "fee", "pay": "fee", "surcharge": "fee", "extra": "fee",
    "km": "kilometer", "kilometres": "kilometer", "kilometers": "kilometer", "distance": "kilometer",
    "aircon": "air", "ac": "air", "compressor": "inverter", "motor": "inverter",
    "stairs": "stair", "floor": "stair", "elevator": "stair", "lift": "stair",
    "cash": "cod", "box": "carton", "packaging": "carton", "unboxing": "carton",
}  # fmt: skip


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn").lower()


def _stem(token: str) -> str:
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def lexical_terms(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", _fold(text))
    terms = []
    for word in words:
        if word in _STOP:
            continue
        word = _CANON.get(word, word)
        terms.append(_CANON.get(_stem(word), _stem(word)))
    return terms


class HashingEmbedder:
    """Deterministic offline embedding: IDF-weighted signed feature hashing (implementation choice)."""

    name = "hashing"
    DIMENSIONS = 4096

    def __init__(self, corpus: Iterable[str] = ()) -> None:
        documents = [set(self._features(text)) for text in corpus]
        self._n = max(len(documents), 1)
        self._df: Counter[str] = Counter(feature for document in documents for feature in document)

    @staticmethod
    def _features(text: str) -> list[str]:
        terms = lexical_terms(text)
        features = [f"w:{term}" for term in terms]
        features += [f"b:{left}_{right}" for left, right in pairwise(terms)]
        for term in terms:
            padded = f"#{term}#"
            features += [f"c:{padded[index : index + 4]}" for index in range(max(1, len(padded) - 3))]
        return features

    def _idf(self, feature: str) -> float:
        return math.log((self._n + 1) / (self._df.get(feature, 0) + 1)) + 1.0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = np.zeros(self.DIMENSIONS, dtype=np.float64)
            counts = Counter(self._features(text))
            for feature, count in counts.items():
                weight = (1.0 + math.log(count)) * self._idf(feature)
                weight *= {"w": 1.0, "b": 0.6, "c": 0.25}[feature[0]]
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.DIMENSIONS
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[index] += sign * weight
            norm = float(np.linalg.norm(vector))
            vectors.append((vector / norm if norm else vector).tolist())
        return vectors


def create_embedder(name: str, corpus: Iterable[str]) -> Embedder:
    if name == "minilm":
        return MiniLMEmbedder()
    if name == "hashing":
        return HashingEmbedder(corpus)
    raise ValueError(f"unknown policy embedding {name!r}; use minilm or hashing")


__all__ = [
    "CALIBRATED_THRESHOLDS",
    "DESIGN_THRESHOLD",
    "Embedder",
    "HashingEmbedder",
    "MiniLMEmbedder",
    "create_embedder",
    "lexical_terms",
]
