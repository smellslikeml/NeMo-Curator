# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provenance-grounded failure diagnosis and adaptive recovery for OCR scoring.

Adapted from "Provenance-Grounded Gating and Adaptive Recovery in Synthetic
Post-Training Data Curation" (arXiv:2606.11127). The paper shows that a curation
gate which permanently discards rejected synthetic samples leaves yield on the
table: many rejections are *near-misses* whose failure can be diagnosed from the
same source evidence that drove the gate, and recovered rather than thrown away.

``OCRScoringQAStage`` currently discards an entire image when no bounding box
clears its quality gate (``bbox_match >= min_bbox_match`` and
``text_errors <= max_text_errors``). This module supplies the two ideas from the
paper that close that discard-vs-recover gap, both grounded in the per-bbox
verifier scores already attached to each ``OCRDenseItem`` (its provenance):

* **Failure diagnosis** — classify *why* each rejected bbox failed, so geometry
  near-misses are distinguished from genuine content errors and ungrounded
  (unscored) entries.
* **Adaptive recovery** — re-admit only the bboxes diagnosed as recoverable
  geometry near-misses, under a bounded ``recovery_margin`` relaxation of the
  gate, and record what was recovered as provenance.

The full paper additionally regenerates rejected samples with a generator model;
that targeted-regeneration loop needs an extra inference call per sample and is
intentionally out of scope here. This module delivers the yield-increasing
result with a deterministic relaxation over the evidence already on hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_curator.tasks.ocr import OCRDenseItem

# Failure categories assigned by :func:`diagnose_bbox`. Only ``NEAR_MISS_BBOX`` is
# considered recoverable — the others reflect missing provenance or genuine
# content/geometry failures that relaxing the geometry gate must not paper over.
UNSCORED = "unscored"  # verifier returned no bbox_match — no source-grounded signal to trust
TEXT_ERRORS = "text_errors"  # transcription is wrong; a geometry relaxation cannot fix content
NEAR_MISS_BBOX = "near_miss_bbox"  # bbox_match just below the gate, text clean — recoverable
LOW_BBOX_MATCH = "low_bbox_match"  # bbox_match far below the gate — geometry genuinely wrong


def diagnose_bbox(
    word: OCRDenseItem,
    *,
    min_bbox_match: int,
    max_text_errors: int,
    recovery_margin: int,
) -> str:
    """Diagnose why a single rejected bbox failed the quality gate.

    The diagnosis is grounded in the bbox's own verifier scores (its provenance):
    ``bbox_match`` and ``text_errors``. A bbox is a recoverable near-miss only
    when its transcription is clean and its geometry score sits within
    ``recovery_margin`` of ``min_bbox_match``.

    Args:
        word: The rejected dense item carrying its verifier scores.
        min_bbox_match: The strict gate's minimum ``bbox_match``.
        max_text_errors: The strict gate's maximum ``text_errors``.
        recovery_margin: How far below ``min_bbox_match`` still counts as a
            near-miss. ``0`` disables near-miss recovery entirely.

    Returns:
        One of the module-level category constants.
    """
    if word.bbox_match is None:
        return UNSCORED
    errors = word.text_errors if word.text_errors is not None else 0
    if errors > max_text_errors:
        return TEXT_ERRORS
    if word.bbox_match >= min_bbox_match - recovery_margin:
        return NEAR_MISS_BBOX
    return LOW_BBOX_MATCH


def diagnose_failures(
    words: list[OCRDenseItem],
    *,
    min_bbox_match: int,
    max_text_errors: int,
    recovery_margin: int,
) -> dict[str, int]:
    """Build a failure-category histogram over the currently-invalid bboxes.

    Only ``valid=False`` items are diagnosed; already-valid items passed the gate
    and are not part of the rejected population.
    """
    histogram: dict[str, int] = {}
    for word in words:
        if word.valid:
            continue
        category = diagnose_bbox(
            word,
            min_bbox_match=min_bbox_match,
            max_text_errors=max_text_errors,
            recovery_margin=recovery_margin,
        )
        histogram[category] = histogram.get(category, 0) + 1
    return histogram


@dataclass
class RecoveryResult:
    """Outcome of an adaptive-recovery attempt over a rejected sample.

    Attributes:
        recovered: The bboxes re-admitted as valid by the recovery pass (also
            mutated in place to ``valid=True``). Empty when nothing was salvageable.
        provenance: A JSON-serializable record of the diagnosis and the recovery
            decision, suitable for attaching to the task as ``ocr_recovery`` so
            downstream consumers can audit why a sample survived.
    """

    recovered: list[OCRDenseItem]
    provenance: dict


def attempt_recovery(
    words: list[OCRDenseItem],
    *,
    min_bbox_match: int,
    max_text_errors: int,
    recovery_margin: int = 2,
) -> RecoveryResult:
    """Adaptively recover near-miss bboxes from an otherwise-discarded sample.

    Invoked at the gate's discard site — when *no* bbox cleared the strict
    threshold. Rather than dropping the whole image, diagnose every rejected
    bbox and re-admit only those whose failure is a geometry near-miss
    (``NEAR_MISS_BBOX``): clean transcription, ``bbox_match`` within
    ``recovery_margin`` of the gate. Content errors, far-off geometry, and
    ungrounded (unscored) bboxes are left rejected.

    Recovered items are flipped to ``valid=True`` in place so the existing
    conversation builders, which re-filter on ``valid``, pick them up unchanged.

    Args:
        words: The full dense-item list (a mix of valid and invalid items).
        min_bbox_match: The strict gate's minimum ``bbox_match``.
        max_text_errors: The strict gate's maximum ``text_errors``.
        recovery_margin: Geometry relaxation budget; ``<= 0`` recovers nothing,
            preserving the naive-discard behavior.

    Returns:
        A :class:`RecoveryResult` with the re-admitted bboxes and a provenance record.
    """
    diagnosis = diagnose_failures(
        words,
        min_bbox_match=min_bbox_match,
        max_text_errors=max_text_errors,
        recovery_margin=recovery_margin,
    )

    recovered: list[OCRDenseItem] = []
    if recovery_margin > 0:
        for word in words:
            if word.valid:
                continue
            category = diagnose_bbox(
                word,
                min_bbox_match=min_bbox_match,
                max_text_errors=max_text_errors,
                recovery_margin=recovery_margin,
            )
            if category == NEAR_MISS_BBOX:
                word.valid = True
                recovered.append(word)

    provenance = {
        "strategy": "near_miss_geometry_relaxation",
        "recovery_margin": recovery_margin,
        "diagnosis": diagnosis,
        "rejected_count": sum(diagnosis.values()),
        "recovered_count": len(recovered),
        "recovered_bbox_match": [w.bbox_match for w in recovered],
    }
    return RecoveryResult(recovered=recovered, provenance=provenance)
