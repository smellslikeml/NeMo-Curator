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

"""Tests for provenance-grounded adaptive recovery in OCRScoringQAStage.

Half of these go through the existing stage (``OCRScoringQAStage.handle_response``)
to prove the recovery module is actually wired into the gate's discard site; the
rest cover the diagnosis/recovery helpers directly.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from nemo_curator.stages.synthetic.omni.ocr_quality_recovery import (
    LOW_BBOX_MATCH,
    NEAR_MISS_BBOX,
    TEXT_ERRORS,
    UNSCORED,
    attempt_recovery,
    diagnose_bbox,
)
from nemo_curator.stages.synthetic.omni.ocr_scoring_qa import OCRScoringQAStage
from nemo_curator.tasks.image import ImageSampleTask
from nemo_curator.tasks.ocr import OCRData, OCRDenseItem


def _make_word(bbox: list[int], text: str, *, valid: bool = False) -> OCRDenseItem:
    return OCRDenseItem(bbox_2d=bbox, text_content=text, valid=valid)


def _make_task(words: list[OCRDenseItem]) -> ImageSampleTask[OCRData]:
    return ImageSampleTask(
        dataset_name="test",
        data=OCRData(image_path=Path("test.jpg"), image_id="img_0", ocr_dense=words),
    )


def _make_stage(**kwargs: object) -> OCRScoringQAStage:
    with patch(
        "nemo_curator.stages.synthetic.omni.ocr_scoring_qa.NVInferenceClient",
        return_value=MagicMock(),
    ):
        return OCRScoringQAStage(**kwargs)


def _verifier_response(items: list[dict]) -> str:
    return json.dumps({"ocr_mode": "word", "text": items, "missing_text": []})


class TestDiagnoseBbox:
    """Each rejected bbox is bucketed from its own verifier provenance."""

    def test_unscored_when_no_bbox_match(self) -> None:
        word = _make_word([0, 0, 1, 1], "X")  # bbox_match left as None
        assert diagnose_bbox(word, min_bbox_match=5, max_text_errors=0, recovery_margin=2) == UNSCORED

    def test_text_errors_dominate_geometry(self) -> None:
        word = _make_word([0, 0, 1, 1], "X")
        word.bbox_match = 10
        word.text_errors = 3
        assert diagnose_bbox(word, min_bbox_match=5, max_text_errors=0, recovery_margin=2) == TEXT_ERRORS

    def test_near_miss_within_margin(self) -> None:
        word = _make_word([0, 0, 1, 1], "X")
        word.bbox_match = 4  # one below the gate of 5, inside a margin of 2
        word.text_errors = 0
        assert diagnose_bbox(word, min_bbox_match=5, max_text_errors=0, recovery_margin=2) == NEAR_MISS_BBOX

    def test_low_match_outside_margin(self) -> None:
        word = _make_word([0, 0, 1, 1], "X")
        word.bbox_match = 1
        word.text_errors = 0
        assert diagnose_bbox(word, min_bbox_match=5, max_text_errors=0, recovery_margin=2) == LOW_BBOX_MATCH


class TestAttemptRecovery:
    """Only near-miss geometry failures are salvaged; provenance is recorded."""

    def test_recovers_near_miss_and_flips_valid(self) -> None:
        near = _make_word([0, 0, 1, 1], "X")
        near.bbox_match, near.text_errors = 4, 0
        hopeless = _make_word([0, 0, 1, 1], "Y")
        hopeless.bbox_match, hopeless.text_errors = 0, 0
        result = attempt_recovery([near, hopeless], min_bbox_match=5, max_text_errors=0, recovery_margin=2)
        assert result.recovered == [near]
        assert near.valid is True
        assert hopeless.valid is False
        assert result.provenance["recovered_count"] == 1
        assert result.provenance["diagnosis"] == {NEAR_MISS_BBOX: 1, LOW_BBOX_MATCH: 1}

    def test_margin_zero_recovers_nothing(self) -> None:
        near = _make_word([0, 0, 1, 1], "X")
        near.bbox_match, near.text_errors = 4, 0
        result = attempt_recovery([near], min_bbox_match=5, max_text_errors=0, recovery_margin=0)
        assert result.recovered == []
        assert near.valid is False
        assert result.provenance["recovered_count"] == 0


class TestStageRecoveryWiring:
    """The stage's discard branch must defer to adaptive recovery before dropping an image."""

    def test_near_miss_image_is_recovered_instead_of_discarded(self) -> None:
        stage = _make_stage(min_bbox_match=5, dense_dump_prob=0.0)
        task = _make_task([_make_word([0, 0, 1, 1], "X")])
        # bbox_match=4 fails the strict gate but is a recoverable near-miss.
        result = stage.handle_response(task, _verifier_response([{"idx": 0, "bbox_match": 4, "text_errors": 0}]))
        assert result.data.is_valid is True
        assert result.data.conversation is not None
        assert result.data.ocr_recovery is not None
        assert result.data.ocr_recovery["recovered_count"] == 1

    def test_hopeless_image_still_discarded_but_carries_diagnosis(self) -> None:
        stage = _make_stage(min_bbox_match=5, dense_dump_prob=0.0)
        task = _make_task([_make_word([0, 0, 1, 1], "X")])
        result = stage.handle_response(task, _verifier_response([{"idx": 0, "bbox_match": 0, "text_errors": 0}]))
        assert result.data.is_valid is False
        assert "no bboxes passed quality threshold" in (result.data.error or "")
        assert result.data.ocr_recovery["recovered_count"] == 0
        assert result.data.ocr_recovery["diagnosis"] == {LOW_BBOX_MATCH: 1}

    def test_recovery_margin_zero_preserves_naive_discard(self) -> None:
        stage = _make_stage(min_bbox_match=5, recovery_margin=0, dense_dump_prob=0.0)
        task = _make_task([_make_word([0, 0, 1, 1], "X")])
        result = stage.handle_response(task, _verifier_response([{"idx": 0, "bbox_match": 4, "text_errors": 0}]))
        assert result.data.is_valid is False
        assert result.data.ocr_recovery["recovered_count"] == 0
