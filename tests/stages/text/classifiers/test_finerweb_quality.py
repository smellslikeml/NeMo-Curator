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

import math

import pandas as pd
import pytest
import torch

from nemo_curator.stages.text.classifiers import FinerWebQualityClassifier
from nemo_curator.stages.text.classifiers.finerweb_quality import (
    DOC_ID_FIELD,
    DOC_TEXT_FIELD,
    FINERWEB_LINE_LABELS,
    FINERWEB_MODEL_IDENTIFIER,
    LINE_ID_FIELD,
    AggregateLinesStage,
    FinerWebModelStage,
    SplitLinesStage,
)
from nemo_curator.tasks import DocumentBatch

CACHE_DIR = "./hf_cache"
LINE_SCORE_FIELD = "finerweb-line-quality-score"
LINE_LABEL_FIELD = "finerweb-line-quality-category"

CLEAN_LABEL = "Clean"
LOW_QUALITY_LABEL = "Navigation_and_Interface_Elements"


def _line_level_batch(lines_per_doc: list[list[str]], scores_per_doc: list[list[float]]) -> DocumentBatch:
    """Build the per-line frame the aggregate stage consumes, in reversed row order."""
    rows = []
    for doc_id, (lines, scores) in enumerate(zip(lines_per_doc, scores_per_doc, strict=True)):
        for line_id, (line, score) in enumerate(zip(lines, scores, strict=True)):
            rows.append(
                {
                    "text": line,
                    "id": f"doc-{doc_id}",
                    DOC_ID_FIELD: doc_id,
                    LINE_ID_FIELD: line_id,
                    DOC_TEXT_FIELD: "\n".join(lines),
                    LINE_SCORE_FIELD: score,
                    LINE_LABEL_FIELD: CLEAN_LABEL if score >= 0.5 else LOW_QUALITY_LABEL,
                }
            )
    frame = pd.DataFrame(rows)
    return DocumentBatch(
        data=frame.iloc[::-1].reset_index(drop=True),
        dataset_name="test_1",
    )


def test_classifier_wires_up_stages() -> None:
    classifier = FinerWebQualityClassifier()

    stages = classifier.decompose()
    assert [stage.name for stage in stages] == [
        "finerweb_split_lines",
        "finerweb_quality_classifier_tokenizer",
        "finerweb_quality_classifier_model",
        "finerweb_aggregate_lines",
    ]

    # Every stage must find its declared inputs in the columns produced upstream
    available_columns = {"text", "id"}
    for stage in stages:
        assert set(stage.inputs()[1]) <= available_columns
        available_columns |= set(stage.outputs()[1])
    assert set(classifier.inputs()[1]) == {"text"}


def test_classifier_wires_up_filter_stage() -> None:
    classifier = FinerWebQualityClassifier(filter_by=["high_quality"])

    stages = classifier.decompose()
    assert stages[-1].name == "filter_fn"


def test_classifier_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="line labels"):
        FinerWebQualityClassifier(labels=["Clean", "Navigation_and_Interface_Elements"])

    with pytest.raises(ValueError, match="clean_label"):
        FinerWebQualityClassifier(clean_label="High_Quality")

    with pytest.raises(ValueError, match="clean_score_threshold"):
        FinerWebQualityClassifier(clean_score_threshold=1.5)


def test_default_labels_place_clean_at_index_one() -> None:
    # The reference training pipeline numbers the nine categories alphabetically
    # (sklearn LabelEncoder), which places Clean at index 1 of the model head.
    assert len(FINERWEB_LINE_LABELS) == 9
    assert FINERWEB_LINE_LABELS.index("Clean") == 1


def test_split_lines_stage() -> None:
    text = ["alpha\nbeta\n\ngamma\n", "single"]
    batch = DocumentBatch(data=pd.DataFrame({"text": text, "id": ["a", "b"]}), dataset_name="test_1")

    result = SplitLinesStage().process(batch).to_pandas()

    assert result["text"].tolist() == ["alpha", "beta", "", "gamma", "", "single"]
    assert result[DOC_ID_FIELD].tolist() == [0, 0, 0, 0, 0, 1]
    assert result[LINE_ID_FIELD].tolist() == [0, 1, 2, 3, 4, 0]
    assert result[DOC_TEXT_FIELD].tolist() == [text[0]] * 5 + [text[1]]
    assert result["id"].tolist() == ["a"] * 5 + ["b"]


def test_model_stage_scores_and_labels_lines() -> None:
    stage = FinerWebModelStage(model_identifier=FINERWEB_MODEL_IDENTIFIER)

    logits = torch.full((3, 9), -5.0)
    logits[0, FINERWEB_LINE_LABELS.index(CLEAN_LABEL)] = 5.0
    logits[1, FINERWEB_LINE_LABELS.index("Promotional_and_Spam_Content")] = 5.0
    logits[2, FINERWEB_LINE_LABELS.index(LOW_QUALITY_LABEL)] = 5.0

    result = stage.process_model_output(logits)

    assert result[LINE_LABEL_FIELD].tolist() == [CLEAN_LABEL, "Promotional_and_Spam_Content", LOW_QUALITY_LABEL]

    # With one dominating logit, the clean probability is e^10 / (e^10 + 8)
    expected_clean = math.exp(10) / (math.exp(10) + 8)
    assert result[LINE_SCORE_FIELD][0] == pytest.approx(expected_clean)
    assert result[LINE_SCORE_FIELD][1] < 1e-3
    assert result[LINE_SCORE_FIELD][2] < 1e-3

    # A custom label list moves the clean index with it
    custom_labels = list(reversed(FINERWEB_LINE_LABELS))
    custom_stage = FinerWebModelStage(
        model_identifier=FINERWEB_MODEL_IDENTIFIER,
        labels=custom_labels,
        clean_label=CLEAN_LABEL,
    )
    custom_logits = torch.full((1, 9), -5.0)
    custom_logits[0, custom_labels.index(CLEAN_LABEL)] = 5.0

    custom_result = custom_stage.process_model_output(custom_logits)

    assert custom_result[LINE_LABEL_FIELD].tolist() == [CLEAN_LABEL]
    assert custom_result[LINE_SCORE_FIELD][0] == pytest.approx(expected_clean)


def test_aggregate_lines_stage_flag_mode() -> None:
    lines_per_doc = [
        ["The telescope observed the comet.", "Subscribe to our newsletter!", "The comet brightened overnight."],
        ["Follow us on social media."],
    ]
    scores_per_doc = [[0.9, 0.2, 0.8], [0.1]]
    batch = _line_level_batch(lines_per_doc, scores_per_doc)

    result = AggregateLinesStage().process(batch).to_pandas()

    assert result["id"].tolist() == ["doc-0", "doc-1"]
    assert result["text"].tolist() == ["\n".join(lines_per_doc[0]), "\n".join(lines_per_doc[1])]
    assert result["finerweb-line-quality"].tolist() == scores_per_doc
    assert result["finerweb-line-categories"].tolist() == [
        [CLEAN_LABEL, LOW_QUALITY_LABEL, CLEAN_LABEL],
        [LOW_QUALITY_LABEL],
    ]
    assert result["finerweb-line-quality-float"].tolist() == pytest.approx([2 / 3, 0.0])
    assert result["finerweb-line-quality-label"].tolist() == ["high_quality", "low_quality"]

    for column in [DOC_ID_FIELD, LINE_ID_FIELD, DOC_TEXT_FIELD, LINE_SCORE_FIELD, LINE_LABEL_FIELD]:
        assert column not in result.columns


def test_aggregate_lines_stage_drop_mode() -> None:
    lines_per_doc = [
        ["The telescope observed the comet.", "Subscribe to our newsletter!", "The comet brightened overnight."],
        ["Follow us on social media."],
    ]
    scores_per_doc = [[0.9, 0.2, 0.8], [0.1]]
    batch = _line_level_batch(lines_per_doc, scores_per_doc)

    stage = AggregateLinesStage(drop_low_quality=True)
    result = stage.process(batch).to_pandas()

    assert result["text"].tolist() == ["The telescope observed the comet.\nThe comet brightened overnight.", ""]
    assert result["finerweb-line-quality-float"].tolist() == pytest.approx([2 / 3, 0.0])


def test_aggregate_lines_stage_respects_custom_threshold() -> None:
    lines_per_doc = [["Clean prose line."]]
    scores_per_doc = [[0.7]]
    batch = _line_level_batch(lines_per_doc, scores_per_doc)

    result = AggregateLinesStage(clean_score_threshold=0.9).process(batch).to_pandas()

    assert result["finerweb-line-quality-float"].tolist() == [0.0]
    assert result["finerweb-line-quality-label"].tolist() == ["low_quality"]
    assert result["text"].tolist() == ["Clean prose line."]


def test_filter_keeps_high_quality_documents() -> None:
    lines_per_doc = [["Mostly clean document.", "One noisy line."], ["All junk here."]]
    scores_per_doc = [[0.9, 0.1], [0.2]]
    classifier = FinerWebQualityClassifier(filter_by=["high_quality"])

    stages = classifier.decompose()
    aggregated = stages[3].process(_line_level_batch(lines_per_doc, scores_per_doc))
    filtered = stages[4].process(aggregated)

    assert filtered.to_pandas()["id"].tolist() == ["doc-0"]
    assert filtered.to_pandas()["finerweb-line-quality-label"].tolist() == ["high_quality"]


@pytest.mark.gpu
@pytest.mark.parametrize("drop_low_quality", [False, True])
def test_finerweb_quality_classifier(drop_low_quality: bool) -> None:
    text = [
        "Quantum computing is set to revolutionize the field of cryptography."
        "\nSubscribe to our newsletter for more updates!"
        "\nAll rights reserved. Unauthorized reproduction is prohibited.",
        "Investing in index funds is a popular strategy for long-term financial growth.",
    ]
    df = pd.DataFrame({"text": text})
    input_dataset = DocumentBatch(
        data=df,
        dataset_name="test_1",
    )

    classifier = FinerWebQualityClassifier(cache_dir=CACHE_DIR, drop_low_quality=drop_low_quality)

    stages = classifier.decompose()

    # Check that the split stage inputs/output columns are correct
    split_stage = stages[0]
    assert all(col in input_dataset.data.columns for col in split_stage.inputs()[1])
    line_batch = split_stage.process(input_dataset)
    assert all(col in line_batch.data.columns for col in split_stage.outputs()[1])
    assert len(line_batch.data) == sum(doc.count("\n") + 1 for doc in text)

    # Check that the tokenizer stage inputs/output columns are correct
    tokenizer_stage = stages[1]
    assert all(col in line_batch.data.columns for col in tokenizer_stage.inputs()[1])
    tokenizer_stage.setup_on_node()
    tokenizer_stage.setup()
    tokenized_batch = tokenizer_stage.process(line_batch)
    assert all(col in tokenized_batch.data.columns for col in tokenizer_stage.outputs()[1])

    # Check that the model stage inputs/output columns are correct
    model_stage = stages[2]
    assert all(col in tokenized_batch.data.columns for col in model_stage.inputs()[1])
    model_stage.setup_on_node()
    model_stage.setup()
    scored_batch = model_stage.process(tokenized_batch)
    assert all(col in scored_batch.data.columns for col in model_stage.outputs()[1])
    assert len(scored_batch.data) == len(line_batch.data)

    # Teardown stages to release GPU memory
    model_stage.teardown()
    tokenizer_stage.teardown()

    # Check that the aggregate stage outputs one row per document
    aggregate_stage = stages[3]
    result = aggregate_stage.process(scored_batch).to_pandas()
    assert all(col in result.columns for col in classifier.outputs()[1])
    assert len(result) == 2

    for doc_id, doc in enumerate(text):
        lines = doc.split("\n")
        scores = result["finerweb-line-quality"][doc_id]

        assert len(scores) == len(lines)
        assert len(result["finerweb-line-categories"][doc_id]) == len(lines)
        assert all(0.0 <= score <= 1.0 for score in scores)

        expected_fraction = sum(score >= 0.5 for score in scores) / len(scores)
        expected_label = "high_quality" if expected_fraction >= 0.5 else "low_quality"
        assert result["finerweb-line-quality-float"][doc_id] == pytest.approx(expected_fraction)
        assert result["finerweb-line-quality-label"][doc_id] == expected_label

        if drop_low_quality:
            expected_text = "\n".join(line for line, score in zip(lines, scores, strict=True) if score >= 0.5)
            assert result["text"][doc_id] == expected_text
        else:
            assert result["text"][doc_id] == doc
