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

import os
from dataclasses import dataclass

os.environ["RAPIDS_NO_INITIALIZE"] = "1"

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.text.filters import Filter
from nemo_curator.stages.text.models.model import ModelStage
from nemo_curator.stages.text.models.tokenizer import TokenizerStage
from nemo_curator.stages.text.models.utils import ATTENTION_MASK_FIELD, INPUT_ID_FIELD
from nemo_curator.tasks import DocumentBatch

from .utils import DEBERTA_TOKENIZER_PADDING_SIDE

FINERWEB_MODEL_IDENTIFIER = "TurkuNLP/finerweb-quality-classifier"
MAX_SEQ_LENGTH = 512

# The released checkpoint only carries generic LABEL_0..LABEL_8 ids in its
# config.json.  The reference training pipeline (TurkuNLP/finerweb-10bt,
# classifier/run.py) encodes the nine categories of arXiv:2501.07314 Table 1
# with sklearn's LabelEncoder, which assigns ids in alphabetical order of the
# label strings; sorting the category names alphabetically therefore recovers
# the head layout, placing Clean at index 1.
FINERWEB_LINE_LABELS = [
    "Bibliographical_and_Citation_References",  # LABEL_0
    "Clean",  # LABEL_1
    "Contact_and_Identification_Information",  # LABEL_2
    "Formatting_Style_and_Errors",  # LABEL_3
    "Legal_and_Administrative_Content",  # LABEL_4
    "Navigation_and_Interface_Elements",  # LABEL_5
    "Offensive_or_Inappropriate_Content",  # LABEL_6
    "Promotional_and_Spam_Content",  # LABEL_7
    "Technical_Specifications_and_Metadata",  # LABEL_8
]

DOC_ID_FIELD = "_curator_finerweb_doc_id"
LINE_ID_FIELD = "_curator_finerweb_line_id"
DOC_TEXT_FIELD = "_curator_finerweb_doc_text"


class SplitLinesStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    Stage that splits each document into newline-delimited lines.

    Every output row is one line of one input document, carrying the original
    document text alongside the (doc_id, line_id) pair needed to reassemble it,
    so downstream per-line predictions can be aggregated back to documents.

    Args:
        text_field: The name of the text field in the input data. Defaults to "text".

    """

    def __init__(self, text_field: str = "text"):
        self.name = "finerweb_split_lines"
        self.text_field = text_field

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field, DOC_ID_FIELD, LINE_ID_FIELD, DOC_TEXT_FIELD]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas().reset_index(drop=True)

        lines = df[self.text_field].str.split("\n")
        num_lines = lines.map(len)

        output = df.copy()
        output[DOC_TEXT_FIELD] = df[self.text_field].to_numpy()
        output[DOC_ID_FIELD] = [[doc_id] * n for doc_id, n in enumerate(num_lines)]
        output[LINE_ID_FIELD] = [list(range(n)) for n in num_lines]
        output[self.text_field] = lines
        output = output.explode([self.text_field, DOC_ID_FIELD, LINE_ID_FIELD], ignore_index=True)

        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=output,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


class FinerWebModelStage(ModelStage):
    """
    Stage for the FinerWeb line-quality model (DeBERTa-v3-base, 9-way head).

    Args:
        model_identifier: The identifier of the Hugging Face model.
        cache_dir: The Hugging Face cache directory. Defaults to None.
        labels: The nine category names indexed by the model's output positions.
        clean_label: The entry of `labels` that marks a line as high quality.
        line_score_field: The name of the per-line quality score column.
        line_label_field: The name of the per-line category column.
        model_inference_batch_size: The size of the batch for model inference. Defaults to 256.
        has_seq_order: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.
        keep_tokens: Whether to keep the input tokens in the output dataframe. Defaults to False.

    """

    def __init__(  # noqa: PLR0913
        self,
        model_identifier: str,
        cache_dir: str | None = None,
        labels: list[str] | None = None,
        clean_label: str = "Clean",
        line_score_field: str = "finerweb-line-quality-score",
        line_label_field: str = "finerweb-line-quality-category",
        model_inference_batch_size: int = 256,
        has_seq_order: bool = True,
        autocast: bool = True,
        keep_tokens: bool = False,
    ):
        super().__init__(
            model_identifier=model_identifier,
            cache_dir=cache_dir,
            has_seq_order=has_seq_order,
            model_inference_batch_size=model_inference_batch_size,
            padding_side=DEBERTA_TOKENIZER_PADDING_SIDE,
            unpack_inference_batch=True,
            autocast=autocast,
        )

        self.labels = list(labels) if labels is not None else list(FINERWEB_LINE_LABELS)
        self.clean_label = clean_label
        self.clean_index = self.labels.index(clean_label)
        self.line_score_field = line_score_field
        self.line_label_field = line_label_field
        self.keep_tokens = keep_tokens

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.line_score_field, self.line_label_field]

    @staticmethod
    def configure_forward(model: torch.nn.Module) -> torch.nn.Module:
        original_forward = model.forward

        @torch.no_grad()
        def custom_forward(*args, **kwargs) -> torch.Tensor:
            output = original_forward(*args, **kwargs)
            del args, kwargs
            # Unlike the single-head FineWeb-Edu regressor, the 9-way
            # classification head returns (batch, 9) logits that must stay intact.
            return output.logits.float()

        model.forward = custom_forward
        return model

    def _setup(self, local_files_only: bool = True) -> None:
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_identifier,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
        ).cuda()
        self.model = self.configure_forward(model)

    def process_model_output(
        self, outputs: torch.Tensor, _: dict[str, torch.Tensor] | None = None
    ) -> dict[str, np.ndarray]:
        probs = torch.softmax(outputs.float(), dim=-1).cpu().numpy()

        line_scores = probs[:, self.clean_index]
        line_labels = np.array([self.labels[idx] for idx in probs.argmax(axis=1)])

        return {
            self.line_score_field: line_scores,
            self.line_label_field: line_labels,
        }

    def create_output_dataframe(self, df_cpu: pd.DataFrame, collected_output: dict[str, np.ndarray]) -> pd.DataFrame:
        if not self.keep_tokens:
            df_cpu = df_cpu.drop(columns=[INPUT_ID_FIELD, ATTENTION_MASK_FIELD])

        df_cpu[self.line_score_field] = collected_output[self.line_score_field]
        df_cpu[self.line_label_field] = collected_output[self.line_label_field]

        return df_cpu


class AggregateLinesStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    Stage that rejoins per-line predictions into per-document fields.

    A line is considered clean when its score >= clean_score_threshold. The
    document keeps its original text unless drop_low_quality is True, in which
    case the low-quality lines are removed from it.

    Args:
        text_field: The name of the text field in the input data. Defaults to "text".
        doc_id_field: The column identifying the document each line came from.
        line_id_field: The column identifying a line's position within its document.
        doc_text_field: The column holding each document's original text.
        line_score_field: The per-line quality score column to aggregate.
        line_label_field: The per-line category column to aggregate.
        float_score_field: The name of the output fraction-of-clean-lines column.
        label_field: The name of the output document label column.
        line_quality_field: The name of the output per-line score list column.
        line_category_field: The name of the output per-line category list column.
        clean_score_threshold: Minimum per-line score for a line to count as clean. Defaults to 0.5.
        drop_low_quality: Whether to remove low-quality lines from the text. Defaults to False.

    """

    def __init__(  # noqa: PLR0913
        self,
        text_field: str = "text",
        doc_id_field: str = DOC_ID_FIELD,
        line_id_field: str = LINE_ID_FIELD,
        doc_text_field: str = DOC_TEXT_FIELD,
        line_score_field: str = "finerweb-line-quality-score",
        line_label_field: str = "finerweb-line-quality-category",
        float_score_field: str = "finerweb-line-quality-float",
        label_field: str = "finerweb-line-quality-label",
        line_quality_field: str = "finerweb-line-quality",
        line_category_field: str = "finerweb-line-categories",
        clean_score_threshold: float = 0.5,
        drop_low_quality: bool = False,
    ):
        self.name = "finerweb_aggregate_lines"
        self.text_field = text_field
        self.doc_id_field = doc_id_field
        self.line_id_field = line_id_field
        self.doc_text_field = doc_text_field
        self.line_score_field = line_score_field
        self.line_label_field = line_label_field
        self.float_score_field = float_score_field
        self.label_field = label_field
        self.line_quality_field = line_quality_field
        self.line_category_field = line_category_field
        self.clean_score_threshold = clean_score_threshold
        self.drop_low_quality = drop_low_quality

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            self.doc_id_field,
            self.line_id_field,
            self.doc_text_field,
            self.line_score_field,
            self.line_label_field,
        ]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            self.text_field,
            self.float_score_field,
            self.label_field,
            self.line_quality_field,
            self.line_category_field,
        ]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        df = df.sort_values([self.doc_id_field, self.line_id_field], kind="stable", ignore_index=True)

        threshold = self.clean_score_threshold
        by_doc = df.groupby(self.doc_id_field, sort=False)
        line_scores = by_doc[self.line_score_field].agg(list)
        line_labels = by_doc[self.line_label_field].agg(list)
        clean_fraction = by_doc[self.line_score_field].agg(lambda scores: float((scores >= threshold).mean()))

        result = df.drop_duplicates(subset=self.doc_id_field, keep="first", ignore_index=True).copy()
        result[self.float_score_field] = clean_fraction.to_numpy()
        result[self.label_field] = [
            "high_quality" if fraction >= threshold else "low_quality" for fraction in clean_fraction
        ]
        result[self.line_quality_field] = line_scores.to_numpy()
        result[self.line_category_field] = line_labels.to_numpy()

        if self.drop_low_quality:
            kept = df[df[self.line_score_field] >= threshold]
            filtered_text = kept.groupby(self.doc_id_field, sort=False)[self.text_field].agg("\n".join)
            result[self.text_field] = filtered_text.reindex(result[self.doc_id_field], fill_value="").to_numpy()
        else:
            result[self.text_field] = result[self.doc_text_field].to_numpy()

        line_columns = [
            self.doc_id_field,
            self.line_id_field,
            self.doc_text_field,
            self.line_score_field,
            self.line_label_field,
        ]
        result = result.drop(columns=line_columns)

        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=result,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


@dataclass(kw_only=True)
class FinerWebQualityClassifier(CompositeStage[DocumentBatch, DocumentBatch]):
    """
    FinerWebQualityClassifier scores web text line by line with the FinerWeb quality
    classifier (https://huggingface.co/TurkuNLP/finerweb-quality-classifier), a
    DeBERTa-v3-base checkpoint released with FinerWeb-10BT ("FinerWeb-10BT: Refining
    Web Data with LLM-Based Line-Level Filtering", arXiv:2501.07314). This is an
    integration of the released Apache-2.0 checkpoint, not a reimplementation.

    Unlike the document-level FineWebEduClassifier and QualityClassifier, each
    document is split into newline-delimited lines, every line is classified as
    Clean or one of eight low-quality categories, and the per-line predictions are
    aggregated back to the document. The classifier is optimized for running on
    multi-node, multi-GPU setups to enable fast and efficient inference on large
    text datasets.

    Per document, the classifier writes:

    - ``line_quality_field``: one quality score in [0, 1] per line (1 = clean),
      mirroring the ``line_quality`` column of the TurkuNLP/finerweb-10bt dataset
    - ``line_category_field``: the predicted category of each line
    - ``float_score_field``: the fraction of lines whose score >= clean_score_threshold
    - ``label_field``: "high_quality" when that fraction >= clean_score_threshold,
      "low_quality" otherwise
    - ``text_field``: the original text, or the text with low-quality lines removed
      when drop_low_quality is True

    The per-line quality score is the softmax probability of the Clean class. The
    reference pipeline Platt-calibrates this probability with platt_scaler.joblib
    shipped in the checkpoint repository; both scores are monotone in the Clean
    logit so lines rank identically, but the paper's thresholds (0.5 and 0.9) were
    tuned on calibrated scores and do not transfer directly to the uncalibrated
    scores exposed here.

    The checkpoint's config.json only exposes generic LABEL_0..LABEL_8 ids. The
    default `labels` list follows the reference training pipeline (sklearn
    LabelEncoder over the nine categories of arXiv:2501.07314 Table 1, which
    numbers labels alphabetically), placing Clean at index 1; pass `labels` and
    `clean_label` to override the mapping.

    Attributes:
        model_identifier: The identifier of the Hugging Face model.
            Defaults to "TurkuNLP/finerweb-quality-classifier".
        cache_dir: The Hugging Face cache directory. Defaults to None.
        text_field: The name of the text field in the input data. Defaults to "text".
        filter_by: The list of document labels to filter the data by
            ("high_quality" / "low_quality"). Defaults to None.
        label_field: The name of the document label column.
            Defaults to "finerweb-line-quality-label".
        float_score_field: The name of the fraction-of-clean-lines column.
            Defaults to "finerweb-line-quality-float".
        line_quality_field: The name of the per-line score list column.
            Defaults to "finerweb-line-quality".
        line_category_field: The name of the per-line category list column.
            Defaults to "finerweb-line-categories".
        labels: The nine category names indexed by the model's output positions.
            Defaults to the LabelEncoder-derived FINERWEB_LINE_LABELS order.
        clean_label: The entry of `labels` that marks a line as high quality. Defaults to "Clean".
        clean_score_threshold: Minimum per-line score for a line to count as clean. Defaults to 0.5.
        drop_low_quality: Whether to remove low-quality lines from the text. Defaults to False.
        max_chars: Limits the total number of characters of each line that can be
            fed to the tokenizer. If None, lines will not be truncated. Defaults to None.
        max_seq_length: Limits the total sequence returned by the tokenizer so that it
            has a maximum length. Defaults to 512.
        sort_by_length: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        model_inference_batch_size: The size of the batch for model inference. Defaults to 256.
        autocast: Whether to use autocast. When True, we trade off minor accuracy for
            faster inference. Defaults to True.
        keep_tokens: Whether to keep the input tokens in the output dataframe. Defaults to False.

    """

    model_identifier: str = FINERWEB_MODEL_IDENTIFIER
    cache_dir: str | None = None
    text_field: str = "text"
    filter_by: list[str] | None = None
    label_field: str = "finerweb-line-quality-label"
    float_score_field: str = "finerweb-line-quality-float"
    line_quality_field: str = "finerweb-line-quality"
    line_category_field: str = "finerweb-line-categories"
    labels: list[str] | None = None
    clean_label: str = "Clean"
    clean_score_threshold: float = 0.5
    drop_low_quality: bool = False
    max_chars: int | None = None
    max_seq_length: int = MAX_SEQ_LENGTH
    sort_by_length: bool = True
    model_inference_batch_size: int = 256
    autocast: bool = True
    keep_tokens: bool = False

    def __post_init__(self) -> None:
        super().__init__()

        self.labels = list(self.labels) if self.labels is not None else list(FINERWEB_LINE_LABELS)
        if len(self.labels) != len(FINERWEB_LINE_LABELS):
            msg = (
                f"FinerWebQualityClassifier expects {len(FINERWEB_LINE_LABELS)} line labels, "
                f"got {len(self.labels)}: {self.labels}"
            )
            raise ValueError(msg)
        if self.clean_label not in self.labels:
            msg = f"clean_label {self.clean_label!r} is not one of the configured labels: {self.labels}"
            raise ValueError(msg)
        if not 0 <= self.clean_score_threshold <= 1:
            msg = f"clean_score_threshold must be in [0, 1], got {self.clean_score_threshold}"
            raise ValueError(msg)

        self.stages = []

        self.stages.append(SplitLinesStage(text_field=self.text_field))

        self.stages.append(
            TokenizerStage(
                model_identifier=self.model_identifier,
                cache_dir=self.cache_dir,
                text_field=self.text_field,
                max_chars=self.max_chars,
                max_seq_length=self.max_seq_length,
                padding_side=DEBERTA_TOKENIZER_PADDING_SIDE,
                sort_by_length=self.sort_by_length,
            )
        )

        self.stages.append(
            FinerWebModelStage(
                model_identifier=self.model_identifier,
                cache_dir=self.cache_dir,
                labels=self.labels,
                clean_label=self.clean_label,
                model_inference_batch_size=self.model_inference_batch_size,
                has_seq_order=self.sort_by_length,
                autocast=self.autocast,
                keep_tokens=self.keep_tokens,
            )
        )

        self.stages.append(
            AggregateLinesStage(
                text_field=self.text_field,
                float_score_field=self.float_score_field,
                label_field=self.label_field,
                line_quality_field=self.line_quality_field,
                line_category_field=self.line_category_field,
                clean_score_threshold=self.clean_score_threshold,
                drop_low_quality=self.drop_low_quality,
            )
        )

        if self.filter_by is not None and len(self.filter_by) > 0:
            self.stages.append(Filter(filter_fn=self.filter_by_category, filter_field=self.label_field))

        self.name = "finerweb_quality_classifier"

    def inputs(self) -> tuple[list[str], list[str]]:
        return self.stages[0].inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        return self.stages[-1].outputs()

    def filter_by_category(self, value: str) -> bool:
        return value in self.filter_by

    def decompose(self) -> list[ProcessingStage]:
        return self.stages
