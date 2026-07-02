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

"""Refine LLM text embeddings by filtering out a dominant, low-information subspace.

This implements the practical result of *"Your UnEmbedding Matrix is Secretly a
Feature Lens for Text Embeddings"* (EmbedFilter, https://arxiv.org/abs/2606.07502):
text embeddings derived from LLM-style backbones over-express a handful of
directions that correlate with frequent-but-uninformative tokens, which suppresses
the nuanced semantics that downstream clustering and retrieval depend on. Removing
that subspace via a single linear transformation sharpens the embeddings and, as a
byproduct, lowers their dimensionality (smaller indexes, faster similarity search).

The paper derives the removed subspace from the backbone's unembedding matrix, which
is a model-only ("data-independent") lens. A curation pipeline only ever sees the
emitted embedding vectors -- not the backbone's unembedding matrix, and the recipe's
SentenceTransformer path does not expose it uniformly -- so :class:`EmbeddingSubspaceFilter`
estimates the same dominant, frequency-aligned subspace directly from a representative
sample of embeddings via PCA (the classic "all-but-the-top" estimator). The fitted
filter is a fixed ``(mean, projection)`` pair applied identically to every shard, so
the refined embedding space stays consistent across the dataset for downstream K-Means.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch

_NORM_EPS = 1e-12


@dataclass
class EmbeddingSubspaceFilter:
    """A fixed linear transform that removes a dominant subspace and reduces dimension.

    The transform applied to a row vector ``x`` is ``(x - mean) @ projection``, optionally
    L2-normalized. ``projection`` has shape ``(input_dim, output_dim)`` and its columns are
    the principal directions of a fitted embedding sample *after* dropping the leading
    (frequency-aligned) directions.

    Args:
        mean (np.ndarray): Per-dimension mean of the fitted sample, shape ``(input_dim,)``.
        projection (np.ndarray): Kept principal directions, shape ``(input_dim, output_dim)``.
    """

    mean: np.ndarray
    projection: np.ndarray

    @property
    def input_dim(self) -> int:
        return int(self.projection.shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.projection.shape[1])

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray | list[list[float]],
        num_components_to_remove: int = 1,
        output_dim: int | None = None,
    ) -> EmbeddingSubspaceFilter:
        """Fit a filter from a representative sample of embeddings.

        Args:
            embeddings: Sample embeddings, shape ``(num_samples, input_dim)``.
            num_components_to_remove: Number of leading principal directions to drop. These
                are the dominant, frequency-aligned directions the paper identifies as
                suppressing semantics.
            output_dim: Number of kept directions (the refined dimensionality). When ``None``,
                every direction after the removed ones is kept.

        Returns:
            EmbeddingSubspaceFilter: The fitted transform.
        """
        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim != 2:  # noqa: PLR2004
            msg = f"Expected a 2D array of embeddings, got shape {x.shape}"
            raise ValueError(msg)
        if num_components_to_remove < 0:
            msg = f"num_components_to_remove must be non-negative, got {num_components_to_remove}"
            raise ValueError(msg)

        mean = x.mean(axis=0)
        centered = x - mean
        # Rows of vh are principal directions ordered by decreasing variance.
        _, _, vh = np.linalg.svd(centered, full_matrices=False)

        if num_components_to_remove >= vh.shape[0]:
            msg = (
                f"num_components_to_remove ({num_components_to_remove}) must be smaller than the "
                f"number of available components ({vh.shape[0]})"
            )
            raise ValueError(msg)

        kept = vh[num_components_to_remove:]
        if output_dim is not None:
            if output_dim < 1 or output_dim > kept.shape[0]:
                msg = f"output_dim must be in [1, {kept.shape[0]}], got {output_dim}"
                raise ValueError(msg)
            kept = kept[:output_dim]

        return cls(mean=mean.astype(np.float32), projection=kept.T.astype(np.float32))

    def transform(self, embeddings: np.ndarray | list[list[float]], normalize: bool = True) -> np.ndarray:
        """Apply the filter to a batch of embeddings.

        Args:
            embeddings: Embeddings to refine, shape ``(num_rows, input_dim)``.
            normalize: Whether to L2-normalize each refined embedding.

        Returns:
            np.ndarray: Refined embeddings, shape ``(num_rows, output_dim)``.
        """
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2:  # noqa: PLR2004
            msg = f"Expected a 2D array of embeddings, got shape {x.shape}"
            raise ValueError(msg)
        if x.shape[1] != self.input_dim:
            msg = f"Embedding dimension {x.shape[1]} does not match filter input dimension {self.input_dim}"
            raise ValueError(msg)

        refined = (x - self.mean) @ self.projection
        if normalize:
            norms = np.linalg.norm(refined, axis=-1, keepdims=True)
            refined = refined / np.clip(norms, _NORM_EPS, None)
        return refined

    def save(self, path: str) -> None:
        """Serialize the filter to a ``.npz`` file."""
        np.savez(path, mean=self.mean, projection=self.projection)

    @classmethod
    def load(cls, path: str) -> EmbeddingSubspaceFilter:
        """Load a filter previously written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as data:
            return cls(mean=data["mean"], projection=data["projection"])


@dataclass
class EmbeddingRefinerStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Refine an embedding column in place with a fitted :class:`EmbeddingSubspaceFilter`.

    Slots between the embedding stage and downstream consumers (e.g. K-Means clustering)
    in a curation pipeline: it reads ``embedding_field``, applies the fixed linear filter,
    and writes the refined (and typically lower-dimensional) embeddings back to the same
    column so no downstream code needs to change.

    Either ``filter_path`` (a ``.npz`` written by :meth:`EmbeddingSubspaceFilter.save`) or an
    in-memory ``embedding_filter`` must be provided.

    Args:
        embedding_field (str): Column holding the embeddings to refine.
        filter_path (str | None): Path to a serialized filter, loaded once per worker.
        embedding_filter (EmbeddingSubspaceFilter | None): An already-fitted filter.
        normalize (bool): Whether to L2-normalize refined embeddings.
    """

    embedding_field: str = "embeddings"
    filter_path: str | None = None
    embedding_filter: EmbeddingSubspaceFilter | None = None
    normalize: bool = True
    name: str = "embedding_refiner"

    def __post_init__(self) -> None:
        if self.filter_path is None and self.embedding_filter is None:
            msg = "EmbeddingRefinerStage requires either filter_path or embedding_filter"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.embedding_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.embedding_field]

    def setup(self, _: object | None = None) -> None:
        if self.embedding_filter is None:
            self.embedding_filter = EmbeddingSubspaceFilter.load(self.filter_path)

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        if self.embedding_filter is None:
            self.embedding_filter = EmbeddingSubspaceFilter.load(self.filter_path)

        df = batch.to_pandas().copy()
        embeddings = np.asarray(df[self.embedding_field].tolist(), dtype=np.float32)
        refined = self.embedding_filter.transform(embeddings, normalize=self.normalize)
        df[self.embedding_field] = refined.tolist()

        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
