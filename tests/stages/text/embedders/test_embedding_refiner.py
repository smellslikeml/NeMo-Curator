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

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nemo_curator.stages.text.embedders.embedding_refiner import (
    EmbeddingRefinerStage,
    EmbeddingSubspaceFilter,
)
from nemo_curator.tasks import DocumentBatch


def _sample_embeddings(num_rows: int = 64, dim: int = 16, seed: int = 0) -> np.ndarray:
    """Embeddings with a single dominant, shared direction plus small semantic noise."""
    rng = np.random.default_rng(seed)
    dominant = np.zeros(dim, dtype=np.float32)
    dominant[0] = 1.0
    # Large shared component on dim 0 (mimics the high-frequency-token subspace) + small signal.
    coeffs = rng.normal(loc=5.0, scale=0.5, size=(num_rows, 1)).astype(np.float32)
    signal = rng.normal(scale=0.1, size=(num_rows, dim)).astype(np.float32)
    return coeffs * dominant + signal


class TestEmbeddingSubspaceFilter:
    def test_fit_reduces_dimension(self) -> None:
        embeddings = _sample_embeddings(dim=16)
        emb_filter = EmbeddingSubspaceFilter.fit(embeddings, num_components_to_remove=1, output_dim=4)
        assert emb_filter.input_dim == 16
        assert emb_filter.output_dim == 4

        refined = emb_filter.transform(embeddings)
        assert refined.shape == (embeddings.shape[0], 4)
        # Normalized rows have unit L2 norm.
        np.testing.assert_allclose(np.linalg.norm(refined, axis=1), 1.0, atol=1e-5)

    def test_dominant_direction_is_suppressed(self) -> None:
        embeddings = _sample_embeddings(dim=16)
        emb_filter = EmbeddingSubspaceFilter.fit(embeddings, num_components_to_remove=1)
        refined = emb_filter.transform(embeddings, normalize=False)
        # The shared dim-0 component dominates the raw variance but is removed after filtering.
        raw_var0 = float(np.var(embeddings[:, 0]))
        refined_max_var = float(np.max(np.var(refined, axis=0)))
        assert refined_max_var < raw_var0

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        embeddings = _sample_embeddings()
        emb_filter = EmbeddingSubspaceFilter.fit(embeddings, num_components_to_remove=2, output_dim=5)
        path = str(tmp_path / "filter.npz")
        emb_filter.save(path)
        loaded = EmbeddingSubspaceFilter.load(path)

        np.testing.assert_allclose(loaded.transform(embeddings), emb_filter.transform(embeddings))

    def test_fit_rejects_too_many_removed_components(self) -> None:
        embeddings = _sample_embeddings(num_rows=4, dim=4)
        with pytest.raises(ValueError, match="num_components_to_remove"):
            EmbeddingSubspaceFilter.fit(embeddings, num_components_to_remove=10)


class TestEmbeddingRefinerStage:
    def test_requires_a_filter(self) -> None:
        with pytest.raises(ValueError, match="filter_path or embedding_filter"):
            EmbeddingRefinerStage(embedding_field="embeddings")

    def test_process_refines_embedding_column(self) -> None:
        embeddings = _sample_embeddings(num_rows=32, dim=16)
        emb_filter = EmbeddingSubspaceFilter.fit(embeddings, num_components_to_remove=1, output_dim=4)

        # Build a DocumentBatch exactly as EmbeddingCreatorStage emits it: a list-of-lists column.
        df = pd.DataFrame(
            {
                "_curator_climb_id": [f"doc_{i}" for i in range(len(embeddings))],
                "text": ["lorem ipsum"] * len(embeddings),
                "embeddings": embeddings.tolist(),
            }
        )
        batch = DocumentBatch(dataset_name="climb", data=df)

        stage = EmbeddingRefinerStage(embedding_field="embeddings", embedding_filter=emb_filter)
        # The stage advertises the embedding column as both its input and (in-place) output.
        assert stage.inputs() == (["data"], ["embeddings"])
        assert stage.outputs() == (["data"], ["embeddings"])

        result = stage.process(batch)
        out_df = result.to_pandas()

        # Same rows, untouched companion columns, refined (lower-dimensional, normalized) embeddings.
        assert list(out_df["_curator_climb_id"]) == list(df["_curator_climb_id"])
        refined = np.asarray(out_df["embeddings"].tolist(), dtype=np.float32)
        assert refined.shape == (len(embeddings), 4)
        np.testing.assert_allclose(np.linalg.norm(refined, axis=1), 1.0, atol=1e-5)
