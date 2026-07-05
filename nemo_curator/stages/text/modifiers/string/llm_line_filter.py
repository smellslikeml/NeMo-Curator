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

from collections.abc import Callable, Iterable

from nemo_curator.stages.text.modifiers.doc_modifier import DocumentModifier


class LLMLineFilter(DocumentModifier):
    """Line-level document filtering judged by an external callable.

    Implements the line-level keep/drop mechanism of FinerWeb-10BT
    (arxiv:2501.07314): rather than keeping or dropping a whole document,
    each line is judged independently and only the low-quality lines
    (boilerplate, navigation, junk) are removed while the rest of the
    document is preserved.

    The judge is supplied by the caller. It is a callable that receives the
    document's lines (in order) and returns a per-line keep mask (``True`` to
    keep). Any per-line labeller plugs in here -- e.g. a DeBERTa classifier
    trained on the FinerWeb line labels, or a thin wrapper around a served LLM
    such as ``VLLMModel``. Model lifecycle and batching are intentionally left
    to the caller (mirroring ``LLMCleanupStage``); this modifier is purely the
    per-row keep/drop applied through the existing ``Modify`` stage.
    """

    def __init__(self, judge: Callable[[list[str]], Iterable[bool]]) -> None:
        """
        Args:
            judge (Callable[[list[str]], Iterable[bool]]): Receives the
                document's lines (in order) and returns an iterable of
                booleans, one per line, where ``True`` keeps the line and
                ``False`` drops it.
        """
        super().__init__()
        self._judge = judge

    def modify_document(self, text: str) -> str:
        lines = text.split("\n")
        mask = list(self._judge(lines))
        # A wrong-arity mask is treated as "no decision" and left untouched.
        # An all-drop mask keeps the document rather than collapsing it to "".
        if len(mask) != len(lines) or not any(mask):
            return text
        return "\n".join(line for line, keep in zip(lines, mask, strict=True) if keep)
