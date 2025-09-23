"""ONNX-based text search provider."""

import os
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnxruntime as ort
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

from ...common.app import app_dirs


# Mean Pooling - Take attention mask into account for correct averaging
def _mean_pooling(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    input_mask_expanded = np.expand_dims(attention_mask, -1) * np.ones_like(token_embeddings)
    return np.sum(token_embeddings * input_mask_expanded, axis=1) / np.clip(
        np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None
    )


def _get_default_local_dir() -> Path:
    return app_dirs.app_models_dir / "sentence-transformers" / "all-MiniLM-L6-v2"


DEFAULT_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_REQUIRED = [
    "tokenizer.json",
    "vocab.txt",
    "config.json",
    "modules.json",
    "onnx/model_quint8_avx2.onnx",
]


def _download_default_models() -> bool:
    from huggingface_hub import snapshot_download

    local_dir = _get_default_local_dir()
    before = all((local_dir / f).exists() for f in DEFAULT_REQUIRED)

    snapshot_download(
        repo_id=DEFAULT_REPO_ID,
        local_dir=local_dir,
        allow_patterns=DEFAULT_REQUIRED,
        force_download=False,
        revision="c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
    )
    after = all((local_dir / f).exists() for f in DEFAULT_REQUIRED)
    return (not before) and after


class ONNXEmbedder(BaseModel):
    """ONNX Text Embedder."""

    onnx_model_path: Path = Field(
        default_factory=lambda: _get_default_local_dir() / "onnx" / "model_quint8_avx2.onnx",
        description="Path to the ONNX model.",
    )
    tokenizer_path: Path = Field(default_factory=_get_default_local_dir, description="Path to the tokenizer.")

    @cached_property
    def session(self) -> ort.InferenceSession:
        """Session."""
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cpu_count = max(1, (os.cpu_count() or 4))
        options.intra_op_num_threads = max(1, cpu_count - 2)
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        return ort.InferenceSession(self.onnx_model_path, sess_options=options, providers=["CPUExecutionProvider"])

    @cached_property
    def tokenizer(self) -> Tokenizer:
        """Tokenizer."""
        tok = Tokenizer.from_file((self.tokenizer_path / "tokenizer.json").as_posix())
        max_len = 512
        tok.enable_truncation(max_length=max_len)
        # Pick a pad token/id that exists in the vocab (common aliases)
        for pad_tok in ("[PAD]", "<pad>", "<|pad|>", "</s>"):
            pad_id = tok.token_to_id(pad_tok)
            if pad_id is not None:
                tok.enable_padding(length=max_len, pad_id=pad_id, pad_token=pad_tok)
                break
        else:
            # Fallback: pad_id=0 (not ideal, but safe)
            tok.enable_padding(length=max_len, pad_id=0, pad_token="[PAD]")
        return tok

    def model_post_init(self, context: Any) -> None:
        """Post init."""
        if self.tokenizer_path == _get_default_local_dir():
            _download_default_models()
        assert self.session is not None
        assert self.tokenizer is not None
        self.embed(["test"])

    def _encode(self, text: str | list[str]) -> dict[str, np.ndarray]:
        """Encode text."""
        batch_text = [text] if isinstance(text, str) else text
        encs = self.tokenizer.encode_batch(batch_text)

        input_ids = np.array([e.ids for e in encs], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encs], dtype=np.int64)

        return cast(
            "dict[str, np.ndarray]",
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )

    def embed(self, text: list[str]) -> np.ndarray:
        """Embed text."""
        batch = self._encode(text)
        embs = self.session.run(
            None,
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "token_type_ids": batch.get("token_type_ids", np.zeros_like(batch["input_ids"])),
            },
        )[0]
        embs = cast("np.ndarray", embs)
        if len(embs.shape) == 1:
            embs = embs.reshape(1, -1)
        return _mean_pooling(embs, batch["attention_mask"])
