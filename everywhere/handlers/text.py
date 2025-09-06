"""Text embedder."""

from functools import cached_property
from pathlib import Path
from typing import Self, cast

import numpy as np
from onnxruntime import InferenceSession
from pydantic import BaseModel, Field, model_validator
from transformers import AutoTokenizer, PreTrainedTokenizerFast


# Mean Pooling - Take attention mask into account for correct averaging
def _mean_pooling(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    input_mask_expanded = np.expand_dims(attention_mask, -1) * np.ones_like(token_embeddings)
    return np.sum(token_embeddings * input_mask_expanded, axis=1) / np.clip(
        np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None
    )


class ONNXTextEmbedder(BaseModel):
    """ONNX Text Embedder."""

    onnx_model_path: Path = Field(description="Path to the ONNX model.")
    tokenizer_path: Path = Field(description="Path to the tokenizer.")

    @model_validator(mode="after")
    def validate_embedder(self) -> Self:
        """Validate the embedder."""
        assert isinstance(self.embed(["test"]), np.ndarray)
        return self

    @cached_property
    def session(self) -> InferenceSession:
        """Session."""
        return InferenceSession(self.onnx_model_path)

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        """Tokenizer."""
        return AutoTokenizer.from_pretrained(self.tokenizer_path)

    def embed(self, text: list[str]) -> np.ndarray:
        """Embed text."""
        batch = self.tokenizer(text, padding=True, truncation=True, return_tensors="np")
        embs = self.session.run(
            None,
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "token_type_ids": batch["token_type_ids"],
            },
        )[0]
        return _mean_pooling(cast("np.ndarray", embs), batch.attention_mask)
