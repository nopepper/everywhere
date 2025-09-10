"""ONNX-based text search provider."""

import os
import threading
from functools import cached_property
from itertools import groupby
from pathlib import Path
from typing import cast

import fitz
import numpy as np
import onnxruntime as ort
from charset_normalizer import from_path
from markitdown import MarkItDown, StreamInfo
from pydantic import Field
from rapidocr import EngineType, LangDet, ModelType, OCRVersion, RapidOCR
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ..common.ann import ANNIndex
from ..common.app import app_dirs
from ..common.pydantic import SearchQuery, SearchResult
from .search_provider import SearchProvider


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
        local_dir_use_symlinks=False,
        allow_patterns=DEFAULT_REQUIRED,
        resume_download=True,
        revision="c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
    )
    after = all((local_dir / f).exists() for f in DEFAULT_REQUIRED)
    return (not before) and after


class ONNXTextSearchProvider(SearchProvider):
    """ONNX Text Embedder."""

    onnx_model_path: Path = Field(
        default_factory=lambda: _get_default_local_dir() / "onnx" / "model_quint8_avx2.onnx",
        description="Path to the ONNX model.",
    )
    tokenizer_path: Path = Field(default_factory=_get_default_local_dir, description="Path to the tokenizer.")
    ocr_enabled: bool = Field(default=False, description="Whether to enable OCR.")

    k: int = Field(default=1000, description="Number of results to return.")
    min_chunk_size: int = Field(default=16, description="Minimum chunk size for the text.")
    max_chunk_size: int = Field(default=1024, description="Maximum chunk size for the text.")
    overlap: int = Field(default=128, description="Overlap for the text chunks.")
    max_filesize_mb: float = Field(default=10, description="Maximum file size to index in MB.")
    ann_cache_dir: Path = Field(
        default_factory=lambda: app_dirs.app_cache_dir / "text_ann", description="Path to the ANN cache directory."
    )

    @cached_property
    def markitdown(self) -> MarkItDown:
        """MarkItDown."""
        return MarkItDown()

    @cached_property
    def ocr_engine(self) -> RapidOCR:
        """RapidOCR."""
        eng = RapidOCR(
            params={
                "Global.log_level": "error",
                "Det.engine_type": EngineType.ONNXRUNTIME,
                "Det.lang_type": LangDet.MULTI,
                "Det.model_type": ModelType.MOBILE,
                "Det.ocr_version": OCRVersion.PPOCRV4,
            }
        )
        return eng

    @property
    def supported_types(self) -> set[str]:
        """Supported document types."""
        return {"txt", "md", "docx", "pdf", "pptx", "epub"}

    def setup(self) -> None:
        """Setup the provider."""
        self._idle = threading.Event()
        if self.tokenizer_path == _get_default_local_dir():
            _download_default_models()
        assert self.session is not None
        assert self.tokenizer is not None
        test_emb = self.embed(["test"])
        self._index = ANNIndex(dims=test_emb.shape[-1], cache_dir=self.ann_cache_dir).start_eventful()
        self._idle.set()

    def teardown(self) -> None:
        """Teardown the provider."""

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
    def tokenizer(self) -> PreTrainedTokenizerFast:
        """Tokenizer."""
        return AutoTokenizer.from_pretrained(self.tokenizer_path)

    def _chunk(self, text: str) -> list[str]:
        """Chunk text."""
        step_size = self.max_chunk_size - self.overlap
        text_chunks = [text[i : i + self.max_chunk_size] for i in range(0, len(text), step_size)]
        text_chunks = [chunk for chunk in text_chunks if len(chunk) >= self.min_chunk_size]
        return text_chunks

    def embed(self, text: list[str]) -> np.ndarray:
        """Embed text."""
        # Fixed-length padding helps performance in ORT by producing uniform shapes
        max_len = min(getattr(self.tokenizer, "model_max_length", 512) or 512, 512)
        batch = cast(
            "dict[str, np.ndarray]",
            self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="np",
            ),
        )
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

    def update(self, path: Path) -> bool:
        """Handle a change event."""
        extension = path.suffix.strip(".")

        if extension not in self.supported_types:
            return False

        if not path.exists():
            return self._index.remove(path)

        if path.stat().st_size > self.max_filesize_mb * 1024 * 1024:
            return False

        if path in self._index:
            return False

        stream_info = None
        if extension in ["txt", "md"]:
            best_guess = from_path(path).best()
            if best_guess is not None:
                stream_info = StreamInfo(
                    charset=best_guess.encoding,
                )

        try:
            text = self.markitdown.convert(
                path,
                stream_info=stream_info,
            ).markdown
            text_chunks = self._chunk(text)
        except Exception:
            # TODO log error
            return False

        if extension == "pdf" and self.ocr_enabled:
            try:
                with fitz.open(path) as doc:
                    for page in doc:
                        for img in page.get_images(full=True):
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            texts = cast(
                                "tuple[str, ...] | None",
                                self.ocr_engine(image_bytes).txts,  # type: ignore
                            )
                            if texts is None or len(texts) == 0:
                                continue
                            ocr_text = "\n".join(texts)
                            text_chunks.extend(self._chunk(ocr_text))
            except Exception:
                # TODO log error
                pass

        if len(text_chunks) == 0:
            return False

        for chunk in text_chunks:
            self._idle.wait()
            emb = self.embed([chunk])[0]
            emb = emb / np.linalg.norm(emb)
            self._index.add(path, emb)
        return True

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search for a query."""
        self._idle.wait()
        self._idle.clear()
        results: list[SearchResult] = []
        query_embedding = self.embed([query.text])[0]
        # Normalize query to align with normalized corpus vectors
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        for path, distance in self._index.query(query_embedding, self.k):
            similarity = 1 - distance
            results.append(SearchResult(value=path, confidence=similarity))

        results_filtered: list[SearchResult] = []
        for path, group in groupby(sorted(results, key=lambda x: x.value), key=lambda x: x.value):
            chunk_score = np.mean([x.confidence for x in group]).item()
            # num_chunks = len(self._index._index_helper._ids_by_path[path.as_posix()])
            doc_score = chunk_score  # / math.log(1 + 0.1 * num_chunks)
            results_filtered.append(SearchResult(value=path, confidence=doc_score))

        self._idle.set()
        return results_filtered
