"""ONNX-based text search provider."""

from functools import cached_property
from pathlib import Path
from typing import cast

import fitz
from charset_normalizer import from_path
from markitdown import MarkItDown, StreamInfo
from pydantic import BaseModel, Field
from rapidocr import EngineType, LangDet, ModelType, OCRVersion, RapidOCR


class TextParser(BaseModel):
    """ONNX Text Embedder."""

    max_filesize_mb: float = Field(default=10, description="Maximum file size to index in MB.")
    ocr_enabled: bool = Field(default=True, description="Whether to enable OCR.")

    @property
    def supported_types(self) -> set[str]:
        """Supported file types."""
        return {"txt", "md", "docx", "pdf", "pptx", "epub"}

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

    def parse(self, path: Path) -> list[str]:
        """Parse a file's text."""
        extension = path.suffix.strip(".")
        results: list[str] = []

        if not path.exists():
            return results

        if path.stat().st_size > self.max_filesize_mb * 1024 * 1024:
            return results

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
            results.append(text)
        except Exception:
            # TODO log error
            pass

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
                            results.extend(texts)
            except Exception:
                # TODO log error
                pass

        return results
