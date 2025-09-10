"""App components."""

from pydantic import BaseModel, Field, model_validator

from ..search_providers.onnx_text_search import ONNXTextSearchProvider
from ..search_providers.search_provider import SearchProvider
from ..watchers.fs_watcher import FSWatcher


class AppComponents(BaseModel):
    """App components."""

    onnx_text_search: ONNXTextSearchProvider = Field(default_factory=ONNXTextSearchProvider)
    fs_watcher: FSWatcher = Field(default_factory=FSWatcher)

    @property
    def search_providers(self) -> list[SearchProvider]:
        """Search providers."""
        return [self.onnx_text_search]

    @model_validator(mode="after")
    def autofill_supported_types(self) -> "AppComponents":
        """Autofill supported types."""
        self.fs_watcher.supported_types = set.union(*[provider.supported_types for provider in self.search_providers])
        return self
