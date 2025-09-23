"""App components."""

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from ..common.app import app_dirs
from ..search_providers.search_provider import SearchProvider
from ..search_providers.text_embedding_search import EmbeddingSearchProvider
from ..watchers.fs_watcher import FSWatcher


class AppComponents(BaseModel):
    """App components."""

    embedding_search: EmbeddingSearchProvider = Field(default_factory=EmbeddingSearchProvider)
    fs_watcher: FSWatcher = Field(default_factory=FSWatcher)

    @property
    def search_providers(self) -> list[SearchProvider]:
        """Search providers."""
        return [self.embedding_search]

    @property
    def indexed_paths(self) -> list[Path]:
        """Indexed paths."""
        return self.fs_watcher.fs_path

    def start(self) -> None:
        """Start the app components."""
        self.embedding_search.start()
        self.fs_watcher.start()

    @model_validator(mode="after")
    def autofill_supported_types(self) -> "AppComponents":
        """Autofill supported types."""
        self.fs_watcher.supported_types = set.union(*[provider.supported_types for provider in self.search_providers])
        return self


# Global app components instance
_APP_COMPONENTS: AppComponents | None = None


def get_app_components() -> AppComponents:
    """Get the global app components instance.

    Returns:
        The global AppComponents instance.

    Raises:
        RuntimeError: If app components have not been initialized.
    """
    if _APP_COMPONENTS is None:
        raise RuntimeError("App components not initialized. Call initialize_app_components first.")
    return _APP_COMPONENTS


def initialize_app_components(config_path: Path | None = None) -> AppComponents:
    """Initialize app components and start services.

    Args:
        config_path: Path to the app configuration. If None, uses default.

    Returns:
        Initialized and started AppComponents instance.
    """
    global _APP_COMPONENTS
    config_path = config_path or app_dirs.app_config_path
    _APP_COMPONENTS = (
        AppComponents() if not config_path.exists() else AppComponents.model_validate_json(config_path.read_text())
    )

    _APP_COMPONENTS.start()

    return _APP_COMPONENTS
