"""App components."""

from itertools import groupby
from pathlib import Path

import numpy as np
from pydantic import Field, model_validator

from ..common.app import app_dirs
from ..common.pydantic import SearchQuery, SearchResult
from ..search_providers.search_provider import SearchProvider
from ..search_providers.text_embedding_search import EmbeddingSearchProvider
from ..watchers.fs_watcher import FSWatcher

RESULT_LIMIT = 1000


def normalize_scores(scores: list[float]) -> list[float]:
    """Normalize scores."""
    if not scores:
        return []

    scores_np = np.array(scores)
    mu = scores_np.mean()
    normalized = (scores_np - mu) / (scores_np.max() - mu + 1e-8)
    normalized = np.clip(normalized, 0, 1)
    return normalized.tolist()


def _normalize_results(results: list[SearchResult]) -> list[SearchResult]:
    """Normalize results."""
    results_agg: list[SearchResult] = []
    for _, group in groupby(sorted(results, key=lambda x: x.value), key=lambda x: x.value):
        results_agg.append(max(group, key=lambda x: x.confidence))

    new_scores = normalize_scores([x.confidence for x in results_agg])
    results_agg = [SearchResult(value=x.value, confidence=new_scores[i]) for i, x in enumerate(results_agg)]
    results_agg.sort(key=lambda x: x.confidence, reverse=True)
    return results_agg[:RESULT_LIMIT]


class AppComponents(SearchProvider):
    """App components."""

    embedding_search: EmbeddingSearchProvider = Field(default_factory=EmbeddingSearchProvider)
    fs_watcher: FSWatcher = Field(default_factory=FSWatcher)

    @property
    def supported_types(self) -> set[str]:
        """Supported types."""
        return set.union(*[provider.supported_types for provider in self.search_providers])

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
        self.fs_watcher.supported_types = self.supported_types
        return self

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search for a query."""
        results = []
        for provider in self.search_providers:
            results.extend(provider.search(query))
        return _normalize_results(results)


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
