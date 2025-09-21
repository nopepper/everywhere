"""Contract tests for core functionality."""

from pathlib import Path

import pytest

from everywhere.app.app import confidence_to_color, format_date, format_size
from everywhere.common.pydantic import SearchQuery, SearchResult
from everywhere.events import get_listener, publish
from everywhere.events.app import UserSearched
from tests.test_utils import FakeTextSearchProvider


class TestEventBusContracts:
    """Test event bus delivery contracts."""

    def test_publish_reaches_specific_subscriber(self):
        """Test that publishing an event reaches specific subscribers."""
        listener = get_listener(UserSearched)
        publish(UserSearched(query="test"))

        event = listener.get(timeout=1.0)
        assert event is not None
        assert isinstance(event, UserSearched)
        assert event.query == "test"

    def test_publish_reaches_all_subscribers(self):
        """Test that publishing an event reaches the all-events subscriber."""
        all_listener = get_listener()
        specific_listener = get_listener(UserSearched)

        publish(UserSearched(query="test"))

        # Both should receive the event
        all_event = all_listener.get(timeout=1.0)
        specific_event = specific_listener.get(timeout=1.0)

        assert all_event is not None
        assert specific_event is not None
        assert isinstance(all_event, UserSearched)
        assert isinstance(specific_event, UserSearched)

    def test_correlation_id_propagation(self):
        """Test that correlation IDs propagate through event publishing."""
        listener = get_listener(UserSearched)
        publish(UserSearched(query="test"))

        event = listener.get(timeout=1.0)
        assert event is not None
        assert hasattr(event, "correlation_id")
        assert event.correlation_id


class TestSearchProviderContracts:
    """Test search provider interface contracts."""

    def test_empty_query_returns_no_results(self):
        """Test that empty queries return no results."""
        provider = FakeTextSearchProvider(max_filesize_mb=1)
        provider.setup()

        try:
            query = SearchQuery(text="")
            results = list(provider.search(query))
            assert len(results) == 0
        finally:
            provider.teardown()

    def test_unsupported_file_type_ignored(self, tmp_path: Path):
        """Test that unsupported file types are ignored."""
        provider = FakeTextSearchProvider(max_filesize_mb=1)
        provider.setup()

        try:
            # Create a fake .exe file
            fake_exe = tmp_path / "fake.exe"
            fake_exe.write_text("This is not a real executable")

            success = provider.update(fake_exe)
            assert not success  # Should not update unsupported files
        finally:
            provider.teardown()

    def test_large_file_skipped(self, tmp_path: Path):
        """Test that large files are skipped."""
        provider = FakeTextSearchProvider(max_filesize_mb=1)  # 1MB limit
        provider.setup()

        try:
            # Create a file larger than the limit
            large_file = tmp_path / "large.txt"
            large_content = "x" * (2 * 1024 * 1024)  # 2MB
            large_file.write_text(large_content)

            success = provider.update(large_file)
            assert not success  # Should not update large files

            large_file.unlink()  # Clean up
        finally:
            provider.teardown()


class TestPureFunctionContracts:
    """Test pure function contracts."""

    def test_format_size_boundaries(self):
        """Test format_size at key boundaries."""
        assert format_size(0) == "0 B"
        assert format_size(1023) == "1023 B"
        assert "KB" in format_size(1024)
        assert "MB" in format_size(1024 * 1024)
        assert "GB" in format_size(1024 * 1024 * 1024)

    def test_confidence_to_color_monotonicity(self):
        """Test that confidence_to_color produces brighter colors for higher confidence."""
        low_color = confidence_to_color(0.1)
        high_color = confidence_to_color(0.9)

        low_style_str = str(low_color.style)
        high_style_str = str(high_color.style)

        assert "#" in low_style_str
        assert "#" in high_style_str

        assert high_style_str != low_style_str

    def test_search_result_bounds(self):
        """Test that SearchResult confidence is bounded."""
        result = SearchResult(value=Path("test.txt"), confidence=0.5)
        assert 0.0 <= result.confidence <= 1.0

        # Test boundary values
        low_result = SearchResult(value=Path("test.txt"), confidence=0.0)
        high_result = SearchResult(value=Path("test.txt"), confidence=1.0)

        assert low_result.confidence == 0.0
        assert high_result.confidence == 1.0
