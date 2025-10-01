"""Results table widget."""

from datetime import datetime

from rich.text import Text
from textual.coordinate import Coordinate
from textual.widgets import DataTable

from ...common.pydantic import SearchResult
from ...events import add_callback
from ...events.app import AppResized


def _format_size(n: int | None) -> str:
    if n is None:
        return ""
    if n == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i, s = 0, float(n)
    while s >= 1024 and i < len(units) - 1:
        s /= 1024
        i += 1
    if i == 0:
        return f"{int(s)} {units[i]}"
    else:
        return f"{s:.1f} {units[i]}"


def _confidence_chip(p: float) -> Text:
    p = max(0.0, min(1.0, p * p))
    r = round(208 + (0 - 208) * p)
    g = round(208 + (255 - 208) * p)
    b = round(208 + (0 - 208) * p)
    return Text("  ", style=f"on #{r:02x}{g:02x}{b:02x}")


def _format_date(ns: int | None) -> str:
    if ns is None:
        return ""
    return datetime.fromtimestamp(ns / 1_000_000_000).strftime("%Y-%m-%d %H:%M:%S")


DEBOUNCE_LATENCY = 0.1
RESULT_LIMIT = 1000


class ResultsTable(DataTable):
    """Owns sizing and diff-updates."""

    def on_mount(self) -> None:
        """Set up the table when mounted."""
        self.add_columns("", "Name", "Path", "Size", "Date Modified")
        self.cursor_type = "cell"
        add_callback(AppResized, self._on_size_changed)

    def _on_size_changed(self, msg: AppResized) -> None:
        """Handle size changes."""
        if len(self.ordered_columns) != 5:
            return
        console_width = msg.width
        total = max(0, console_width - 2)  # margins
        if total <= 10:
            return
        CONF, SIZE, DATE, OVERHEAD = 2, 8, 20, 12  # noqa: N806
        fixed = CONF + SIZE + DATE + OVERHEAD
        free = max(0, total - fixed)
        name_w = max(25, free // 4)
        path_w = max(1, free - name_w)
        widths = [CONF, name_w, path_w, SIZE, DATE]
        for i, w in enumerate(widths):
            col = self.ordered_columns[i]
            col.auto_width = False
            col.width = w
        # internal refresh hint
        if hasattr(self, "_require_update_dimensions"):
            self._require_update_dimensions = True
        self.refresh()

    def update_results(self, results: list[SearchResult]) -> None:
        """Handle search results."""
        results = sorted(results, key=lambda x: x.confidence, reverse=True)
        for i, result in enumerate(results):
            path = result.value
            confidence_label = _confidence_chip(result.confidence)
            size_label = _format_size(result.size_bytes)
            date_label = _format_date(result.last_modified_ns)
            new_col_values = [
                confidence_label,
                path.name,
                str(path),
                size_label,
                date_label,
            ]
            if i < self.row_count:
                old_col_values = self.get_row_at(i)
                for j, (old_value, new_value) in enumerate(zip(old_col_values, new_col_values, strict=True)):
                    if old_value != new_value:
                        self.update_cell_at(Coordinate(row=i, column=j), new_value)
            else:
                self.add_row(*new_col_values)
        if self.row_count > len(results):
            keys_to_remove = [
                self.coordinate_to_cell_key(Coordinate(row=row_index, column=0))[0]
                for row_index in range(len(results), self.row_count)
            ]
            for row_key in keys_to_remove:
                self.remove_row(row_key)
