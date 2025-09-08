"""Directory selector modal for choosing indexed directories."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from textual import on
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Tree

if TYPE_CHECKING:
    from collections.abc import Iterable

    from textual.app import ComposeResult
    from textual.events import Key
    from textual.widgets.tree import TreeNode


CHECKED = "[green]☑\ufe0e[/] "
UNCHECKED = "[white]☐\ufe0e[/] "
PARTIAL = "[yellow]☐\ufe0e[/] "


@dataclass(frozen=True)
class NodeData:
    """Data associated with each tree node."""

    path: Path


class DirectorySelector(ModalScreen[set[Path]]):
    """Modal directory picker with tri-state checkboxes."""

    DEFAULT_CSS = """
    DirectorySelector { align: center middle; }
    DirectorySelector > Container {
        width: 80;
        height: 80%;
        background: $surface;
        border: round $primary;
    }
    #hdr { dock: top; height: 3; padding: 1; background: $primary; }
    #title { color: $text; text-style: bold; }
    #tree-wrap { dock: top; height: 1fr; padding: 1; overflow-y: auto; }
    #btns { dock: bottom; align: center middle; height: 3; }
    #btns Button {
        margin: 0 1;
        min-width: 30;
        height: auto;
        content-align: center middle;
    }
    #btns #ok {
        background: $primary;
        color: $text-primary;
    }
    #btns #cancel {
        background: $surface-lighten-1;
        color: $text;
    }
    """

    def __init__(
        self,
        initial_paths: Iterable[Path] | Path | None = None,
        root_path: Path | None = None,
        show_hidden: bool = False,
    ) -> None:
        """Initialize the directory selector.

        Args:
            initial_paths: Initially selected directories.
            root_path: Root directory to show in the tree.
            show_hidden: Whether to show hidden directories.
        """
        super().__init__()
        if initial_paths is None:
            self.selected: set[Path] = set()
        elif isinstance(initial_paths, Path):
            self.selected = {initial_paths.resolve()}
        else:
            self.selected = {p.resolve() for p in initial_paths}

        if root_path is not None:
            self.roots = [root_path.resolve()]
        else:
            self.roots = self._detect_roots()

        self.show_hidden = show_hidden
        self.path_index: dict[Path, TreeNode[NodeData]] = {}

    def compose(self) -> ComposeResult:
        """Compose the modal layout with header, tree and buttons."""
        with Container():
            with Container(id="hdr"):
                yield Label("Select directories to index", id="title")
            with Container(id="tree-wrap"):
                yield Tree("Root", id="tree")
            with Horizontal(id="btns"):
                yield Button("OK", id="ok", compact=True)
                yield Button("Cancel", id="cancel", compact=True)

    def on_mount(self) -> None:
        """Initialize the tree and reveal initial selections."""
        tree = self.query_one("#tree", Tree)
        tree.show_root = False
        tree.guide_depth = 2
        tree.auto_expand = False
        for root in self.roots:
            node = tree.root.add(self._label_for(root), data=NodeData(root), allow_expand=self._has_children(root))
            self.path_index[root] = node
        for path in sorted(self.selected):
            self._reveal_path(path)
        self._refresh_labels(tree.root)
        tree.focus()

    def _detect_roots(self) -> list[Path]:
        if Path("/").anchor == "/":
            return [Path("/")]
        roots = []
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            p = Path(f"{c}:/")
            try:
                if p.exists():
                    roots.append(p)
            except OSError:
                pass
        return roots or [Path("/")]

    def _has_children(self, path: Path) -> bool:
        """Check if a directory has any subdirectories."""
        try:
            with os.scandir(path) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False) and (
                        self.show_hidden or not Path(entry.name).name.startswith(".")
                    ):
                        return True
            return False
        except Exception:
            return False

    def _load_children(self, node: TreeNode[NodeData]) -> None:
        """Populate immediate subdirectories lazily."""
        data = node.data
        if not data:
            return
        path = data.path
        if node.children:
            return
        try:
            with os.scandir(path) as it:
                subdirs = [
                    Path(e.path)
                    for e in it
                    if e.is_dir(follow_symlinks=False) and (self.show_hidden or not Path(e.name).name.startswith("."))
                ]
        except Exception:
            subdirs = []

        subdirs.sort(key=lambda p: p.name.lower())
        for d in subdirs:
            child = node.add(self._label_for(d), data=NodeData(d), allow_expand=self._has_children(d))
            self.path_index[d] = child

    def _label_for(self, path: Path) -> str:
        state = self._state_for(path)
        mark = CHECKED if state == "checked" else PARTIAL if state == "partial" else UNCHECKED

        # root of a volume (C:) or POSIX root (/)
        is_root = path == Path(path.anchor)

        # Use ternary operator as suggested by linter
        name = (path.drive or path.anchor).rstrip("\\/") if is_root else path.name or str(path)

        # Add a placeholder for alignment if there are no children
        prefix = "  " if not self._has_children(path) else ""
        return f"{prefix}{mark}{name}"

    def _state_for(self, path: Path) -> str:
        if path in self.selected:
            return "checked"
        if any(path.is_relative_to(s) for s in self.selected):
            return "checked"
        if any(s.is_relative_to(path) for s in self.selected):
            return "partial"
        return "unchecked"

    def _refresh_labels(self, node: TreeNode[NodeData]) -> None:
        if node is not None and node.data:
            node.set_label(self._label_for(node.data.path))
        for c in node.children:
            self._refresh_labels(c)

    def _refresh_branch(self, node: TreeNode[NodeData]) -> None:
        """Refresh labels from the nearest root to avoid flicker."""
        while node.parent is not None:
            node = node.parent
        self._refresh_labels(node)

    def _toggle(self, path: Path) -> None:
        path = path.resolve()
        overlaps = {p for p in self.selected if p.is_relative_to(path) or path.is_relative_to(p)}
        if path in overlaps and len(overlaps) == 1:
            self.selected.remove(path)
        else:
            self.selected -= overlaps
            self.selected.add(path)

    def _reveal_path(self, path: Path) -> None:
        root = next((r for r in self.roots if path.anchor.lower() == r.anchor.lower()), self.roots[0])
        cur = self.path_index.get(root)
        if not cur:
            return
        cur.expand()
        parts = path.resolve().parts
        if not parts:
            return
        start_i = 1 if len(parts) > 1 else 0
        base = Path(parts[0]) if parts else root
        for part in parts[start_i:]:
            base = base / part
            self._load_children(cur)
            nxt = self.path_index.get(base)
            if not nxt:
                nxt = cur.add(self._label_for(base), data=NodeData(base), allow_expand=self._has_children(base))
                self.path_index[base] = nxt
            cur = nxt
            cur.expand()

    @on(Tree.NodeExpanded, "#tree")
    def _on_expanded(self, ev: Tree.NodeExpanded) -> None:
        self._load_children(ev.node)
        self._refresh_branch(ev.node)

    @on(Tree.NodeSelected, "#tree")
    def _on_selected(self, ev: Tree.NodeSelected) -> None:
        """Toggle selection without changing expansion state."""
        node = ev.node
        if not node.data:
            return
        self._toggle(node.data.path)
        self._refresh_branch(node)
        ev.stop()

    def on_key(self, event: Key) -> None:
        """Handle keyboard events for space/escape."""
        if event.key == "space":
            tree = self.query_one("#tree", Tree)
            if tree.cursor_node and tree.cursor_node.data:
                self._toggle(tree.cursor_node.data.path)
                self._refresh_branch(tree.cursor_node)
                event.stop()
        elif event.key == "escape":
            self.dismiss(None)
            event.stop()

    @on(Button.Pressed, "#ok")
    def _ok(self) -> None:
        self.dismiss(set(self.selected))

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        self.dismiss(None)
