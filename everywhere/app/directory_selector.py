"""Directory selector modal for choosing indexed directories."""

import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Tree
from textual.widgets.tree import TreeNode

if sys.platform.startswith("win"):
    CHECKED = "[green](x) [/] "
    UNCHECKED = "[white]( ) [/] "
    PARTIAL = "[yellow](-) [/] "
else:
    CHECKED = "[green]☑ [/] "
    UNCHECKED = "[white]☐ [/] "
    PARTIAL = "[yellow]☐ [/] "


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
    #tree .tree--cursor,
    #tree .tree--highlight,
    #tree .tree--highlight-line,
    #tree .tree--guides-selected,
    #tree .tree--guides-hover {
        background: transparent;
        text-style: none;
        outline: none;
    }
    #tree:focus {
        outline: none;
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
        if not self.selected:
            self._reveal_path(Path.cwd())

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
        return bool(self._iter_subdirs(path))

    def _load_children(self, node: TreeNode[NodeData]) -> None:
        """Populate immediate subdirectories lazily."""
        data = node.data
        if not data:
            return
        path = data.path
        if node.children:
            return
        subdirs = self._iter_subdirs(path)

        subdirs.sort(key=lambda p: p.name.lower())
        for d in subdirs:
            child = node.add(self._label_for(d), data=NodeData(d), allow_expand=self._has_children(d))
            self.path_index[d] = child

    def _iter_subdirs(self, path: Path) -> list[Path]:
        try:
            with os.scandir(path) as it:
                return [
                    Path(e.path)
                    for e in it
                    if e.is_dir(follow_symlinks=False) and (self.show_hidden or not Path(e.name).name.startswith("."))
                ]
        except Exception:
            return []

    def _label_for(self, path: Path) -> str:
        state = self._state_for(path)
        mark = CHECKED if state == "checked" else PARTIAL if state == "partial" else UNCHECKED
        is_root = path == Path(path.anchor)
        name = (path.drive or path.anchor).rstrip("\\/") if is_root else path.name or str(path)
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
        while node.parent is not None:
            node = node.parent
        self._refresh_labels(node)

    def _toggle(self, path: Path) -> None:
        path = path.resolve()
        state = self._state_for(path)

        if state == "checked":
            selected_directly = path in self.selected
            selected_ancestors = [p for p in self.selected if p != path and path.is_relative_to(p)]
            self.selected = {p for p in self.selected if not p.is_relative_to(path)}

            if selected_ancestors and not selected_directly:
                parent_in_selected = max(selected_ancestors, key=lambda p: len(p.parts))
                self.selected.discard(parent_in_selected)
                for child_path in self._iter_subdirs(parent_in_selected):
                    if child_path != path:
                        self.selected.add(child_path)
            return
        else:
            self.selected = {p for p in self.selected if not p.is_relative_to(path)}
            self.selected.add(path)
            if path.parent != path:
                self._consolidate_selection(path.parent)

    def _consolidate_selection(self, parent: Path) -> None:
        if parent == parent.parent:
            return

        subdirs = self._iter_subdirs(parent)
        if subdirs and all(self._state_for(d) == "checked" for d in subdirs):
            self.selected = {p for p in self.selected if not p.is_relative_to(parent)}
            self.selected.add(parent)
            if parent.parent != parent:
                self._consolidate_selection(parent.parent)

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
        tree = self.query_one("#tree", Tree)
        tree.unselect()

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
