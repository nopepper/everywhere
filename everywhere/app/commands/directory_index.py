"""Directory index command provider."""

from textual.command import DiscoveryHit, Hit, Hits, Provider


class DirectoryIndexCommand(Provider):
    """Command provider for directory indexing and application commands."""

    def __init__(self, *args, **kwargs):
        """Initialize the directory index command."""
        super().__init__(*args, **kwargs)
        self.action_select_directories = self.app.action_select_directories  # type: ignore
        self.action_close_app = self.app.action_close_app  # type: ignore

    async def discover(self) -> Hits:
        """Expose common actions in the command palette."""
        yield DiscoveryHit(
            "Pick indexed directories...", self.action_select_directories, help="Select directories to be indexed"
        )
        yield DiscoveryHit("Close application", self.action_close_app, help="Close the application")

    async def search(self, query: str) -> Hits:
        """Return actions when the query matches."""
        matcher = self.matcher(query)

        # Directory selection command
        if query.lower().startswith("pick") or query.lower().startswith("dir") or query.lower().startswith("index"):
            yield DiscoveryHit(
                "Pick indexed directories...",
                self.action_select_directories,
                help="Select directories to be indexed",
            )

        # Close application command
        close_command = "Close application"
        score = matcher.match(close_command)
        if score > 0:
            yield Hit(
                score,
                matcher.highlight(close_command),
                self.action_close_app,
                help="Close the application",
            )
