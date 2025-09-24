"""App constants."""

from pathlib import Path
from tempfile import TemporaryDirectory

from platformdirs import user_data_dir

APP_NAME = "EverywhereApp"
APP_AUTHOR = "nopepper"


class AppDirs:
    """App directories."""

    def __init__(self) -> None:
        """Initialize app directories."""
        self._temp_dir: TemporaryDirectory | None = None
        self.app_data_dir = Path(user_data_dir(appname=APP_NAME, appauthor=APP_AUTHOR))
        self.app_cache_dir = self.app_data_dir / "cache"
        self.app_models_dir = self.app_data_dir / "models"
        self.app_config_path = self.app_data_dir / "config.json"

    def use_temp_app_data_dir(self) -> None:
        """Set app data to a temporary directory."""
        self._temp_dir = TemporaryDirectory()
        self.app_data_dir = Path(self._temp_dir.name)
        self.app_cache_dir = self.app_data_dir / "cache"
        self.app_models_dir = self.app_data_dir / "models"
        self.app_config_path = self.app_data_dir / "config.json"


app_dirs = AppDirs()
