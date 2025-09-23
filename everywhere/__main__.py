"""Application entry point."""

import argparse
import shutil

from .app.app import EverywhereApp
from .app.app_config import initialize_app_components
from .common.app import app_dirs


def reset_all() -> None:
    """Delete both config and cache directories."""
    if app_dirs.app_data_dir.exists():
        shutil.rmtree(app_dirs.app_data_dir)
        print(f"App data directory deleted: {app_dirs.app_data_dir}")
    else:
        print(f"App data directory does not exist: {app_dirs.app_data_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Everywhere - File search application")
    parser.add_argument("--temp", action="store_true", help="Run in temporary mode")
    parser.add_argument("--reset", action="store_true", help="Delete all app data")

    args = parser.parse_args()

    if args.reset:
        reset_all()
        return

    if args.temp:
        app_dirs.use_temp_app_data_dir()

    initialize_app_components()
    app = EverywhereApp()
    app.run()


if __name__ == "__main__":
    main()
