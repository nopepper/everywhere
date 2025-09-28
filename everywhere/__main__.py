"""Application entry point."""

import argparse
import shutil

from .app.app import EverywhereApp
from .app.app_config import AppConfig, build_controller
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

    if not app_dirs.app_config_path.exists():
        config = AppConfig()
    else:
        config = AppConfig.model_validate_json(app_dirs.app_config_path.read_text())

    controller = build_controller(config)
    with controller:
        app = EverywhereApp(config, controller)
        try:
            app.run()
        finally:
            app_dirs.app_config_path.parent.mkdir(parents=True, exist_ok=True)
            app_dirs.app_config_path.write_text(app.dump_config().model_dump_json(indent=2))


if __name__ == "__main__":
    main()
