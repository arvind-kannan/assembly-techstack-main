"""The product root (#257): the absolute folder under which the product folders live on this machine.

It is never stored in an export; the export records paths relative to it. One function, `resolve_product_root`,
knows where it comes from: the app setting, else the environment variable, else the code default.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from disassembly_pomdp.graph_export import files_repository_root

ENV_VAR = "PRODUCT_EXPLORER_PRODUCT_ROOT"
SETTINGS_FILE = "settings.json"
SETTING = "product_root"

APP_SETTING = "app setting"
ENVIRONMENT = "environment variable"
CODE_DEFAULT = "code default"


class InvalidProductRoot(ValueError):
    """A product root that is not an absolute path of an existing folder."""


@dataclass(frozen=True)
class ProductRoot:
    path: Path
    source: str  # APP_SETTING, ENVIRONMENT or CODE_DEFAULT


def config_dir() -> Path:
    """The Product explorer's folder in the user's configuration directory."""
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "product_explorer"


def read_setting(config: Path | None = None) -> str | None:
    """The product root of the app setting, or None when none is set (or the file is unreadable)."""
    path = (config or config_dir()) / SETTINGS_FILE
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get(SETTING)
    except (OSError, ValueError, AttributeError):
        return None
    return value or None


def write_setting(root: str | Path | None, config: Path | None = None) -> None:
    """Set the app setting to `root` (validated), or clear it with None. Other settings stay."""
    path = (config or config_dir()) / SETTINGS_FILE
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            settings = {}
    except (OSError, ValueError):
        settings = {}
    if root is None:
        settings.pop(SETTING, None)
    else:
        settings[SETTING] = str(_checked(root))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def resolve_product_root(
    product_folder: str | Path | None,
    config: Path | None = None,
    environ: dict[str, str] | None = None,
) -> ProductRoot | None:
    """The product root in effect and its source, or None when no source gives one. The code default is the
    root of the 3D files repository that holds `product_folder`."""
    setting = read_setting(config)
    if setting:
        return ProductRoot(_checked(setting), APP_SETTING)
    value = (os.environ if environ is None else environ).get(ENV_VAR)
    if value:
        return ProductRoot(_checked(value), ENVIRONMENT)
    if product_folder is not None:
        repo = files_repository_root(product_folder)
        if repo is not None:
            return ProductRoot(repo, CODE_DEFAULT)
    return None


def _checked(root: str | Path) -> Path:
    path = Path(root).expanduser()
    if not path.is_absolute():
        raise InvalidProductRoot(f"The product root must be an absolute path: {root}")
    if not path.is_dir():
        raise InvalidProductRoot(f"The product root is not a folder: {root}")
    return path.resolve()
