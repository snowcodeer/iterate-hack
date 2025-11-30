"""UniWrap Generated Environments."""
import importlib
from pathlib import Path

# Auto-discover and import all environment submodules
_env_dir = Path(__file__).parent
for _subdir in _env_dir.iterdir():
    if _subdir.is_dir() and not _subdir.name.startswith('_'):
        _init_file = _subdir / "__init__.py"
        if _init_file.exists():
            try:
                importlib.import_module(f".{_subdir.name}", package=__name__)
            except ImportError as e:
                print(f"Warning: Could not import environment {_subdir.name}: {e}")
from . import onlinespiele_sammlung_de_tetris_tetrisgames_xin_yang
from . import python_tetris
from . import sandtris_com
