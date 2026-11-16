"""`import disassembly_pomdp` must work from a clean shell, without conftest or PYTHONPATH."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_env_builder_imports_with_only_src_on_path(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); "
        "import disassembly_pomdp.env_builder"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
