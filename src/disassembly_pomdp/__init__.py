import sys
from pathlib import Path

# Make the subtree packages importable without an install or PYTHONPATH:
# `crc_disassembly_sequence_generator` (found from the repo root) and
# `diss_assembly_dataclasses` (found from its src/ folder).
_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "crc_diss_assembly_dataclasses" / "src"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))
