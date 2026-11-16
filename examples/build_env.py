# examples/build_env.py
"""How to build the disassembly POMDP env for EMO; run it for one reset and one step.

The env's behaviour on EMO is tested in tests/integration/test_emo_env.py.
"""

import sys
from pathlib import Path

# Add project src/ folder to sys.path so that `import disassembly_pomdp` works,
# and the dataclasses subtree's src/ so that `import diss_assembly_dataclasses` works
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "crc_diss_assembly_dataclasses" / "src"))

import numpy as np
from diss_assembly_dataclasses.assembly import Assembly

from disassembly_pomdp.config import CellSettings
from disassembly_pomdp.disassembly_pomdp_gym import DisassemblyPOMDPEnv
from disassembly_pomdp.env_builder import build_env as build_product_env
from disassembly_pomdp.env_builder import default_cell_settings, load_product

EMO_FOLDER = ROOT / "emo_sr_v2"


def load_emo_sr_v2_assembly() -> Assembly:
    """Load the EMO assembly from the emo_sr_v2 product folder."""
    return load_product(EMO_FOLDER)[0]


def make_cell_settings() -> CellSettings:
    return default_cell_settings()


def build_env() -> DisassemblyPOMDPEnv:
    """EMO on the default cell; any other product: `disassembly_pomdp.env_builder.build_env`."""
    return build_product_env(EMO_FOLDER)


if __name__ == "__main__":
    env = build_env()
    obs, info = env.reset(seed=0)
    print("capabilities:", [type(cap).__name__ for cap in env.capabilities])
    print("action space:", env.action_space, "| observation keys:", list(obs))

    obs, reward, terminated, truncated, info = env.step(np.array([0, 0, 0]))
    print("first action:", info.get("semantic_action"))
    print(
        "success:",
        info.get("success"),
        "| reward:",
        round(reward, 3),
        "| terminated:",
        terminated,
    )
