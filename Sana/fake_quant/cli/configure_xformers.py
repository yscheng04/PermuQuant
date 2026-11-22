"""Disable xformers 0.0.32.post2's Hopper FA3 default for RTX 5090.
The bundled FA3 dispatcher incorrectly selects its Hopper kernel on sm_120.
FA2/CUTLASS dispatch is retained; both GPUs were verified with BF16 attention.
Run with the permuquant environment's Python after installing xformers.
"""

from pathlib import Path
import importlib.util

spec = importlib.util.find_spec("xformers")
p = Path(spec.origin).parent / "ops/fmha/dispatch.py"
s = p.read_text()
old = "_USE_FLASH_ATTENTION_3 = True"
new = "_USE_FLASH_ATTENTION_3 = False  # PermuQuant RTX 5090 compatibility: use FA2/CUTLASS"
if old in s:
    p.write_text(s.replace(old, new, 1))
    print("Configured RTX 5090 attention dispatch:", p)
elif new in s:
    print("RTX 5090 attention dispatch already configured")
else:
    raise RuntimeError(
        "Unexpected xformers version or dispatch source; inspect before changing"
    )
