from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# dghs-imgutils pins numpy<2 in its package metadata, which would downgrade
# ComfyUI's numpy. requirements.txt lists the dependencies its PixAI code path
# actually needs; dghs-imgutils itself must be installed with --no-deps so pip
# never sees that pin.


def pip_install(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])


def main() -> None:
    pip_install("-r", os.path.join(ROOT, "requirements.txt"))
    pip_install("--no-deps", "dghs-imgutils>=0.19.0")


if __name__ == "__main__":
    main()
