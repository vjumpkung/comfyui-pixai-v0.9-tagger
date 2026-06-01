from __future__ import annotations

import subprocess
import sys


DEPENDENCIES = [
    "dghs-imgutils>=0.19.0",
    "timm",
    "pillow",
    "pandas",
    "numpy<2.3",
]


def install_dependency(dependency: str) -> None:
    print(f"Installing {dependency}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", dependency])


def main() -> None:
    for dependency in DEPENDENCIES:
        install_dependency(dependency)


if __name__ == "__main__":
    main()
