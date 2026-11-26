"""测试引导：保证未执行 ``pip install -e .`` 时也能从 src/ 导入。"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
