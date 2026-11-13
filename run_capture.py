#!/usr/bin/env python3 -I
"""隔离模式（python -I）驱动入口：脚本目录自动在 sys.path 上。

用法：
  ./run_capture.py analyze data/normal.json --format markdown
  ./run_capture.py analyze data/normal.json --format json -o out.json
  ./run_capture.py history data/normal.json
"""
import sys
from pathlib import Path

# -I（隔离模式）在本环境 Python 构建中不自动放入脚本目录，显式加入项目根。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture_analysis.cli import main

if __name__ == '__main__':
    main(sys.argv[1:])
