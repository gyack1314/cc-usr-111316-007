#!/usr/bin/env python3 -I
"""隔离模式测试驱动：python -I 不自动放入当前目录，显式加入项目根。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.discover(str(Path(__file__).resolve().parent / 'tests'),
                            pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
