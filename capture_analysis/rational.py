"""清单中有理数字符串的解析与数值口径。"""
from fractions import Fraction


def parse_fraction(x):
    """把 '19/20'、'-1'、'0.25' 等严格解析为 Fraction，拒绝非字符串。"""
    if not isinstance(x, str) or not x:
        raise ValueError('有理数量必须是非空字符串')
    return Fraction(x)


def parse_confidence(x):
    """置信度字符串 '19/20' -> (Fraction 水平, 尾概率 Fraction)。"""
    level = parse_fraction(x)
    if not 0 < level < 1:
        raise ValueError('置信水平必须在 (0,1) 内')
    return level, 1 - level
