"""对数线性模型项与设计矩阵。

每个模型含截距与清单声明的登记交互项（位指标乘积）。
层级原则：任一交互项的所有非空真子集（主效应/低阶交互）必须同时声明。
结构零格不进入设计矩阵（也不参与后续预测）。
"""
from itertools import combinations
from .linalg import exact_rank, numeric_rank

INTERCEPT = ('1',)


def _term_key(term):
    return tuple(sorted(term))


def closure_terms(interactions, channels):
    """把声明项规范化并按层级原则补全/校验，返回排序后的元组项列表（不含截距）。"""
    named = set(channels)
    declared = set()
    for term in interactions:
        if not isinstance(term, list) or not term:
            raise ValueError('交互项必须是非空渠道名列表')
        key = _term_key(term)
        if any(ch not in named for ch in key):
            raise ValueError(f'交互项 {term} 含未知渠道')
        if len(set(key)) != len(key):
            raise ValueError(f'交互项 {term} 内渠道重复')
        declared.add(key)
    # 层级原则：补齐所有子集；若原始清单缺项则报错而不是静默接受
    for key in list(declared):
        for size in range(1, len(key)):
            for sub in combinations(key, size):
                if sub not in declared:
                    raise ValueError(
                        f'违反层级原则：声明了 {list(key)} 但缺少低阶项 {list(sub)}')
    return sorted(declared, key=lambda t: (len(t), t))


def term_labels(interactions, channels):
    return [INTERCEPT] + closure_terms(interactions, channels)


def design_row(mask, channels, terms):
    """一条捕获历史对应的设计行：截距 1 + 各项位指标乘积。"""
    bit = {ch: int(b) for ch, b in zip(channels, mask)}
    row = [1.0]
    for term in terms:
        v = 1.0
        for ch in term:
            v *= bit[ch]
        row.append(v)
    return row


def build_design(history, interactions):
    """为某一分层的可能格子构建设计矩阵（结构零已在 possible_masks 中排除）。"""
    channels = history['channels']
    terms = closure_terms(interactions, channels)
    labels = [INTERCEPT] + terms
    masks = history['possible_masks']
    x = [design_row(m, channels, terms) for m in masks]
    r_exact = exact_rank(x)
    r_num = numeric_rank(x)
    return {
        'terms': labels,
        'masks': masks,
        'matrix': x,
        'n_columns': len(labels),
        'rank_exact': r_exact,
        'rank_numeric': r_num,
        'rank_deficient': r_exact < len(labels),
    }
