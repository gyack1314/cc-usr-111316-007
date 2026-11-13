"""家庭行 -> 捕获历史表。

口径（见 docs/data-contract.md）：
- 家庭身份已由人工核验的稳定 ID 保证，本模块不做姓名匹配；
- 重复家庭按位 OR 合并位图，分层不一致直接阻断（StratumConflictError）；
- 每个家庭至少一个位为 1；结构零全分层统一，结构零中出现观测即输入矛盾；
- 输出枚举 2^k 个格子：观测格计数、未观测但可能的格子显式记 0、结构零单列。
"""
from itertools import product


class StratumConflictError(ValueError):
    """同一家庭 ID 出现在不同登记分层。"""


def all_masks(k):
    return [''.join(bits) for bits in product('01', repeat=k)]


def zero_mask(k):
    return '0' * k


def _check_mask(mask, k):
    if not isinstance(mask, str) or len(mask) != k or set(mask) - {'0', '1'}:
        raise ValueError(f'位图 {mask!r} 与渠道数 {k} 不符')


def _or_masks(a, b):
    return ''.join('1' if x == '1' or y == '1' else '0' for x, y in zip(a, b))


def build_history(rows, channels, structural_zero_masks):
    """构造分层捕获历史表。

    rows: [{'household': id, 'stratum': s, 'mask': m}, ...]
    返回 dict，含完整格子枚举、合并审计与结构零标记。
    """
    k = len(channels)
    if not 3 <= k <= 6:
        raise ValueError('渠道数必须在 3 到 6 之间')
    if len(set(channels)) != k:
        raise ValueError('渠道名称重复')

    structural = list(structural_zero_masks)
    for m in structural:
        _check_mask(m, k)
    if len(set(structural)) != len(structural):
        raise ValueError('结构零位图重复声明')
    structural_set = set(structural)

    # 家庭 -> (合并位图, 分层)；保留首次出现行号做审计
    merged = {}
    order = []
    merge_log = []
    for idx, r in enumerate(rows):
        hid = r['household']
        stratum = r['stratum']
        mask = r['mask']
        _check_mask(mask, k)
        if mask == zero_mask(k):
            raise ValueError(f'家庭 {hid!r} 位图全零，已观测家庭至少出现于一个渠道')
        if not isinstance(stratum, str) or not stratum:
            raise ValueError(f'家庭 {hid!r} 分层缺失')
        if hid not in merged:
            merged[hid] = [mask, stratum, idx]
            order.append(hid)
        else:
            old_mask, old_stratum, first_idx = merged[hid]
            if old_stratum != stratum:
                raise StratumConflictError(
                    f'家庭 {hid!r} 分层冲突：第 {first_idx} 行 {old_stratum!r} '
                    f'vs 第 {idx} 行 {stratum!r}')
            new_mask = _or_masks(old_mask, mask)
            if new_mask != old_mask:
                merge_log.append({'household': hid, 'stratum': stratum,
                                  'from': old_mask, 'added': mask, 'to': new_mask})
            merged[hid][0] = new_mask

    # 结构零中不得出现观测
    for hid, (mask, stratum, _) in merged.items():
        if mask in structural_set:
            raise ValueError(
                f'结构零格 {mask} 中出现观测家庭 {hid!r}（分层 {stratum!r}），输入矛盾')

    # 分层 -> 格 -> 计数（只填非结构零）
    strata = sorted({v[1] for v in merged.values()})
    full = all_masks(k)
    tables = {}
    for s in strata:
        cell = {m: 0 for m in full if m not in structural_set}
        for hid in order:
            mask, st, _ = merged[hid]
            if st == s:
                cell[mask] += 1
        tables[s] = cell

    observed_counts = {s: sum(c for m, c in cell.items() if m != zero_mask(k))
                       for s, cell in tables.items()}
    possible_masks = [m for m in full if m not in structural_set]
    unobserved_possible = {s: [m for m, c in cell.items() if c == 0]
                           for s, cell in tables.items()}

    return {
        'channels': list(channels),
        'n_channels': k,
        'structural_zero_masks': structural,
        'zero_is_structural': zero_mask(k) in structural_set,
        'all_zero_possible': zero_mask(k) not in structural_set,
        'possible_masks': possible_masks,
        'strata': strata,
        'tables': tables,
        'observed_counts': observed_counts,
        'unobserved_possible': unobserved_possible,
        'raw_rows': len(rows),
        'merged_families': len(merged),
        'duplicate_rows': len(rows) - len(merged),
        'merge_log': merge_log,
        'zero_mask': zero_mask(k),
    }
