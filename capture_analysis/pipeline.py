"""端到端管线：冻结清单 -> 历史表 -> 模型拟合 -> 区间/平均 -> 敏感性 -> 报告。"""
import json
from pathlib import Path

from analysis_data.io import load, asset
from .histories import build_history, zero_mask
from .models import run_all
from .sensitivity import highest_order_sensitivity, unobserved_cell_sensitivity
from .rational import parse_fraction, parse_confidence


def load_manifest(path):
    """经只读模块加载并校验资产身份，返回 (manifest dict, sha256, 行列表)。"""
    manifest, sha = load(path)
    channels = manifest['channels']
    if not 3 <= len(channels) <= 6 or len(set(channels)) != len(channels):
        raise ValueError('渠道数或渠道身份错误')
    root = Path(path).parent
    raw = asset(root, manifest['families'])
    rows = [json.loads(line) for line in raw.decode('utf8').splitlines() if line]
    return manifest, sha, rows


def analyze(path, perturbations=(1, 2, 5, 10)):
    manifest, sha, rows = load_manifest(path)
    channels = manifest['channels']
    structural = list(manifest.get('structural_zero_masks', []))
    history = build_history(rows, channels, structural)

    level, tail = parse_confidence(manifest['profile_confidence'])
    alpha = float(tail)
    tol = float(parse_fraction(manifest.get('numeric_tolerance', '1/100000000')))

    fitted = run_all(history, manifest['models'], alpha=alpha, tol=tol)

    fixed_values = [float(parse_fraction(v))
                    for v in manifest.get('sensitivity', {})
                    .get('highest_order_log_interaction', [])]
    sens = []
    for spec in manifest['models']:
        if fixed_values:
            sens.append(highest_order_sensitivity(history, spec, fixed_values, tol=tol))
        sens.append(unobserved_cell_sensitivity(
            history, spec, perturbations=perturbations, tol=tol))

    assumptions = _assumptions(history, manifest)
    return {'manifest_sha256': sha, 'manifest_path': str(path),
            'confidence_level': str(level), 'alpha': alpha,
            'numeric_tolerance': str(tol),
            'history': _history_view(history),
            'assumptions': assumptions,
            'model_results': fitted['models'],
            'model_average': fitted['model_average'],
            'sensitivity': sens}


def _history_view(history):
    """可回溯视图：每个分层每个格子的计数与归类。"""
    strata = {}
    z = zero_mask(history['n_channels'])
    for s in history['strata']:
        cells = []
        for m in history['possible_masks']:
            c = history['tables'][s][m]
            kind = 'unobserved_possible' if c == 0 else 'observed'
            if m == z:
                kind = 'all_zero_unobserved'
            cells.append({'mask': m, 'count': c, 'kind': kind})
        strata[s] = {
            'cells': cells,
            'n_observed_union': history['observed_counts'][s],
            'unobserved_possible_masks': history['unobserved_possible'][s],
        }
    return {
        'channels': history['channels'],
        'n_channels': history['n_channels'],
        'structural_zero_masks': history['structural_zero_masks'],
        'all_zero_mask': z,
        'all_zero_structural': history['zero_is_structural'],
        'all_zero_possible': not history['zero_is_structural'],
        'strata': strata,
        'raw_rows': history['raw_rows'],
        'merged_families': history['merged_families'],
        'duplicate_rows_merged': history['duplicate_rows'],
        'merge_log': history['merge_log'],
        'reconciliation': {
            'sum_cell_counts': {s: sum(history['tables'][s].values())
                                for s in history['strata']},
            'merged_families': history['merged_families'],
        },
    }


def _assumptions(history, manifest):
    return [
        '研究窗口内总体封闭：无家庭进出、无渠道覆盖范围变化（使用者承担，数据无法验证）。',
        '家庭身份为人工核验的稳定 ID，误链接已消除；本次不做姓名匹配，ID 匹配不确定性不进入遗漏估计。',
        '分层独立拟合，层间不共享参数；同一家庭重复行按位 OR 合并，分层冲突即阻断。',
        f'结构零（全分层统一）：{history["structural_zero_masks"] or "无"}；'
        f'全零格 {history["zero_mask"]} '
        + ('是结构零，未观测数固定为 0。' if history['zero_is_structural']
           else '不是结构零，其均值由模型参数外推，本身没有观测似然。'),
        '饱和或其他设计秩不足的模型不产生人数估计；边界发散与优化未收敛分别报告。',
        '轮廓似然区间为给定模型下的渐近卡方(1)区间；区间很宽或上界无穷是正式结果。',
        '渠道独立性无法从无名单家庭验证：任何最高阶依赖取值都是假设，见敏感性路径 B。',
    ]
