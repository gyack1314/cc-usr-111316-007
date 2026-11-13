"""把分析结果渲染为 Markdown 摘要；JSON 由 CLI 直接序列化。"""
import math


def _num(x, digits=4):
    if x is None:
        return '—'
    if isinstance(x, str):
        return x
    if math.isinf(x):
        return '∞'
    if isinstance(x, int) or (isinstance(x, float) and x.is_integer() and abs(x) < 1e15):
        return str(int(x))
    return f'{x:.{digits}f}'


def _status_cn(status):
    return {
        'ok': '可估',
        'design_rank_deficient': '设计秩不足（不可识别）',
        'mle_infinite': '有限 MLE 不存在（边界发散）',
        'optimization_failed': '优化未收敛',
    }.get(status, status)


def render_markdown(result):
    lines = []
    h = result['history']
    lines.append('# 多名单捕获分析报告')
    lines.append('')
    lines.append(f"- 清单指纹 `{result['manifest_sha256']}`")
    lines.append(f"- 渠道（{h['n_channels']}）：{', '.join(h['channels'])}")
    lines.append(f"- 结构零格子：{', '.join(h['structural_zero_masks']) or '无'}；"
                 f"全零格 {h['all_zero_mask']} "
                 + ('为结构零（未观测固定 0）' if h['all_zero_structural']
                    else '可能出现但未观测（需外推）'))
    lines.append(f"- 原始行 {h['raw_rows']}，合并后家庭 {h['merged_families']}，"
                 f"合并重复行 {h['duplicate_rows_merged']}")
    lines.append(f"- 轮廓置信水平 {result['confidence_level']}（渐近卡方1阈值）")

    lines.append('')
    lines.append('## 1. 捕获历史表（可回溯）')
    for s, view in h['strata'].items():
        lines.append('')
        lines.append(f"### 分层 {s}：观测并集 {view['n_observed_union']} 户")
        lines.append('| 位图 | 计数 | 归类 |')
        lines.append('| --- | ---: | --- |')
        for c in view['cells']:
            lines.append(f"| {c['mask']} | {c['count']} | {c['kind']} |")
        recon = sum(c['count'] for c in view['cells'])
        lines.append(f"_对账：格子计数之和 {recon} = 合并家庭数_")

    lines.append('')
    lines.append('## 2. 模型拟合结果')
    for mr in result['model_results']:
        lines.append('')
        lines.append(f"### 模型 `{mr['model_id']}`")
        for s, e in mr['strata'].items():
            lines.append(f"- 分层 {s}：**{_status_cn(e['status'])}**"
                         + (f" — {e['status_reason']}" if e['status_reason'] else ''))
            if e['status'] != 'ok':
                if e['status'] == 'mle_infinite':
                    for zc in e['divergence']['zero_cells_to_zero']:
                        lines.append(f"  - 零格 {zc['mask']} 拟合值趋于 {zc['fitted_mu']:.2e}")
                continue
            lines.append(f"  - 设计秩 {e['design_rank']}/{e['n_params_full']}，"
                         f"自由参数 k={e['k_free']}，迭代 {e['iterations']}，"
                         f"梯度∞ {e['grad_inf']:.2e}")
            lines.append(f"  - 信息矩阵最小特征值 {e['info_min_eigenvalue']:.3e}，"
                         f"条件数 {_num(e['info_condition'])}"
                         + ('，近奇异' if e['info_near_singular'] else ''))
            lines.append(f"  - 对数似然 {e['loglik']:.6f}，AICc = {_num(e['aicc'])}"
                         + (f"（不定义：{e['aicc_undefined_note']}）"
                            if e['aicc'] is None else ''))
            if e.get('lambda0_interval'):
                ci = e['lambda0_interval']
                ident = '' if e.get('lambda0_identifiable') else '（λ₀ 不可识别，不参与平均）'
                lines.append(f"  - λ₀ 点估计 {e['lambda0']:.4f}，"
                             f"区间 [{_num(ci['lower'])}, {_num(ci['upper'])}]"
                             + ('（上界无穷，保留）' if ci['upper_infinite'] else '')
                             + ident)
            elif 'lambda0_note' in e:
                lines.append(f"  - λ₀：{e['lambda0_note']}")
            lines.append(f"  - 总人数点估计 {_num(e['n_total_point'])}，"
                         f"区间 [{_num(e['n_total_interval'][0])}, "
                         f"{_num(e['n_total_interval'][1])}]")

    lines.append('')
    lines.append('## 3. AICc 模型平均（仅点估计）')
    for s, a in result['model_average'].items():
        if not a.get('eligible_models'):
            lines.append(f"- 分层 {s}：{a['note']}")
            continue
        w = ', '.join(f"{m}={w:.4f}" for m, w in a['aicc_weights'].items())
        lines.append(f"- 分层 {s}：权重（{w}）；平均 λ₀={a['lambda0_avg']:.4f}，"
                     f"平均总人数={a['n_total_avg']:.2f}")
        lines.append(f"  - {a['note']}")

    lines.append('')
    lines.append('## 4. 敏感性路径')
    for sp in result['sensitivity']:
        lines.append('')
        name = 'B 最高阶依赖固定' if sp['path'] == 'highest_order_dependence' \
            else 'A 未观测格注入'
        lines.append(f"### {name}（模型 `{sp['model_id']}`）")
        if sp.get('status') == 'design_rank_deficient':
            lines.append(f"- 跳过：{sp.get('reason', '设计秩不足')}")
            if 'unverifiable_note' in sp:
                lines.append(f"- {sp['unverifiable_note']}")
            continue
        if sp['path'] == 'highest_order_dependence':
            lines.append(f"- 最高阶项 {sp.get('highest_order_term')}，"
                         f"模型是否含该项：{sp['term_present_in_model']}")
            lines.append(f"- {sp['unverifiable_note']}")
            lines.append('| 分层 | 固定值 | λ₀ | 总人数 | 对数似然 |')
            lines.append('| --- | ---: | ---: | ---: | ---: |')
            for p in sp['points']:
                if p.get('status') == 'optimization_failed':
                    lines.append(f"| {p['stratum']} | {p['fixed_value']} | 未收敛 | — | — |")
                elif 'fixed_value' in p:
                    lines.append(
                        f"| {p['stratum']} | {p['fixed_value']} | "
                        f"{_num(p.get('lambda0'))} | {_num(p.get('n_total'))} | "
                        f"{_num(p.get('loglik'), 4)} |")
                else:
                    lines.append(f"| {p['stratum']} | 0（隐式） | — | — | {p['note']} |")
        else:
            lines.append(f"- {sp['injected_count_note']}")
            lines.append('| 分层 | 注入格 | 注入数 | 情景 λ₀ | 情景总人数 |')
            lines.append('| --- | --- | ---: | ---: | ---: |')
            for p in sp['points']:
                if p.get('status') == 'optimization_failed':
                    lines.append(f"| {p['stratum']} | {p['cell']} | {p['injected']} | 未收敛 | — |")
                else:
                    lines.append(
                        f"| {p['stratum']} | {p['cell']} | {p['injected']} | "
                        f"{_num(p['lambda0_under_scenario'])} | "
                        f"{_num(p['n_total_under_scenario'])} |")

    lines.append('')
    lines.append('## 5. 假设与不可验证部分')
    for a in result['assumptions']:
        lines.append(f'- {a}')
    return '\n'.join(lines) + '\n'
