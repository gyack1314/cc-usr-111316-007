"""两条预登记敏感性路径。

路径 A（未观测格）：对每个未观测但可能的格子（含 000）逐个注入预设的小整数计数，
在同一模型族上重拟合，观察 lambda0/总人数的变化。零计数本身不加 0.5，
注入值是显式的假设情景而非数据修正。

路径 B（最高阶依赖）：把完整 k 渠道交互系数固定为预设值，其余参数重拟合，
报告无法由无名单家庭验证的这条假设对遗漏估计的影响。
渠道独立性无法从无名单家庭验证——该声明随结果一起输出。
"""
import math
from .design import build_design
from .fit import constrained_fit, expand_beta, poisson_loglik, OK
from .profile import zero_design_row

DEFAULT_PERTURBATIONS = (1, 2, 5, 10)


def _highest_order_term(history):
    return tuple(history['channels'])


def highest_order_sensitivity(history, model_spec, fixed_values, tol=1e-10):
    """路径 B：固定完整 k 渠道交互系数 c'beta = v，其余参数重拟合。

    若原模型不含最高阶项，则按层级原则临时补入该项（及缺失的低阶项），
    在增广模型上固定系数。补入意味着这是超出原模型的显式假设，
    固定值 0 对应原模型隐含的渠道独立性，其余值展示依赖假设的影响。
    """
    ho = tuple(sorted(_highest_order_term(history)))
    declared = [list(t) for t in model_spec['interactions']]
    has_ho = ho in tuple(sorted(t) for t in declared)
    # 层级补全：加入最高阶项所需的全部真子集
    from itertools import combinations
    augmented = [list(t) for t in declared]
    seen = {tuple(sorted(t)) for t in augmented}
    for size in range(1, len(ho) + 1):
        for sub in combinations(ho, size):
            if sub not in seen:
                augmented.append(list(sub))
                seen.add(sub)

    design = build_design(history, augmented)
    if design['rank_deficient']:
        return {'path': 'highest_order_dependence', 'model_id': model_spec['id'],
                'status': 'design_rank_deficient',
                'reason': '增广后设计秩不足，无法在固定最高阶系数下重拟合',
                'unverifiable_note': _unverifiable_note(history),
                'points': []}
    term_cols = {tuple(t): j for j, t in enumerate(design['terms'])}
    j = term_cols[ho]
    out = {'path': 'highest_order_dependence', 'model_id': model_spec['id'],
           'highest_order_term': ho,
           'term_present_in_model': has_ho,
           'augmentation_note': (None if has_ho else
                                 '原模型不含最高阶项，已按层级原则临时补入；'
                                 '固定值 0 即原模型隐含的独立性假设'),
           'unverifiable_note': _unverifiable_note(history),
           'points': []}
    c = [1.0 if t == j else 0.0 for t in range(len(design['terms']))]
    for stratum in history['strata']:
        counts = history['tables'][stratum]
        y = [counts[m] for m in design['masks']]
        n_obs = history['observed_counts'][stratum]
        for v in fixed_values:
            res = constrained_fit(design['matrix'], y, c, float(v), tol=tol)
            if not res['converged']:
                out['points'].append({'stratum': stratum, 'fixed_value': v,
                                      'status': 'optimization_failed'})
                continue
            beta = expand_beta(res, c, float(v))
            c0 = zero_design_row(design, history)
            lam0 = (0.0 if history['zero_is_structural']
                    else math.exp(sum(c0[t] * beta[t] for t in range(len(beta)))))
            out['points'].append({
                'stratum': stratum, 'fixed_value': v,
                'loglik': res['loglik'],
                'lambda0': lam0,
                'n_total': n_obs + lam0,
                'fitted_mu': dict(zip(design['masks'], res['mu'])),
                'status': OK,
            })
    return out


def unobserved_cell_sensitivity(history, model_spec,
                                perturbations=DEFAULT_PERTURBATIONS, tol=1e-10):
    """路径 A：逐个向未观测可能格注入小整数计数后重拟合同一模型。"""
    design = build_design(history, model_spec['interactions'])
    out = {'path': 'unobserved_cells', 'model_id': model_spec['id'],
           'perturbations': list(perturbations),
           'injected_count_note': '注入值为假设情景，不修改原始计数，不加 0.5',
           'points': []}
    if design['rank_deficient']:
        out['status'] = 'design_rank_deficient'
        return out
    for stratum in history['strata']:
        counts = history['tables'][stratum]
        n_obs0 = history['observed_counts'][stratum]
        zero_cells = history['unobserved_possible'][stratum]
        for cell in zero_cells:
            for add in perturbations:
                y = [counts[m] + (add if m == cell else 0) for m in design['masks']]
                from .fit import _irls, _info_diagnostics
                res = _irls(design['matrix'], y, tol=tol)
                if not res['converged']:
                    out['points'].append({'stratum': stratum, 'cell': cell,
                                          'injected': add,
                                          'status': 'optimization_failed'})
                    continue
                c0 = zero_design_row(design, history)
                lam0 = (0.0 if history['zero_is_structural']
                        else math.exp(sum(c0[t] * res['beta'][t]
                                          for t in range(len(res['beta'])))))
                out['points'].append({
                    'stratum': stratum, 'cell': cell, 'injected': add,
                    'status': OK, 'loglik': res['loglik'],
                    'lambda0_under_scenario': lam0,
                    'n_total_under_scenario': (n_obs0 + add) + lam0,
                    'n_observed_scenario': n_obs0 + add,
                })
    return out


def _unverifiable_note(history):
    return (
        '全零格家庭不出现于任何名单，其计数没有任何观测似然；'
        '渠道独立性（最高阶交互=0）以及任何依赖取值都无法由无名单家庭验证，'
        f'只能作为假设固定。当前渠道数 k={history["n_channels"]}，'
        '结构上不可能的格子见结构零清单。')
