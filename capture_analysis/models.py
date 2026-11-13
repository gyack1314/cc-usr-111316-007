"""逐分层模型编排、AICc 与模型平均。

- 所有模型使用同一观测格（结构零除外）的完整 Poisson 似然，含阶乘常数；
- k = 可识别自由参数数（满秩设计的列数）；n = 该层观测家庭数；
- n <= k+1 时 AICc 不定义（返回 None），与优化失败/不可识别分别标记；
- 模型平均仅在可估模型间按 AICc 权重平均点估计，绝不平均区间端点；
- 总人数 = 观测并集 n_obs + lambda0；000 为结构零时固定 0，不外推。
"""
import math
from .design import build_design
from .fit import fit_poisson, OK, DESIGN_RANK_DEFICIENT, MLE_INFINITE, OPTIMIZATION_FAILED
from .profile import profile_interval, zero_design_row


def aicc(loglik, k, n):
    if n <= k + 1:
        return None
    return -2 * loglik + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def _lambda0_from_fit(design, fit_result, history):
    if history['zero_is_structural']:
        return 0.0
    c = zero_design_row(design, history)
    return math.exp(sum(c[j] * fit_result['beta'][j] for j in range(len(c))))


def run_model(history, model_spec, alpha=0.05, tol=1e-10):
    """对单个模型在全部分层上拟合，返回完整审计结果。"""
    interactions = model_spec['interactions']
    per_stratum = {}
    for stratum in history['strata']:
        design = build_design(history, interactions)
        counts = history['tables'][stratum]
        n_obs = history['observed_counts'][stratum]
        entry = {'n_observed': n_obs, 'design_rank': design['rank_exact'],
                 'n_params_full': design['n_columns'],
                 'rank_numeric': design['rank_numeric'],
                 'terms': [list(t) if t != ('1',) else ['(intercept)']
                           for t in design['terms']],
                 'cell_order': design['masks']}
        fit = fit_poisson(design, counts, tol=tol)
        entry['status'] = fit['status']
        entry['status_reason'] = fit['reason']

        if fit['status'] != OK:
            if fit['status'] == MLE_INFINITE:
                entry['divergence'] = fit['divergence']
                entry['probe_loglik'] = fit['probe']['loglik']
            if fit['status'] == OPTIMIZATION_FAILED:
                entry['best_loglik'] = fit['best_loglik']
                entry['info'] = {'min_eigenvalue': fit['info']['min_eigenvalue'],
                                 'condition': fit['info']['condition']}
            per_stratum[stratum] = entry
            continue

        entry['beta'] = fit['beta']
        entry['fitted_mu'] = fit['fitted']
        entry['loglik'] = fit['loglik']
        entry['iterations'] = fit['iterations']
        entry['grad_inf'] = fit['grad_inf']
        entry['info_min_eigenvalue'] = fit['info']['min_eigenvalue']
        entry['info_condition'] = fit['info']['condition']
        entry['info_near_singular'] = fit['info']['near_singular']
        entry['k_free'] = fit['k_free']
        entry['aicc'] = aicc(fit['loglik'], fit['k_free'], n_obs)
        entry['aicc_undefined_note'] = (
            f'n={n_obs} <= k+1={fit["k_free"] + 1}' if entry['aicc'] is None else None)

        if history['zero_is_structural']:
            entry['lambda0'] = 0.0
            entry['lambda0_interval'] = None
            entry['lambda0_identifiable'] = True
            entry['lambda0_note'] = '000 为结构零：未观测数固定为 0，不做外推'
            entry['n_total_point'] = float(n_obs)
            entry['n_total_interval'] = [float(n_obs), float(n_obs)]
        else:
            lam = _lambda0_from_fit(design, fit, history)
            entry['lambda0'] = lam
            ci = profile_interval(design, fit, history, alpha=alpha, tol=tol)
            entry['lambda0_interval'] = ci
            # 可识别性：轮廓区间有限且点估计不在零边界。
            # 上界无穷（含点估计被压到 0 的饱和情形）意味着 lambda0 无有限 MLE 信息，
            # 其点估计只是边界极限，不得当作可平均的人数估计。
            entry['lambda0_identifiable'] = (
                not ci['upper_infinite'] and ci['lower'] > 0.0)
            if not entry['lambda0_identifiable']:
                entry['lambda0_identifiability_note'] = (
                    'λ₀ 轮廓区间触及 0 或上界无穷：该模型对全零格均值不可识别，'
                    '点估计为边界极限，不参与 AICc 点估计平均')
            entry['n_total_point'] = n_obs + lam
            entry['n_total_interval'] = [n_obs + ci['lower'],
                                         (math.inf if ci['upper_infinite']
                                          else n_obs + ci['upper'])]
        per_stratum[stratum] = entry

    # 跨层汇总
    estimable = [s for s, e in per_stratum.items() if e['status'] == OK]
    overall = {
        'n_observed_total': sum(history['observed_counts'].values()),
        'estimable_strata': estimable,
        'rejected_strata': {s: {'status': e['status'], 'reason': e['status_reason']}
                            for s, e in per_stratum.items() if e['status'] != OK},
    }
    if estimable and not history['zero_is_structural']:
        overall['n_total_point'] = sum(per_stratum[s]['n_total_point'] for s in estimable)
        # 逐层端点相加的包络：明确标注非联合似然区间
        lo = sum(per_stratum[s]['n_total_interval'][0] for s in estimable)
        ups = [per_stratum[s]['n_total_interval'][1] for s in estimable]
        hi = math.inf if any(math.isinf(u) for u in ups) else sum(ups)
        overall['n_total_envelope_interval'] = [lo, hi]
        overall['envelope_note'] = '各层轮廓区间端点之和，是保守包络而非联合似然区间'
    elif estimable:
        overall['n_total_point'] = float(overall['n_observed_total'])
        overall['n_total_envelope_interval'] = [float(overall['n_observed_total']),
                                                float(overall['n_observed_total'])]
    return {'model_id': model_spec['id'], 'interactions': model_spec['interactions'],
            'strata': per_stratum, 'overall': overall}


def model_averaged_estimate(model_results):
    """仅对可估且 AICc 有定义的模型/分层做 AICc 权重平均点估计。"""
    out = {}
    strata = next(iter(model_results))['strata'].keys() if model_results else []
    for stratum in strata:
        rows = [r for r in model_results
                if r['strata'][stratum]['status'] == OK
                and r['strata'][stratum]['aicc'] is not None
                and r['strata'][stratum].get('lambda0_identifiable', False)]
        excluded = [r['model_id'] for r in model_results
                    if r['strata'][stratum]['status'] == OK
                    and not r['strata'][stratum].get('lambda0_identifiable', False)]
        if not rows:
            out[stratum] = {'eligible_models': [],
                            'excluded_unidentifiable_models': excluded,
                            'note': '没有可参与平均的可估且 λ₀ 可识别模型'}
            continue
        best = min(r['strata'][stratum]['aicc'] for r in rows)
        weights = {r['model_id']: math.exp(-0.5 * (r['strata'][stratum]['aicc'] - best))
                   for r in rows}
        wsum = sum(weights.values())
        weights = {k: v / wsum for k, v in weights.items()}
        avg_lambda0 = sum(weights[r['model_id']] * r['strata'][stratum]['lambda0']
                          for r in rows)
        n_obs = rows[0]['strata'][stratum]['n_observed']
        out[stratum] = {
            'eligible_models': [r['model_id'] for r in rows],
            'excluded_unidentifiable_models': excluded,
            'aicc_weights': weights,
            'lambda0_avg': avg_lambda0,
            'n_total_avg': n_obs + avg_lambda0,
            'note': '点估计为 AICc 权重平均；区间不得对各模型端点加权，未提供平均区间',
        }
    return out


def run_all(history, model_specs, alpha=0.05, tol=1e-10):
    results = [run_model(history, spec, alpha=alpha, tol=tol) for spec in model_specs]
    average = model_averaged_estimate(results)
    return {'models': results, 'model_average': average}
