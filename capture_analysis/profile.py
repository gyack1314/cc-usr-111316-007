"""全零格均值 lambda0 的轮廓似然区间。

lambda0 = exp(c'beta)，c 为全零格设计行（即截距列）；轮廓约束 c'beta = log(v)。
区间阈值：2[loglik(beta_hat) - max_{c'beta=log v} loglik(beta)] = chi2_1(alpha)。
若在可行上确界处轮廓统计量仍未越过阈值，上界记为无穷并保留，
不通过外推编造有限端点。零计数不加 0.5。约束点沿序列热启动。
"""
import math
from .fit import constrained_fit, expand_beta
from .stats import chi2_1_quantile

# log(lambda0) 的搜索边界：再小/再大对计数数据没有可辨识差别
_LOG_FLOOR = -40.0
_LOG_CEIL = 40.0


def zero_design_row(design, history):
    """全零格设计行：截距为 1，所有位指标乘积项为 0。"""
    return [1.0] + [0.0] * (len(design['terms']) - 1)


def profile_loglik(x, y, c, log_v, tol=1e-10, beta_start=None):
    # 热启动优先；失败（慢收敛/线搜索失败）回退冷启动，二者皆败才上报
    res = constrained_fit(x, y, c, log_v, tol=tol, beta_start_full=beta_start)
    if not res['converged'] and beta_start is not None:
        res = constrained_fit(x, y, c, log_v, tol=tol, beta_start_full=None)
    if not res['converged']:
        raise RuntimeError(f'轮廓拟合未收敛（log_v={log_v:.4f}，{res["reason"]}）')
    beta = expand_beta(res, c, log_v)
    return res['loglik'], beta, res['mu']


def _bisect(stat_fn, lo, hi, f_lo, f_hi, threshold, tol_stat=1e-9,
            tol_x=1e-10, max_iter=100):
    """在 log(lambda0) 空间对 stat_fn=threshold 二分，不预设单调方向。

    维护不变量：阈值始终介于两端点统计量之间。每步用中点统计量替换
    与其同侧的端点。调用方保证 threshold 跨过 [f_lo, f_hi]。
    """
    a, b, fa, fb = lo, hi, f_lo, f_hi
    if not ((fa <= threshold <= fb) or (fb <= threshold <= fa)):
        raise ValueError('二分括号未跨过阈值')
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = stat_fn(mid)
        if abs(fm - threshold) <= tol_stat * threshold or (b - a) < tol_x:
            return mid
        # mid 与 a 在阈值同侧 -> 用 mid 替换 a；否则替换 b
        a_side = (fa - threshold) * (fm - threshold) > 0
        if a_side:
            a, fa = mid, fm
        else:
            b, fb = mid, fm
    return 0.5 * (a + b)


def profile_interval(design, fit_result, history, alpha=0.05, tol=1e-10):
    """lambda0 点估计与轮廓似然区间（端点为 log 空间二分的结果）。"""
    c = zero_design_row(design, history)
    x = design['matrix']
    y = list(fit_result['y'])
    ll_max = fit_result['loglik']
    threshold = chi2_1_quantile(alpha)
    point = math.exp(sum(c[j] * fit_result['beta'][j] for j in range(len(c))))
    log_point = math.log(point)

    warm = list(fit_result['beta'])

    def stat_at_log(lv):
        nonlocal warm
        ll, beta, _ = profile_loglik(x, y, c, lv, tol=tol, beta_start=warm)
        warm = beta
        return 2.0 * (ll_max - ll)

    # ---------- 下界：从 log_point 向 _LOG_FLOOR 扫描括号 ----------
    lower = 0.0
    grid_logs = []
    prev_lv = log_point
    prev_s = 0.0
    lv = log_point
    bracket = None
    while lv > _LOG_FLOOR:
        lv = max(lv - 1.0, _LOG_FLOOR)
        s = stat_at_log(lv)
        grid_logs.append((lv, s))
        if s >= threshold and bracket is None:
            bracket = (prev_lv, lv, prev_s, s)
            break
        prev_lv, prev_s = lv, s
    if bracket is not None:
        # 规范化为递增 log 区间（下界扫描方向是递减，括号可能反向）
        blo, bhi = sorted((bracket[0], bracket[1]))
        fblo = bracket[2] if bracket[0] == blo else bracket[3]
        fbhi = bracket[3] if bracket[1] == bhi else bracket[2]
        log_lower = _bisect(
            lambda z: stat_at_log(z), blo, bhi, fblo, fbhi, threshold)
        lower = math.exp(log_lower)
    else:
        # 到搜索地板仍未越阈：下界保持 0（极少见，记录证据）
        lower = 0.0

    # ---------- 上界：从 log_point 向 _LOG_CEIL 扫描 ----------
    warm = list(fit_result['beta'])
    upper = math.inf
    upper_platform = None
    lv = log_point
    prev_lv, prev_s = log_point, 0.0
    bracket_hi = None
    while lv < _LOG_CEIL:
        lv = min(lv + 1.0, _LOG_CEIL)
        s = stat_at_log(lv)
        grid_logs.append((lv, s))
        if s >= threshold:
            bracket_hi = (prev_lv, lv, prev_s, s)
            break
        # 平台化检测：连续两步统计量几乎不变且远低于阈值
        if abs(s - prev_s) < 1e-6 and s < 0.5 * threshold and lv - log_point > 4:
            upper_platform = s
            break
        prev_lv, prev_s = lv, s
    if bracket_hi is not None:
        log_upper = _bisect(
            lambda z: stat_at_log(z),
            bracket_hi[0], bracket_hi[1], bracket_hi[2], bracket_hi[3], threshold)
        upper = math.exp(log_upper)
    elif upper_platform is None:
        # 抵达 _LOG_CEIL 仍未越阈：按无穷处理，记录天花板处统计量
        upper_platform = s

    grid = [{'log_lambda0': lv, 'lambda0': math.exp(lv), 'statistic': s}
            for lv, s in sorted(grid_logs)]
    return {
        'point': point,
        'lower': lower,
        'upper': upper,
        'upper_infinite': not math.isfinite(upper),
        'upper_platform_statistic': upper_platform,
        'threshold': threshold,
        'alpha': alpha,
        'profile_grid': grid,
        'search_log_bounds': [_LOG_FLOOR, _LOG_CEIL],
        'method': 'profile_likelihood_chi2_1_asymptotic',
    }
