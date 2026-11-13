"""Poisson 对数线性拟合与可估性诊断。

状态机（每个模型/分层一个结果）：
- ok                    有限 MLE，优化收敛，信息矩阵非奇异
- design_rank_deficient 设计秩 < 参数数（如饱和模型），不做拟合
- mle_infinite          有限 MLE 不存在：零计数格上的 MLE 存在性 LP 不可行，
                        系数向边界发散，零格拟合一律 0，lambda0 不可估
- optimization_failed   LP 可行（理论上有限 MLE 存在）但数值优化未收敛，
                        与结构性不可识别分别报告

零计数绝不加 0.5；Poisson 似然含 log(y!) 阶乘常数（常数不影响区间但影响审计）。
"""
import math
from .linalg import cholesky, chol_solve, jacobi_eigen, lp_feasible_ge

OK = 'ok'
DESIGN_RANK_DEFICIENT = 'design_rank_deficient'
MLE_INFINITE = 'mle_infinite'
OPTIMIZATION_FAILED = 'optimization_failed'


def _safe_exp(e):
    if e > 700.0:
        return math.inf
    if e < -700.0:
        return 0.0
    return math.exp(e)


def poisson_loglik(y, mu):
    total = 0.0
    for yi, mui in zip(y, mu):
        total += (yi * math.log(mui) if yi > 0 and mui > 0 else 0.0) \
            - mui - math.lgamma(yi + 1)
    return total


def _irls(x, y, offset=None, beta_start=None, tol=1e-10, max_iter=500):
    """带可选偏置的 Poisson IRLS，阻尼线搜索保证对数似然单调。"""
    m, p = len(x), len(x[0])
    if offset is None:
        offset = [0.0] * m
    beta = list(beta_start) if beta_start is not None else [0.0] * p
    eta = [offset[i] + sum(x[i][j] * beta[j] for j in range(p)) for i in range(m)]
    mu = [_safe_exp(e) for e in eta]
    ll = poisson_loglik(y, mu)
    history = []
    for it in range(1, max_iter + 1):
        grad = [sum(x[i][j] * (y[i] - mu[i]) for i in range(m)) for j in range(p)]
        # 标准 IRLS 法方程：X'WX delta = X'(y-mu)，偏置经 mu=exp(o+Xb) 生效
        info = [[sum(x[i][a] * mu[i] * x[i][b] for i in range(m))
                 for b in range(p)] for a in range(p)]
        rhs = [sum(x[i][j] * (y[i] - mu[i]) for i in range(m)) for j in range(p)]
        try:
            # 牛顿/IRLS 增量：(X'WX) delta = X'(y-mu)
            step_dir = chol_solve(cholesky(info), rhs)
        except ValueError:
            step_dir = None
        if step_dir is None:
            # 信息矩阵病态：退回梯度方向小步
            step_dir = [g * 1e-4 for g in grad]
        beta_try, mu_try, ll_try, eta_try = beta, mu, ll, eta
        accepted = False
        step = 1.0
        for _ in range(100):
            beta_try = [beta[j] + step * step_dir[j] for j in range(p)]
            eta_try = [offset[i] + sum(x[i][j] * beta_try[j] for j in range(p))
                       for i in range(m)]
            mu_try = [_safe_exp(e) for e in eta_try]
            ll_try = poisson_loglik(y, mu_try)
            # 相对容差：大似然绝对值下允许微小数值回退
            tol_ll = 1e-12 * max(1.0, abs(ll))
            if math.isfinite(ll_try) and ll_try >= ll - tol_ll:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            return {'converged': False, 'beta': beta, 'mu': mu, 'loglik': ll,
                    'iterations': it, 'grad_inf': max(map(abs, grad)),
                    'reason': 'line_search_failed'}
        db = max(abs(beta_try[j] - beta[j]) for j in range(p))
        beta, eta, mu, ll = beta_try, eta_try, mu_try, ll_try
        grad_inf = max(abs(sum(x[i][j] * (y[i] - mu[i]) for i in range(m)))
                       for j in range(p))
        history.append(ll)
        # 梯度（一阶最优性）为主判据；参数增量用独立的机器精度级阈值
        if grad_inf <= tol or (grad_inf <= 1e-8 and db <= 1e-10):
            return {'converged': True, 'beta': beta, 'mu': mu, 'loglik': ll,
                    'iterations': it, 'grad_inf': grad_inf, 'reason': None}
    grad_inf = max(abs(sum(x[i][j] * (y[i] - mu[i]) for i in range(m)))
                   for j in range(p))
    return {'converged': False, 'beta': beta, 'mu': mu, 'loglik': ll,
            'iterations': max_iter, 'grad_inf': grad_inf, 'reason': 'max_iter'}


def _info_diagnostics(x, mu):
    p = len(x[0])
    info = [[sum(x[i][a] * mu[i] * x[i][b] for i in range(len(x)))
             for b in range(p)] for a in range(p)]
    eig = jacobi_eigen(info)
    emin, emax = eig[0], eig[-1]
    return {'matrix': info, 'eigenvalues': eig,
            'min_eigenvalue': emin, 'condition': (emax / emin) if emin > 0 else math.inf,
            'near_singular': emin <= 1e-10 * max(1.0, emax)}


def _bfgs(x, y, offset=None, beta_start=None, tol=1e-10, max_iter=2000):
    """带 Armijo 回溯的 BFGS，作为 IRLS 在病态/紧约束下的后备。

    目标 -loglik（凸），梯度 g = X'(mu-y)。Hessian 逆用单位阵初始化，
    仅在曲率为正（y_k's_k>0）时更新，保证正定。
    """
    m, p = len(x), len(x[0])
    if offset is None:
        offset = [0.0] * m
    beta = list(beta_start) if beta_start is not None else [0.0] * p

    def moment(b):
        eta = [offset[i] + sum(x[i][j] * b[j] for j in range(p)) for i in range(m)]
        mu = [_safe_exp(e) for e in eta]
        ll = poisson_loglik(y, mu)
        g = [sum(x[i][j] * (mu[i] - y[i]) for i in range(m)) for j in range(p)]
        return mu, ll, g

    mu, ll, g = moment(beta)
    hinv = [[1.0 if i == j else 0.0 for j in range(p)] for i in range(p)]
    for it in range(1, max_iter + 1):
        ginf = max(abs(v) for v in g)
        if ginf <= tol:
            return {'converged': True, 'beta': beta, 'mu': mu, 'loglik': ll,
                    'iterations': it, 'grad_inf': ginf, 'reason': None, 'solver': 'bfgs'}
        # d = -Hinv g
        d = [-sum(hinv[a][b] * g[b] for b in range(p)) for a in range(p)]
        # 非下降方向保护：退回最速下降
        if sum(d[a] * g[a] for a in range(p)) >= 0:
            d = [-v for v in g]
            hinv = [[1.0 if i == j else 0.0 for j in range(p)] for i in range(p)]
        # Armijo 回溯
        step = 1.0
        dg = sum(d[a] * g[a] for a in range(p))
        accepted = False
        for _ in range(80):
            b1 = [beta[a] + step * d[a] for a in range(p)]
            _, ll1, g1 = moment(b1)
            if math.isfinite(ll1) and ll1 >= ll + 1e-4 * step * dg - 1e-12 * max(1, abs(ll)):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            return {'converged': False, 'beta': beta, 'mu': mu, 'loglik': ll,
                    'iterations': it, 'grad_inf': ginf,
                    'reason': 'bfgs_line_search_failed', 'solver': 'bfgs'}
        s = [step * d[a] for a in range(p)]
        beta_new = [beta[a] + s[a] for a in range(p)]
        mu, ll_new, g_new = moment(beta_new)
        yk = [g_new[a] - g[a] for a in range(p)]
        sy = sum(s[a] * yk[a] for a in range(p))
        if sy > 1e-12 * max(1.0, sum(s[a] ** 2 for a in range(p)) ** 0.5):
            # BFGS 逆更新（显式矩阵形式）
            rho = 1.0 / sy
            r = [[(1.0 if a == b else 0.0) - rho * s[a] * yk[b] for b in range(p)]
                 for a in range(p)]
            h2 = [[0.0] * p for _ in range(p)]
            for a in range(p):
                for b in range(p):
                    h2[a][b] = sum(r[a][i] * hinv[i][j] * r[b][j]
                                   for i in range(p) for j in range(p)) \
                        + rho * s[a] * s[b]
            hinv = h2
        beta, g, ll = beta_new, g_new, ll_new
    return {'converged': False, 'beta': beta, 'mu': mu, 'loglik': ll,
            'iterations': max_iter, 'grad_inf': max(abs(v) for v in g),
            'reason': 'bfgs_max_iter', 'solver': 'bfgs'}


def _info_diagnostics(x, mu):
    p = len(x[0])
    info = [[sum(x[i][a] * mu[i] * x[i][b] for i in range(len(x)))
             for b in range(p)] for a in range(p)]
    eig = jacobi_eigen(info)
    emin, emax = eig[0], eig[-1]
    return {'matrix': info, 'eigenvalues': eig,
            'min_eigenvalue': emin, 'condition': (emax / emin) if emin > 0 else math.inf,
            'near_singular': emin <= 1e-10 * max(1.0, emax)}


def finite_mle_exists(x, y):
    """Haberman (1973) 判据：存在 beta 使零计数格上 X beta >= 1。"""
    zero_rows = [row for row, yi in zip(x, y) if yi == 0]
    if not zero_rows:
        return True, None
    return lp_feasible_ge(zero_rows, 1)


def fit_poisson(design, counts, tol=1e-10):
    """对一个分层拟合格子计数。counts 与 design['masks'] 对齐。"""
    x = design['matrix']
    masks = design['masks']
    y = [counts[m] for m in masks]
    p = design['n_columns']

    if design['rank_deficient']:
        return {'status': DESIGN_RANK_DEFICIENT,
                'reason': f"设计秩 {design['rank_exact']} < 参数数 {p}",
                'design_rank': design['rank_exact'], 'n_params_full': p,
                'masks': masks, 'y': y}

    feasible, witness = finite_mle_exists(x, y)
    if not feasible:
        # 仍跑一次 IRLS 以记录发散方向（系数增大、零格 mu 趋零）
        probe = _irls(x, y, tol=tol, max_iter=100)
        divergence = _divergence_direction(design, probe, y)
        return {'status': MLE_INFINITE,
                'reason': '零计数格上的可行性 LP 无解：MLE 在参数空间边界，系数发散',
                'design_rank': design['rank_exact'], 'n_params_full': p,
                'masks': masks, 'y': y, 'probe': probe, 'divergence': divergence}

    # 有限 MLE：多起点拟合，默认起点 + LP 见证缩放起点
    attempts = [_irls(x, y, tol=tol)]
    if witness is not None:
        scale = 0.25 / max(1.0, max(map(abs, witness)))
        attempts.append(_irls(x, y, beta_start=[w * scale for w in witness], tol=tol))
    best = max(attempts, key=lambda r: r['loglik'])
    diag = _info_diagnostics(x, best['mu'])
    if not best['converged']:
        return {'status': OPTIMIZATION_FAILED,
                'reason': f"IRLS 未收敛（{best['reason']}），梯度无穷范 {best['grad_inf']:.3e}",
                'design_rank': design['rank_exact'], 'n_params_full': p,
                'masks': masks, 'y': y, 'attempts': attempts,
                'best_loglik': best['loglik'], 'info': diag}
    return {'status': OK, 'reason': None,
            'design_rank': design['rank_exact'], 'n_params_full': p,
            'k_free': p, 'beta': best['beta'],
            'fitted': dict(zip(masks, best['mu'])),
            'loglik': best['loglik'],
            'iterations': best['iterations'], 'grad_inf': best['grad_inf'],
            'masks': masks, 'y': y, 'info': diag,
            'lp_witness': witness}


def _divergence_direction(design, probe, y):
    terms = design['terms']
    big = [(terms[j], probe['beta'][j]) for j in range(len(terms))
           if abs(probe['beta'][j]) > 10.0]
    zero_cells = [{'mask': m, 'fitted_mu': mu}
                  for m, yi, mu in zip(design['masks'], y, probe['mu'])
                  if yi == 0 and mu < 1e-6]
    return {'large_coefficients': sorted(big, key=lambda t: -abs(t[1])),
            'zero_cells_to_zero': zero_cells}


# ------------------------------------------------------------ 线性约束拟合

def constrained_fit(x, y, constraint, d, offset=None, tol=1e-10,
                    beta_start_full=None, max_iter=500):
    """在 c'beta = d 下拟合 Poisson 模型（用于轮廓似然与敏感性路径）。

    选 |c_j| 最大者消元：beta_j = (d - sum c_l beta_l)/c_j，
    转化为带偏置的无约束低维 IRLS。beta_start_full 为上一约束点的完整 beta，
    沿约束序列热启动，避免边界附近冷启动慢行。
    """
    m, p = len(x), len(x[0])
    if len(constraint) != p:
        raise ValueError('约束向量维度与设计矩阵不符')
    pivot = max(range(p), key=lambda j: abs(constraint[j]))
    if abs(constraint[pivot]) < 1e-15:
        raise ValueError('约束为零向量')
    cp = constraint[pivot]
    keep = [j for j in range(p) if j != pivot]
    if offset is None:
        offset = [0.0] * m
    new_offset = [offset[i] + d / cp * x[i][pivot] for i in range(m)]
    xr = [[x[i][j] - constraint[j] / cp * x[i][pivot] for j in keep]
          for i in range(m)]
    if not keep:
        mu = [_safe_exp(o) for o in new_offset]
        return {'converged': True, 'beta_reduced': [], 'mu': mu,
                'loglik': poisson_loglik(y, mu), 'iterations': 0,
                'grad_inf': 0.0, 'pivot': pivot, 'keep': keep,
                'xr': xr, 'offset': new_offset}
    warm = None
    if beta_start_full is not None:
        # 热启动：保持非主元列系数；主元列在偏置中消去，低维系数直接继承
        warm = [beta_start_full[j] for j in keep]
    res = _irls(xr, y, offset=new_offset, tol=tol,
                beta_start=warm, max_iter=max_iter)
    if not res['converged']:
        # IRLS 在紧约束/病态下慢收敛：BFGS 从同一点接管，失败再冷启动一次
        bf = _bfgs(xr, y, offset=new_offset, tol=tol,
                   beta_start=res['beta'], max_iter=max_iter * 4)
        if not bf['converged'] and warm is not None:
            bf = _bfgs(xr, y, offset=new_offset, tol=tol,
                       beta_start=None, max_iter=max_iter * 4)
        if bf['converged']:
            res = bf
    res['beta_reduced'] = res.pop('beta')
    res.update({'pivot': pivot, 'keep': keep, 'xr': xr, 'offset': new_offset})
    return res


def expand_beta(constrained, constraint, d):
    """把消元后的低维系数还原为完整 beta。"""
    p = len(constraint)
    pivot, keep = constrained['pivot'], constrained['keep']
    cp = constraint[pivot]
    beta = [0.0] * p
    for j, g in zip(keep, constrained['beta_reduced']):
        beta[j] = g
    beta[pivot] = (d - sum(constraint[j] * beta[j] for j in keep)) / cp
    return beta
