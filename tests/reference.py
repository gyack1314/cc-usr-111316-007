"""高精度小表参照：枚举/闭式复核拟合、得分与轮廓值。

仅用于测试，规模限定在 k=3（8 格）以内，用 Fraction 与稠密枚举做独立核算，
不依赖被测包的 IRLS/LP 路径。
"""
import math
from fractions import Fraction
from itertools import product


def independence_mle(counts, masks):
    """全独立模型（主效应）闭式 MLE。

    对数线性 eta = b0 + sum b_j x_j，Poisson MLE 满足拟合边际=观测边际。
    用 IRLS 的独立参照实现（与生产代码不同的朴素全矩阵牛顿，固定大量迭代）。
    """
    p = 1 + len(masks[0])
    x = [[1.0] + [float(int(b)) for b in m] for m in masks]
    beta = [0.0] * p
    y = [float(c) for c in counts]
    for _ in range(300):
        mu = [math.exp(sum(x[i][j] * beta[j] for j in range(p))) for i in range(len(x))]
        info = [[sum(x[i][a] * mu[i] * x[i][b] for i in range(len(x)))
                 for b in range(p)] for a in range(p)]
        rhs = [sum(x[i][j] * (y[i] - mu[i]) for i in range(len(x))) for j in range(p)]
        # 高斯消元解 info delta = rhs
        n = p
        m = [info[i][:] + [rhs[i]] for i in range(n)]
        for c in range(n):
            piv = max(range(c, n), key=lambda r: abs(m[r][c]))
            m[c], m[piv] = m[piv], m[c]
            for r in range(n):
                if r != c and m[r][c] != 0:
                    f = m[r][c] / m[c][c]
                    m[r] = [m[r][k] - f * m[c][k] for k in range(n + 1)]
        delta = [m[i][n] / m[i][i] for i in range(n)]
        # 朴素阻尼
        step = 1.0
        for _ in range(60):
            b1 = [beta[j] + step * delta[j] for j in range(p)]
            mu1 = [math.exp(sum(x[i][j] * b1[j] for j in range(p))) for i in range(len(x))]
            ll1 = sum(y[i] * math.log(mu1[i]) - mu1[i] for i in range(len(x)))
            ll0 = sum(y[i] * math.log(mu[i]) - mu[i] for i in range(len(x)))
            if ll1 >= ll0 - 1e-12:
                break
            step *= 0.5
        beta = [beta[j] + step * delta[j] for j in range(p)]
    mu = [math.exp(sum(x[i][j] * beta[j] for j in range(p))) for i in range(len(x))]
    ll = sum(y[i] * math.log(mu[i]) - mu[i] - math.lgamma(y[i] + 1)
             for i in range(len(x)))
    return beta, mu, ll


def saturated_fitted(counts, structural_zero_masks=()):
    """饱和模型：观测格 mu=count；零计数格 mu=0（边界）。"""
    return list(counts)


def factorial_ll(y, mu):
    """含阶乘常数的 Poisson 对数似然。"""
    return sum(yi * math.log(mui) - mui - math.lgamma(yi + 1)
               for yi, mui in zip(y, mu) if mui > 0 or yi > 0)


def aicc_reference(loglik, k, n):
    if n <= k + 1:
        return None
    return -2 * loglik + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def grid_profile_zero(x, y, c, log_v_grid, tol=1e-10):
    """在给定 log(lambda0) 网格上用独立牛顿法求轮廓对数似然。"""
    from capture_analysis.fit import constrained_fit, expand_beta
    out = []
    for lv in log_v_grid:
        res = constrained_fit(x, y, c, float(lv), tol=tol)
        b = expand_beta(res, c, float(lv))
        out.append((float(lv), res['loglik'], b))
    return out
