"""小规模纯标准库线性代数。

设计矩阵至多 (2^6-1) 行、64 列，所有运算都在小矩阵规模内，
因此不依赖 numpy：高斯消元、Cholesky、数值/精确秩、对称特征值、
两阶段有理单纯形（可行性 LP，用于有限 MLE 存在性检验）。
"""
from fractions import Fraction

# ---------------------------------------------------------------- 基础运算

def matmul(a, b):
    m, k, n = len(a), len(a[0]), len(b[0])
    if len(b) != k:
        raise ValueError('矩阵维度不匹配')
    return [[sum(a[i][t] * b[t][j] for t in range(k)) for j in range(n)]
            for i in range(m)]


def transpose(a):
    return [list(row) for row in zip(*a)]


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


# ---------------------------------------------------------------- LU 求解

def lu_solve(a, b):
    """带部分主元的高斯消元解 A x = b，返回 x（浮点）。"""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-14:
            raise ZeroDivisionError('矩阵奇异')
        if piv != col:
            m[col], m[piv] = m[piv], m[col]
        pivot = m[col][col]
        for r in range(col + 1, n):
            factor = m[r][col] / pivot
            if factor:
                m[r][col] = 0.0
                for c in range(col + 1, n + 1):
                    m[r][c] -= factor * m[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / m[i][i]
    return x


def inverse(a):
    n = len(a)
    cols = [lu_solve(a, [1.0 if i == j else 0.0 for i in range(n)])
            for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


# ---------------------------------------------------------------- Cholesky

def cholesky(a):
    """对称正定矩阵的 Cholesky 下三角 L，非正定抛 ValueError。"""
    n = len(a)
    l = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(l[i][k] * l[j][k] for k in range(j))
            if i == j:
                d = a[i][i] - s
                if d <= 1e-12:
                    raise ValueError('矩阵非正定')
                l[i][j] = d ** 0.5
            else:
                l[i][j] = (a[i][j] - s) / l[j][j]
    return l


def chol_solve(l, b):
    """给定 Cholesky L（A = L L^T）解 A x = b。"""
    n = len(l)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(l[i][k] * y[k] for k in range(i))) / l[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(l[k][i] * x[k] for k in range(i + 1, n))) / l[i][i]
    return x


# ---------------------------------------------------------------- 秩

def exact_rank(rows):
    """0/1（或任意有理数）矩阵的精确列秩，Fraction 高斯消元。"""
    if not rows:
        return 0
    m = [[Fraction(v) for v in row] for row in rows]
    n_rows, n_cols = len(m), len(m[0])
    r = 0
    for col in range(n_cols):
        piv = next((i for i in range(r, n_rows) if m[i][col] != 0), None)
        if piv is None:
            continue
        m[r], m[piv] = m[piv], m[r]
        pv = m[r][col]
        m[r] = [v / pv for v in m[r]]
        for i in range(n_rows):
            if i != r and m[i][col] != 0:
                f = m[i][col]
                m[i] = [a - f * b for a, b in zip(m[i], m[r])]
        r += 1
        if r == n_rows:
            break
    return r


def numeric_rank(rows, tol=1e-10):
    """Householder QR 带列主元的数值秩。"""
    m = [list(map(float, row)) for row in rows]
    if not m:
        return 0
    n_rows, n_cols = len(m), len(m[0])
    diag = []
    r = 0
    for k in range(min(n_rows, n_cols)):
        # 选剩余列中范数最大的列做主元列
        norms = [(sum(m[i][j] ** 2 for i in range(k, n_rows)), j)
                 for j in range(k, n_cols)]
        _, q = max(norms)
        if q != k:
            for i in range(n_rows):
                m[i][k], m[i][q] = m[i][q], m[i][k]
        x = [m[i][k] for i in range(k, n_rows)]
        nx = sum(v * v for v in x) ** 0.5
        if nx < tol:
            diag.append(0.0)
            continue
        alpha = -nx if x[0] >= 0 else nx
        v = x[:]
        v[0] -= alpha
        nv = sum(vv * vv for vv in v) ** 0.5
        if nv < 1e-15:
            diag.append(nx)
            r += 1
            continue
        v = [vv / nv for vv in v]
        for j in range(k, n_cols):
            dot = sum(v[t] * m[k + t][j] for t in range(len(v)))
            for t in range(len(v)):
                m[k + t][j] -= 2 * dot * v[t]
        diag.append(abs(m[k][k]))
        r += 1
    scale = max(diag, default=0.0)
    return sum(1 for d in diag if d > tol * max(1.0, scale))


# ---------------------------------------------------------------- 对称特征值

def jacobi_eigen(a, tol=1e-13, max_sweeps=100):
    """循环 Jacobi 求对称矩阵特征值，返回升序特征值列表。"""
    n = len(a)
    a = [row[:] for row in a]
    for _ in range(max_sweeps):
        off = sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i < j)
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-15:
                    continue
                # 数值稳定的旋转角：t = sign(tau)/(|tau|+sqrt(1+tau^2))，tau=0 时 t=1
                tau = (a[q][q] - a[p][p]) / (2 * a[p][q])
                if tau == 0:
                    t = 1.0
                else:
                    t = (1.0 if tau > 0 else -1.0) / (abs(tau) + (1 + tau * tau) ** 0.5)
                c = 1 / (1 + t * t) ** 0.5
                s = t * c
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
    return sorted(a[i][i] for i in range(n))


# ---------------------------------------------------------------- 有理单纯形

def lp_feasible_ge(matrix, rhs=Fraction(1)):
    """可行性 LP：是否存在自由向量 beta 使 A beta >= rhs（逐分量）。

    标准两阶段单纯形（全部 Fraction 精确运算，Bland 反循环规则）：
      自由变量 beta_j = u_j - v_j（u,v>=0），剩余变量 s_i>=0：
          A u - A v - s = rhs
      第一阶段对每行加人工变量 e_i>=0，最小化 sum(e)。
      sum(e) 的最小值为 0 <=> 原系统可行。
    返回 (可行?, beta 见证浮点列表或 None)。
    """
    a = [[Fraction(x) for x in row] for row in matrix]
    m = len(a)
    p = len(a[0]) if m else 0
    if m == 0:
        return True, [0.0] * p
    rhs = Fraction(rhs)

    # 变量列：u(p) | v(p) | s(m) | e(m)，最后一列为 RHS
    n = 2 * p + m
    art0 = n
    ncols = n + m

    def coeff(i, j):
        """第 i 个约束在变量列 j 上的系数。"""
        if j < p:
            return a[i][j]
        if j < 2 * p:
            return -a[i][j - p]
        if j < 2 * p + m:
            return Fraction(-1) if j == 2 * p + i else Fraction(0)
        return Fraction(1) if j == art0 + i else Fraction(0)

    # 表体 T：m 个约束行（下标 0..m-1）+ 1 个目标行（下标 m）
    T = [[Fraction(0) for _ in range(ncols + 1)] for _ in range(m + 1)]
    for i in range(m):
        for j in range(ncols):
            T[i][j] = coeff(i, j)
        T[i][ncols] = rhs
    # 第一阶段目标 w = min sum(e)，表中用最小化约定：检验数 c_j - c_B B^-1 A_j
    for i in range(m):
        T[m][art0 + i] = Fraction(1)
    # 消去基变量（人工列）在目标行的分量
    for i in range(m):
        f = T[m][art0 + i]
        if f != 0:
            T[m] = [x - f * y for x, y in zip(T[m], T[i])]
    basic = [art0 + i for i in range(m)]  # basic[i] = 第 i 行的基变量列

    def pivot(r, q):
        pv = T[r][q]
        T[r] = [x / pv for x in T[r]]
        for i in range(m + 1):
            if i != r and T[i][q] != 0:
                f = T[i][q]
                T[i] = [x - f * y for x, y in zip(T[i], T[r])]
        basic[r] = q

    # Bland：最小化表中入基选检验数 < 0 的最小列；出基选最小比，并列取最小基下标
    cap = 4 * (ncols + 1) * (m + 1) + 100
    for _ in range(cap):
        q = next((j for j in range(ncols) if j not in basic and T[m][j] < 0), None)
        if q is None:
            break
        ratios = [(T[i][ncols] / T[i][q], basic[i], i)
                  for i in range(m) if T[i][q] > 0]
        if not ratios:
            raise RuntimeError('第一阶段无界，不应发生（人工变量有界）')
        _, _, r = min(ratios, key=lambda t: (t[0], t[1]))
        pivot(r, q)
    else:
        raise RuntimeError('单纯形迭代超上限，疑似循环')

    # 最小化目标值在 T[m][RHS]；为 0 才可行
    if T[m][ncols] != 0:
        return False, None

    # 清理基中残留的零值人工变量
    for i in range(m):
        if basic[i] >= art0:
            q = next((j for j in range(art0) if T[i][j] != 0), None)
            if q is not None:
                pivot(i, q)
            elif T[i][ncols] != 0:
                return False, None  # 0 = 非零，矛盾行

    vals = [Fraction(0)] * ncols
    for i in range(m):
        if basic[i] < art0:
            vals[basic[i]] = T[i][ncols]
    beta = [float(vals[j] - vals[p + j]) for j in range(p)]
    # 显式复核见证，不满足即视为实现错误而非悄悄返回
    for i, row in enumerate(a):
        got = sum(float(row[j]) * beta[j] for j in range(p))
        if got < float(rhs) - 1e-9:
            raise RuntimeError(f'LP 见证复核失败：行 {i} 得 {got} < {float(rhs)}')
    return True, beta
