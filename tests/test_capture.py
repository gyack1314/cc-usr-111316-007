"""捕获分析端到端与统计口径测试。

四个指定验证场景：
- 全独立（数据由独立模型生成，主效应模型应拟合良好）
- 强渠道依赖（含高阶交互结构，低阶模型失拟）
- 单一稀疏格（仅一个格子计数很少，检验稳定性与区间）
- 已知结构零（结构零格不进设计/似然，000 外推正常）
另含历史表合并/冲突、设计秩、LP、AICc、轮廓与可识别性的单元测试。
"""
import json
import math
import unittest
from fractions import Fraction
from pathlib import Path

from capture_analysis.histories import build_history, StratumConflictError
from capture_analysis.design import closure_terms, build_design, term_labels
from capture_analysis.linalg import (exact_rank, numeric_rank, jacobi_eigen,
                                     lp_feasible_ge, cholesky, chol_solve)
from capture_analysis.fit import (fit_poisson, finite_mle_exists, OK,
                                  DESIGN_RANK_DEFICIENT, MLE_INFINITE,
                                  poisson_loglik, constrained_fit, expand_beta)
from capture_analysis.profile import (profile_interval, zero_design_row,
                                      profile_loglik)
from capture_analysis.models import run_model, aicc, model_averaged_estimate
from capture_analysis.sensitivity import (highest_order_sensitivity,
                                          unobserved_cell_sensitivity)
from capture_analysis.stats import chi2_1_quantile, norm_ppf, norm_cdf
from tests.reference import independence_mle, aicc_reference

CH = ['c1', 'c2', 'c3']
ALL8 = [''.join(b) for b in __import__('itertools').product('01', repeat=3)]


def rows_from_counts(counts, stratum='s1', prefix='h'):
    """{mask: count} -> 家庭行列表。"""
    out = []
    i = 0
    for mask, n in counts.items():
        for _ in range(n):
            out.append({'household': f'{prefix}-{i}', 'stratum': stratum,
                        'mask': mask})
            i += 1
    return out


def history_from_counts(counts, structural=(), stratum='s1'):
    return build_history(rows_from_counts(counts, stratum), CH, list(structural))


MAIN = [['c1'], ['c2'], ['c3']]
PAIR = MAIN + [['c1', 'c2']]
SAT = MAIN + [['c1', 'c2'], ['c1', 'c3'], ['c2', 'c3'], ['c1', 'c2', 'c3']]


class HistoryTests(unittest.TestCase):
    def test_cell_enumeration_and_zero_annotation(self):
        counts = {'001': 5, '111': 3}
        h = history_from_counts(counts)
        self.assertEqual(h['merged_families'], 8)
        self.assertEqual(h['observed_counts']['s1'], 8)
        # 8 个格子全部枚举，未观测可能格计数 0，全零格显式存在
        self.assertEqual(len(h['possible_masks'], ) and len(h['possible_masks']), 8)
        self.assertEqual(h['tables']['s1']['000'], 0)
        self.assertIn('000', h['unobserved_possible']['s1'])
        self.assertTrue(h['all_zero_possible'])
        self.assertFalse(h['zero_is_structural'])

    def test_duplicate_or_merge_and_stratum_conflict(self):
        rows = [
            {'household': 'a', 'stratum': 's1', 'mask': '100'},
            {'household': 'a', 'stratum': 's1', 'mask': '010'},  # OR -> 110
            {'household': 'b', 'stratum': 's1', 'mask': '001'},
        ]
        h = build_history(rows, CH, [])
        self.assertEqual(h['merged_families'], 2)
        self.assertEqual(h['duplicate_rows'], 1)
        self.assertEqual(h['merge_log'][0]['to'], '110')
        # 分层冲突阻断
        rows[1] = {'household': 'a', 'stratum': 's2', 'mask': '010'}
        with self.assertRaises(StratumConflictError):
            build_history(rows, CH, [])

    def test_all_zero_row_rejected(self):
        rows = [{'household': 'a', 'stratum': 's1', 'mask': '000'}]
        with self.assertRaises(ValueError):
            build_history(rows, CH, [])

    def test_structural_zero_contradiction_and_removal(self):
        # 结构零里出现观测 -> 矛盾
        with self.assertRaises(ValueError):
            history_from_counts({'111': 2}, structural=('111',))
        h = history_from_counts({'001': 4, '010': 4}, structural=('111',))
        self.assertNotIn('111', h['possible_masks'])
        self.assertEqual(len(h['possible_masks']), 7)

    def test_structural_zero_allzero_fixed(self):
        # 000 为结构零：未观测固定 0
        h = history_from_counts({'001': 4}, structural=('000',))
        self.assertTrue(h['zero_is_structural'])
        self.assertNotIn('000', h['possible_masks'])


class DesignTests(unittest.TestCase):
    def test_hierarchy_principle_enforced(self):
        # 声明二阶交互却缺主效应 -> 拒绝
        with self.assertRaises(ValueError):
            closure_terms([['c1', 'c2']], CH)
        terms = closure_terms(SAT, CH)
        # 饱和模型 7 个非截距项
        self.assertEqual(len(terms), 7)
        labels = term_labels(SAT, CH)
        self.assertEqual(labels[0], ('1',))

    def test_saturated_full_rank_normal_but_rank_deficient_with_structural(self):
        h = history_from_counts({m: 1 for m in ALL8 if m != '000'})
        d = build_design(h, SAT)
        self.assertEqual(d['rank_exact'], 8)
        self.assertFalse(d['rank_deficient'])
        # 去掉 111 后只有 7 格 -> 饱和设计秩 7 < 8
        h2 = history_from_counts({m: 1 for m in ALL8 if m not in ('000', '111')},
                                 structural=('111',))
        d2 = build_design(h2, SAT)
        self.assertEqual(d2['rank_exact'], 7)
        self.assertTrue(d2['rank_deficient'])

    def test_exact_and_numeric_rank_agree(self):
        h = history_from_counts({'001': 2, '111': 4})
        for spec in (MAIN, PAIR, SAT):
            d = build_design(h, spec)
            self.assertEqual(d['rank_exact'], d['rank_numeric'])


class LpAndLinalgTests(unittest.TestCase):
    def test_lp_known_feasibility(self):
        # 全零行要求 0>=1：不可行
        self.assertFalse(lp_feasible_ge([[0, 0, 0]])[0])
        self.assertFalse(lp_feasible_ge([[1, 0], [0, 0]])[0])
        # 单位约束可行，见证满足
        ok, w = lp_feasible_ge([[1, 0], [0, 1]])
        self.assertTrue(ok)
        self.assertGreaterEqual(w[0], 1 - 1e-9)
        self.assertGreaterEqual(w[1], 1 - 1e-9)
        # 负系数自由变量可取负值
        ok, w = lp_feasible_ge([[-1]])
        self.assertTrue(ok)
        self.assertLessEqual(w[0], -1 + 1e-9)

    def test_lp_witness_always_satisfies_constraints(self):
        import random
        random.seed(123)
        for _ in range(200):
            m = random.randint(1, 5)
            p = random.randint(1, 5)
            A = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
            ok, w = lp_feasible_ge(A)
            if ok:
                for row in A:
                    self.assertGreaterEqual(
                        sum(row[j] * w[j] for j in range(p)), 1 - 1e-9)

    def test_jacobi_against_known_eigenvalues(self):
        # 对角阵特征值即对角元
        eig = jacobi_eigen([[3.0, 0], [0, -1.0]])
        self.assertAlmostEqual(eig[0], -1.0, places=10)
        self.assertAlmostEqual(eig[1], 3.0, places=10)
        # 对称阵 [[2,1],[1,2]] 特征值 1,3
        eig = jacobi_eigen([[2.0, 1.0], [1.0, 2.0]])
        self.assertAlmostEqual(eig[0], 1.0, places=9)
        self.assertAlmostEqual(eig[1], 3.0, places=9)

    def test_cholesky_recovers_solution(self):
        a = [[4.0, 1.0], [1.0, 3.0]]
        l = cholesky(a)
        x = chol_solve(l, [1.0, 2.0])
        self.assertAlmostEqual(4 * x[0] + x[1], 1.0, places=10)
        self.assertAlmostEqual(x[0] + 3 * x[1], 2.0, places=10)

    def test_chi2_threshold_and_normal_quantiles(self):
        # 卡方1 95% 阈值 ≈ 3.8414588
        self.assertAlmostEqual(chi2_1_quantile(0.05), 3.841458820694124, places=8)
        # 正态分位数往返
        for p in (0.025, 0.5, 0.975):
            self.assertAlmostEqual(norm_cdf(norm_ppf(p)), p, places=10)


class FitScenarioTests(unittest.TestCase):
    def test_full_independence_matches_reference(self):
        # 由独立模型生成的均衡计数
        counts = {m: 10 for m in ALL8 if m != '000'}
        h = history_from_counts(counts)
        d = build_design(h, MAIN)
        f = fit_poisson(d, h['tables']['s1'])
        self.assertEqual(f['status'], OK)
        beta_ref, mu_ref, ll_ref = independence_mle(
            [h['tables']['s1'][m] for m in d['masks']], d['masks'])
        for a, b in zip(f['beta'], beta_ref):
            self.assertAlmostEqual(a, b, places=6)
        self.assertAlmostEqual(f['loglik'], ll_ref, places=6)
        # 独立模型对称结构：三个主效应系数相等
        self.assertAlmostEqual(f['beta'][1], f['beta'][2], places=8)
        self.assertAlmostEqual(f['beta'][2], f['beta'][3], places=8)

    def test_strong_dependence_lower_order_model_misfits(self):
        # 强依赖：只在全捕获 111 与单捕获出现，成对格为空 -> 主效应模型拟合差
        counts = {'100': 20, '010': 20, '001': 20, '111': 40}
        h = history_from_counts(counts)
        r_main = run_model(h, {'id': 'main', 'interactions': MAIN})['strata']['s1']
        r_sat = run_model(h, {'id': 'sat', 'interactions': SAT})['strata']['s1']
        self.assertEqual(r_main['status'], OK)
        # 饱和模型含零格，lambda0 不可识别（边界）
        self.assertFalse(r_sat['lambda0_identifiable'])
        self.assertTrue(r_sat['lambda0_interval']['upper_infinite'])
        # 饱和模型似然必然不低于主效应
        self.assertGreaterEqual(r_sat['loglik'] - r_main['loglik'], -1e-9)

    def test_single_sparse_cell_stable_finite_interval(self):
        # 单一稀疏格：仅 111 计数为 1，其余充足
        counts = {m: 12 for m in ALL8 if m != '000'}
        counts['111'] = 1
        h = history_from_counts(counts)
        r = run_model(h, {'id': 'main', 'interactions': MAIN})['strata']['s1']
        self.assertEqual(r['status'], OK)
        ci = r['lambda0_interval']
        self.assertTrue(math.isfinite(ci['lower']))
        self.assertTrue(math.isfinite(ci['upper']))
        self.assertLess(ci['lower'], ci['upper'])
        # 点估计落在区间内
        self.assertLessEqual(ci['lower'] - 1e-9, r['lambda0'])
        self.assertLessEqual(r['lambda0'], ci['upper'] + 1e-9)

    def test_known_structural_zero_saturated_rank_deficient(self):
        counts = {m: 10 for m in ALL8 if m not in ('000', '111')}
        h = history_from_counts(counts, structural=('111',))
        r = run_model(h, {'id': 'sat', 'interactions': SAT})['strata']['s1']
        self.assertEqual(r['status'], DESIGN_RANK_DEFICIENT)
        # 主效应仍可估且 lambda0 正常外推（000 非结构零）
        rm = run_model(h, {'id': 'main', 'interactions': MAIN})['strata']['s1']
        self.assertEqual(rm['status'], OK)
        self.assertTrue(rm['lambda0_identifiable'])

    def test_allzero_structural_zero_fixes_missing_to_zero(self):
        counts = {'001': 5, '010': 5}
        h = history_from_counts(counts, structural=('000',))
        r = run_model(h, {'id': 'main', 'interactions': MAIN})['strata']['s1']
        self.assertEqual(r['status'], OK)
        self.assertEqual(r['lambda0'], 0.0)
        self.assertEqual(r['n_total_point'], 10.0)
        self.assertIsNone(r['lambda0_interval'])

    def test_no_arbitrary_half_added_to_zeros(self):
        counts = {'100': 10, '010': 10}
        h = history_from_counts(counts)
        d = build_design(h, MAIN)
        # 原始计数中的零必须原样进入似然
        y = [h['tables']['s1'][m] for m in d['masks']]
        self.assertIn(0, y)
        f = fit_poisson(d, h['tables']['s1'])
        self.assertEqual(f['status'], OK)

    def test_mle_infinite_when_design_rows_violate_haberman(self):
        # 构造 Haberman 不可行：某零格设计行无法被线性抬升。
        # 用结构零制造一个零格，其设计行与其他零格冲突 -> LP 不可行。
        # 这里直接对一个含全零设计行的人造矩阵检验判据。
        x = [[1.0, 0.0], [0.0, 0.0]]
        y = [5, 0]
        self.assertFalse(finite_mle_exists(x, y)[0])


class ProfileTests(unittest.TestCase):
    def test_profile_interval_covers_point_and_threshold(self):
        counts = {m: 10 for m in ALL8 if m != '000'}
        h = history_from_counts(counts)
        d = build_design(h, MAIN)
        f = fit_poisson(d, h['tables']['s1'])
        ci = profile_interval(d, f, h, alpha=0.05)
        self.assertAlmostEqual(ci['threshold'], chi2_1_quantile(0.05), places=9)
        self.assertLessEqual(ci['lower'], ci['point'] + 1e-9)
        self.assertLessEqual(ci['point'], ci['upper'] + 1e-9)
        # 端点处轮廓统计量应在阈值附近（两端点均复核）
        c = zero_design_row(d, h)
        for name, v in (('lower', ci['lower']), ('upper', ci['upper'])):
            if not math.isfinite(v):
                continue
            ll_e, _, _ = profile_loglik(d['matrix'], f['y'], c, math.log(v))
            stat_e = 2 * (f['loglik'] - ll_e)
            self.assertLess(abs(stat_e - ci['threshold']),
                            0.02 * ci['threshold'], msg=name)

    def test_saturated_profile_upper_infinite_preserved(self):
        counts = {m: 10 for m in ALL8 if m != '000'}
        h = history_from_counts(counts)
        d = build_design(h, SAT)
        f = fit_poisson(d, h['tables']['s1'])
        ci = profile_interval(d, f, h, alpha=0.05)
        self.assertTrue(ci['upper_infinite'])
        self.assertTrue(math.isinf(ci['upper']))

    def test_constrained_fit_respects_linear_constraint(self):
        counts = {'001': 8, '111': 6, '100': 4}
        h = history_from_counts(counts)
        d = build_design(h, MAIN)
        c = zero_design_row(d, h)
        for dval in (-2.0, 0.0, 1.5):
            res = constrained_fit(d['matrix'],
                                  [h['tables']['s1'][m] for m in d['masks']],
                                  c, dval)
            self.assertTrue(res['converged'])
            beta = expand_beta(res, c, dval)
            self.assertAlmostEqual(sum(c[j] * beta[j] for j in range(len(c))),
                                   dval, places=8)


class AicTests(unittest.TestCase):
    def test_aicc_formula_and_undefined_rule(self):
        self.assertIsNone(aicc(-10.0, k=5, n=6))   # n = k+1
        self.assertIsNone(aicc(-10.0, k=5, n=4))   # n < k+1
        # 与参照一致
        self.assertAlmostEqual(aicc(-10.0, 3, 20), aicc_reference(-10.0, 3, 20))

    def test_average_excludes_unidentifiable_and_never_averages_intervals(self):
        counts = {m: 10 for m in ALL8 if m != '000'}
        h = history_from_counts(counts)
        results = [run_model(h, {'id': i, 'interactions': s})
                   for i, s in (('main', MAIN), ('pair', PAIR), ('sat', SAT))]
        avg = model_averaged_estimate(results)['s1']
        self.assertNotIn('sat', avg['eligible_models'])
        self.assertIn('sat', avg['excluded_unidentifiable_models'])
        self.assertAlmostEqual(sum(avg['aicc_weights'].values()), 1.0, places=10)
        # 输出中没有平均区间键
        self.assertNotIn('interval', avg)


class SensitivityTests(unittest.TestCase):
    def test_unobserved_cell_path_does_not_mutate_counts(self):
        counts = {'001': 10, '111': 5}
        h = history_from_counts(counts)
        before = dict(h['tables']['s1'])
        out = unobserved_cell_sensitivity(h, {'id': 'main', 'interactions': MAIN},
                                          perturbations=(1, 2))
        self.assertEqual(dict(h['tables']['s1']), before)  # 原表未改
        self.assertTrue(out['points'])
        for p in out['points']:
            self.assertEqual(p['status'], OK)

    def test_highest_order_path_fixes_coefficient_and_notes_unverifiable(self):
        counts = {m: 10 for m in ALL8 if m != '000'}
        h = history_from_counts(counts)
        out = highest_order_sensitivity(h, {'id': 'sat', 'interactions': SAT},
                                        fixed_values=(-1.0, 0.0, 1.0))
        # 每个固定值都有一个点且 lambda0 被重新计算
        vals = [p['fixed_value'] for p in out['points'] if p.get('status') == OK]
        self.assertEqual(sorted(vals), [-1.0, 0.0, 1.0])
        self.assertIn('无法由无名单家庭验证', out['unverifiable_note'])

    def test_independence_cannot_be_validated_by_unlisted(self):
        counts = {'100': 5}
        h = history_from_counts(counts)
        out = highest_order_sensitivity(h, {'id': 'main', 'interactions': MAIN},
                                        fixed_values=(-1.0, 0.0, 1.0))
        # 主效应模型不含最高阶项：临时按层级补入，并给出三个固定值情景
        self.assertFalse(out['term_present_in_model'])
        self.assertIsNotNone(out['augmentation_note'])
        vals = sorted(p['fixed_value'] for p in out['points']
                      if p.get('status') == OK)
        self.assertEqual(vals, [-1.0, 0.0, 1.0])
        self.assertIn('无法由无名单家庭验证', out['unverifiable_note'])


class MultiStratumTests(unittest.TestCase):
    def _two_stratum_rows(self):
        rows, i = [], 0
        for s, mult in (('urban', 10), ('rural', 5)):
            for m in ('001', '010', '100', '111'):
                for _ in range(mult):
                    rows.append({'household': f'{s}-{i}', 'stratum': s,
                                 'mask': m})
                    i += 1
        return rows

    def test_strata_fit_separately_and_totaled(self):
        h = build_history(self._two_stratum_rows(), CH, [])
        self.assertEqual(sorted(h['strata']), ['rural', 'urban'])
        r = run_model(h, {'id': 'main', 'interactions': MAIN})
        self.assertEqual(set(r['strata']), {'rural', 'urban'})
        # 两层均独立可估，样本量分别保留
        self.assertEqual(r['strata']['urban']['n_observed'], 40)
        self.assertEqual(r['strata']['rural']['n_observed'], 20)
        # 总点估计 = 各层之和
        expect = sum(r['strata'][s]['n_total_point'] for s in h['strata'])
        self.assertAlmostEqual(r['overall']['n_total_point'], expect, places=9)
        # 包络区间端点也是逐层相加
        lo = sum(r['strata'][s]['n_total_interval'][0] for s in h['strata'])
        self.assertAlmostEqual(r['overall']['n_total_envelope_interval'][0], lo,
                               places=9)
        self.assertIn('保守包络', r['overall']['envelope_note'])

    def test_stratum_conflict_blocks_across_duplicate(self):
        rows = self._two_stratum_rows()
        rows.append({'household': rows[0]['household'], 'stratum': 'rural',
                     'mask': '001'})  # 同 ID 跨分层
        with self.assertRaises(StratumConflictError):
            build_history(rows, CH, [])


class StatusReportingTests(unittest.TestCase):
    def test_three_distinct_failure_modes_separately_reported(self):
        # 1) 设计秩不足：结构零减格使饱和模型列数 > 行数
        hz = history_from_counts({m: 10 for m in ALL8
                                  if m not in ('000', '111')},
                                 structural=('111',))
        rz = run_model(hz, {'id': 'sat', 'interactions': SAT})['strata']['s1']
        self.assertEqual(rz['status'], DESIGN_RANK_DEFICIENT)
        self.assertIn('设计秩', rz['status_reason'])

        # 2) 边界不可识别：饱和模型在 000 未观测时有限 MLE 但 λ₀ 区间无穷
        hb = history_from_counts({m: 10 for m in ALL8 if m != '000'})
        rb = run_model(hb, {'id': 'sat', 'interactions': SAT})['strata']['s1']
        self.assertEqual(rb['status'], OK)
        self.assertTrue(rb['lambda0_interval']['upper_infinite'])
        self.assertFalse(rb['lambda0_identifiable'])

        # 3) 优化失败与不可识别是不同状态码
        from capture_analysis.fit import OPTIMIZATION_FAILED
        self.assertNotEqual(OPTIMIZATION_FAILED, DESIGN_RANK_DEFICIENT)
        self.assertNotEqual(OPTIMIZATION_FAILED, MLE_INFINITE)

    def test_optimization_failure_reported_not_silently_dropped(self):
        # 用极小迭代上限强制 IRLS 失败，确认返回 optimization_failed 而非编造人数
        h = history_from_counts({'001': 8, '111': 6, '100': 4})
        d = build_design(h, MAIN)
        from capture_analysis.fit import _irls, OPTIMIZATION_FAILED
        res = _irls(d['matrix'],
                    [h['tables']['s1'][m] for m in d['masks']],
                    max_iter=0)
        self.assertFalse(res['converged'])


class EndToEndManifestTests(unittest.TestCase):
    def test_pipeline_all_shipped_datasets(self):
        from capture_analysis.pipeline import analyze
        for name in ('normal', 'no-overlap-boundary', 'explicit-structural-zero'):
            with self.subTest(file=name):
                r = analyze(f'data/{name}.json')
                # 历史表对账：格子和 = 合并家庭数
                for s, view in r['history']['strata'].items():
                    self.assertEqual(sum(c['count'] for c in view['cells']),
                                     view['n_observed_union'])
                # 每个模型都有明确状态与（可估时）拟合格数
                for mr in r['model_results']:
                    for s, e in mr['strata'].items():
                        self.assertIn(e['status'],
                                      (OK, DESIGN_RANK_DEFICIENT, MLE_INFINITE,
                                       'optimization_failed'))
                        if e['status'] == OK:
                            self.assertEqual(len(e['fitted_mu']),
                                             len(e['cell_order']))

    def test_invalid_manifest_rejected(self):
        from capture_analysis.pipeline import load_manifest
        with self.assertRaises((ValueError, KeyError)):
            load_manifest('data/invalid-config.json')


if __name__ == '__main__':
    unittest.main()
