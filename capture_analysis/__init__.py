"""多名单捕获（捕获-再捕获）对数线性分析。

模块划分：
- rational   清单独占有理数字符串解析
- linalg     纯标准库小矩阵运算（LU/Cholesky/秩/特征值/单纯形）
- histories  家庭行合并、分层冲突、捕获历史表与结构零
- design     层级原则模型项、设计矩阵、设计秩
- fit        Poisson 对数线性 IRLS、有限 MLE 检验、信息矩阵
- profile    全零均值 lambda0 的轮廓似然区间
- models     AICc、模型平均、逐模型编排与拒绝原因
- sensitivity 未观测格扰动与最高阶交互固定两条预设路径
- report     可回溯的 JSON/Markdown 报告
- cli        命令行入口
"""

__all__ = [
    'rational', 'linalg', 'histories', 'design', 'fit',
    'profile', 'models', 'sensitivity', 'report', 'cli',
]
