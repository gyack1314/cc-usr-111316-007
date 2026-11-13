"""命令行入口：analyze 输出 JSON 或 Markdown；history 仅输出可回溯历史表。"""
import argparse
import json
import math
import sys

from .pipeline import analyze
from .report import render_markdown


def _json_default(o):
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f'不可序列化对象 {type(o)}')


def _sanitize_inf(obj):
    """把 +-inf 转为 None；配套的 *_infinite 布尔标志保留语义，保证严格合法 JSON。"""
    if isinstance(obj, float) and math.isinf(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_inf(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_inf(v) for v in obj]
    return obj


def main(argv=None):
    p = argparse.ArgumentParser(prog='capture_analysis',
                                description='多名单捕获对数线性遗漏人数分析')
    sub = p.add_subparsers(dest='command', required=True)

    pa = sub.add_parser('analyze', help='完整拟合并输出报告')
    pa.add_argument('file')
    pa.add_argument('--format', choices=['json', 'markdown'], default='markdown')
    pa.add_argument('--output', '-o', help='输出文件，缺省打印到标准输出')
    pa.add_argument('--perturbations', default='1,2,5,10',
                    help='未观测格注入计数（逗号分隔整数）')

    ph = sub.add_parser('history', help='只输出捕获历史表与对账，不拟合')
    ph.add_argument('file')
    ph.add_argument('--output', '-o', help='输出文件，缺省打印到标准输出')
    args = p.parse_args(argv)

    if args.command == 'history':
        result = analyze(args.file)
        payload = {'manifest_sha256': result['manifest_sha256'],
                   'history': result['history']}
        text = json.dumps(_sanitize_inf(payload), ensure_ascii=False, indent=2,
                          default=_json_default, allow_nan=False)
    else:
        perturbations = tuple(int(x) for x in args.perturbations.split(',') if x)
        result = analyze(args.file, perturbations=perturbations)
        if args.format == 'json':
            text = json.dumps(_sanitize_inf(result), ensure_ascii=False, indent=2,
                              default=_json_default, allow_nan=False)
        else:
            text = render_markdown(result)
    if args.output:
        with open(args.output, 'w', encoding='utf8') as f:
            f.write(text)
    else:
        sys.stdout.write(text + ('\n' if not text.endswith('\n') else ''))


if __name__ == '__main__':
    main()
