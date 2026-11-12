"""仅核对输入形状，不生成分析结果。"""
import json
from .io import asset
def summarize(v,root):
    k=len(v['channels'])
    if not 3<=k<=6 or len(set(v['channels']))!=k:raise ValueError('渠道数或身份错误')
    rows=[json.loads(x) for x in asset(root,v['families']).splitlines()]
    for r in rows:
        if len(r['mask'])!=k or set(r['mask'])-{'0','1'}:raise ValueError('位图错误')
    return {'raw_family_rows':len(rows),'channels':k,'candidate_models':len(v['models'])}
