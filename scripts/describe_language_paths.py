"""Describe each 0.01 ancestor, including unguided completion and final readout.

Final decoded coordinates are a fresh model readout after integration; they must
not be substituted for the t=1 integrated proposal. No per-step physical energy
is imputed. Output is a human-readable, coordinate-backed language artifact.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from evomolsteer.io import read_json, digest, write_json
from evomolsteer.continuous.elite_path_credit import terminal_ancestors
from build_language_lineage_trial import trajectory, features, sentence


def describe(repo, config, output):
    root=repo/config['dataset']/'results'/config['campaign']
    cfg=read_json(root/'config.json')
    ref=json.loads(gzip.decompress((repo/config['reference']).read_bytes()))
    points,origin=np.array(ref['landmarks_A']),np.array(ref['origin_A'])
    donors=read_json(repo/config['output']/'oracle/donors.json')['donors']
    lines=['# 优质Steer候选的每0.01祖先描述',
        '主agent生成；完整0–1轨迹。盲选agent只收到另行冻结的0–0.5教师语言描述和匿名测试状态，没有收到此文中的终态质量标签。',
        '前半程描述选择窗口；后半程是该历史运行中停止重采样后的积分。t=1另列最终解码读出。当前态、预测终点、最终读出有不同语义，不能混用。高斯密度、壳层核、回转半径都是坐标描述，不是相互作用能。',
        'R00对应坐标锚点VAL A53 CG1，R05对应GLY A48 O，R15对应GLU A52 CB；锚点只是对齐受体中的几何参照。',
        '所有变化沿真实selected_indices父子索引，绝不直接比较同一槽位的两个时刻。短程对核是距离壳层密度；不能把它解释为接触数量。']
    sources=[]
    for donor in donors:
        a,com,path=trajectory(root,donor['batch'])
        T,N=a['selected_indices'].shape
        if not np.all(a['mask']) or not np.allclose(a['score_time'][:,0],np.arange(T)/T,atol=1e-6):
            raise ValueError('Complete fixed-atom uniform-clock trajectory required')
        anc=terminal_ancestors(a['selected_indices'])[:,donor['terminal_slot']]
        lines.append(f"\n## {donor['label']}：批次{donor['batch']}、终态槽位{donor['terminal_slot']}\n最终共同重评分pIC50={donor['pic50']:.6f}；MMFF松弛应变/重原子={donor['strain']:.6f} kcal/mol；有效、PB快速通过、能量收敛。\n最终SMILES：`{donor['smiles']}`。这些标签只用于主agent核验。")
        previous_f=None
        for step in range(T+1):
            slot=int(anc[step])
            if step:
                parent=int(anc[step-1]);px=a['current_coords'][step-1,parent].astype(float)*cfg['coord_scale']+com
                py=a['predicted_coords'][step-1,parent].astype(float)*cfg['coord_scale']+com
            else:px=py=None
            if step<T:
                x=a['current_coords'][step,slot].astype(float)*cfg['coord_scale']+com
                y=a['predicted_coords'][step,slot].astype(float)*cfg['coord_scale']+com
                atoms,bonds=a['predicted_atomics'][step,slot],a['predicted_bonds'][step,slot]
                if step and not np.array_equal(a['current_coords'][step,slot],a['proposal_coords'][step-1,parent]):
                    raise ValueError('Parent coordinate identity failed')
            else:
                x=a['proposal_coords'][T-1,parent].astype(float)*cfg['coord_scale']+com
                final_path=path.parent/'final_prediction.pt.gz'
                # Trusted original local tensors, no model execution or remote call.
                with gzip.open(final_path,'rb') as file:
                    final=torch.load(file,map_location='cpu',weights_only=True)
                y=final['coords'][slot].numpy().astype(float)
                atoms=final['atomics'][slot].argmax(-1).numpy();bonds=final['bonds'][slot].argmax(-1).numpy()
                sources.append({'path':str(final_path.relative_to(repo)),'sha256':digest(final_path)})
            f=features(x,y,atoms,bonds,points,origin,px,py,config['regions'])
            delta=None if previous_f is None else {k:f[k]-previous_f[k] for k in f if isinstance(f[k],(float,int)) and isinstance(previous_f[k],(float,int))}
            text=sentence(f,changes=delta,regions=config['regions'])
            if step==T:
                text=text.replace('当前噪声/生成槽位云','t=1积分态').replace('此刻模型预测的终点云','t=1最终解码读出云')
                text+='\n最终读出重新调用模型，亲和力/应变评估对应解码分子，不是直接评估积分proposal。'
            lines.append(f'\n### t={step/T:.2f}\n'+text)
            previous_f=f
        sources.append({'path':str(path.relative_to(repo)),'sha256':digest(path)})
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'quality_paths_full_generation.zh-CN.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    write_json(output/'full_description_provenance.json',{'config':config,'sources':sources,'script_sha256':digest(__file__),
        'nodes_per_path':T+1,'described_paths':len(donors),'final_readout_not_proposal':True})
    print('Described',len(donors),'paths,',T+1,'nodes per path')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[1];describe(repo,read_json(repo/a.config),a.output)
