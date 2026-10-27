"""Render the completed paired pilot audit into tables and a source-bound report."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean


def csv_file(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--valence", required=True)
    parser.add_argument("--output", required=True)
    args=parser.parse_args()
    raw=Path(args.audit).read_bytes();audit=json.loads(gzip.decompress(raw))
    valence=json.loads(Path(args.valence).read_text(encoding="utf-8"))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    result=audit["results"]
    assert audit["source_files_unchanged"] and sum(r["count"] for r in audit["pilot_attempts"].values())==100
    summary=[];paired=audit["paired_records"];seeds=[];derived={}
    for entry in result["results"]:
        target=entry["target"];group=audit["groups"][target]
        assert group["all_step_coordinates_unreplaced"] and group["native_random_streams_preserved"] and group["chemical_step_count"]==90
        assert all(r["equal"] for r in group["first_step"].values())
        derived[target]=dict(changes=entry["changes"],valence=valence[target],arms={})
        for arm in ["control","treatment"]:
            data=group[arm];agg=data["aggregate"];records=data["records"];qc=data["quality_complete"]
            assert agg["attempts"]==len(records)==50 and qc["checked_attempts"]==10 and qc["not_selected"]==40
            assert {(r['logical_seed'],r['logical_slot']) for r in qc['selection']} == {(s,i) for s in audit['study']['seeds'] for i in (0,1)}
            built=[r for r in records if r["build_success"]]
            recoveries=[r["plip_recovery"] for r in data["quality"] if r.get("plip_recovery") is not None]
            values=dict(target=target,arm=arm,attempts=50,mean_pic50=agg["mean_pic50"],C_A=agg["C"],gap=agg["gap"],
                best_pic50=agg["best"],built=agg["built"],built_connected=sum(r.get("connected",False) for r in records),
                unique_smiles=agg["unique_smiles"],built_only_mean_pic50=mean(r["pic50_on_rescore"] for r in built) if built else None,
                checked_attempts=10,not_selected=40,coverage=.2,posebusters_pass=qc["posebusters_pass"],
                posebusters_evaluable=qc["posebusters_evaluable"],plip_evaluable=qc["plip_evaluable"],
                selected_mean_plip_recovery=mean(recoveries) if recoveries else None,
                selected_plip_mean_denominator=len(recoveries),plip_unevaluable=qc["plip_unevaluable_built"])
            assert abs(mean(r["pic50_on_rescore"] for r in records)-agg["mean_pic50"])<1e-12
            assert abs(mean(r["condition_rmsd_A"] for r in records)-agg["C"])<1e-12
            summary.append(values)
            derived[target]["arms"][arm]=dict(values,failure_reasons=dict(Counter(r["failure_reason"] for r in records if not r["build_success"])))
        for i,seed in enumerate(audit["study"]["seeds"]):
            seeds.append(dict(target=target,seed=seed,paired_slots=10,
                delta_pic50=entry["changes"]["pic50_on_rescore"]["seed_deltas"][i],
                delta_C_A=entry["changes"]["condition_rmsd_A"]["seed_deltas"][i]))
    csv_file(out/"summary.csv",summary);csv_file(out/"paired_records.csv",paired);csv_file(out/"seed_deltas.csv",seeds)
    evidence=dict(time_shanghai=audit["time_shanghai"],code_commit=audit["code_commit"],audit_sha256=hashlib.sha256(raw).hexdigest(),
        source_json_sha256=hashlib.sha256(gzip.decompress(raw)).hexdigest(),new_attempts=100,reused_attempts=100,
        source_campaign_paused=True,results=derived)
    (out/"summary.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    lines=["# 化学信息 inpainting 配对实验结果", "",
        "2026-10-27，Asia/Shanghai。新增100次尝试已完成；两教师各复用50个原始对照，并各生成50个化学inpainting候选。两组固定10/50的PLIP/PoseBusters诊断均完成。", "",
        "**结论：当前硬注入方案改善了两位教师的几何条件和抽样互作恢复，但平均pIC50收益不稳定，同时降低构建率、连通性和部分多样性，尚不适合直接替换现行坐标梯度引导。**", "",
        "## 主要比较", "",
        "| 教师 | 条件 | 平均重评分pIC50 ↑ | C（Å）↓ | gap ↓ | 构建/50 | 构建且连通/50 | 唯一SMILES |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        label="坐标梯度" if r['arm']=='control' else "坐标梯度＋化学inpainting"
        lines.append(f"| {r['target'].split('_')[0]} | {label} | {r['mean_pic50']:.6f} | {r['C_A']:.6f} | {r['gap']:.6f} | {r['built']} | {r['built_connected']} | {r['unique_smiles']} |")
    lines += ["", "所有50次尝试都进入平均pIC50和C，包含构建失败的潜在分子。pIC50来自原模型对终态类别与坐标的重新评分，不能视为实验活性测量。", "",
        "| 教师 | ΔpIC50（处理−对照） | 五种子探索性95%区间 | ΔC（Å） | 五种子探索性95%区间 |", "|---|---:|---|---:|---|"]
    for target,data in derived.items():
        p=data['changes']['pic50_on_rescore'];c=data['changes']['condition_rmsd_A']
        fmt=lambda a:f"[{a[0]:+.4f}, {a[1]:+.4f}]"
        lines.append(f"| {target.split('_')[0]} | {p['mean_delta']:+.6f} | {fmt(p['exploratory_95pct_t_interval'])} | {c['mean_delta']:+.6f} | {fmt(c['exploratory_95pct_t_interval'])} |")
    lines += ["", "每个种子保留10个配对槽位，以五个种子均值形成t区间（df=4）。两教师的pIC50区间均跨零；C在每个种子均下降。配置选择也使用了这五个种子，区间仅描述本次固定种子测试，不构成留出验证。", "",
        "## 固定10/50抽样诊断", "",
        "| 教师 | 条件 | PoseBusters通过/抽样10 | PLIP可评价/10 | PLIP平均互作恢复 |", "|---|---|---:|---:|---:|"]
    for r in summary:
        label="坐标梯度" if r['arm']=='control' else "＋化学inpainting"
        lines.append(f"| {r['target'].split('_')[0]} | {label} | {r['posebusters_pass']}/10 | {r['plip_evaluable']}/10 | {r['selected_mean_plip_recovery']:.6f} |")
    lines += ["", "抽样固定为每种子的logical_slot=0、1。四组的抽样槽均可构建、PLIP均可评价，没有把工程异常记成零互作或通过。两新组各有1个PoseBusters测量不通过。其余40槽未补查；这些比例不外推到50候选。", "",
        "## 失败节点和化学原因", "",
        "6pyy新增6个构建失败，3w54新增4个；首次报告的超价原子全部映射到被注入的核心，且均存在核心外部连接。已用raw SDF坐标与原终态tensor逐原子匹配（最大误差<0.001 Å），不是按SDF原子序号直接假设对应。", "",
        "| 教师 | 种子 / 槽位 | 报错原子 | 核内键级和 | 核外键级和 |", "|---|---|---|---:|---:|"]
    for target,record in valence.items():
        for f in record['failures']:
            assert f.get('in_inpainted_core') is True
            lines.append(f"| {target.split('_')[0]} | {f['seed']} / {f['logical_slot']} | slot {f['bad_atom_slot']}，{f['symbol']}，电荷{f['formal_charge']} | {f['core_bond_order']:g} | {f['external_bond_order']:g} |")
    lines += ["", "典型冲突是：中性O被保留为O，但其外连键级和为3或4；Br接上双键；中性C的核内、核外总键级达到5或6。当前投影只固定核心内部的元素、电荷和诱导键，核心—外部连接仍由模型自由生成，未联合满足核心剩余价态。观测直接支持边界价态冲突的诊断；单次组合干预尚不能区分逐步投影和最终投影各自的贡献。", "",
        "因此下一步应先比较元素/电荷注入与键注入的消融，并对核心外连键加入剩余价态兼容约束，再使用留出种子复验。当前不应直接通过替换坐标或删除失败样本改善指标。", "",
        "## 执行配置与坐标约束", "",
        "两组选择均按完整50人口的平均重评分pIC50最高，未按单候选最高分或C–gap代表组代替。6pyy为UG04/R05/redistribute，scale=[181.01933598375618,724.0773439350247,181.01933598375618,64]；3w54为UG04/R05/decrease，scale=[90.50966799187809,181.01933598375618,181.01933598375618,181.01933598375618]。", "",
        "模型固定flowr_root_v2.ckpt（SHA256 f28e863b2b208718f3d3f85f09837c2f907a71123436db6f98a25d2f1858b6a0），使用原PLIP直接互作核心，每教师9个重原子。实际模型可注入的头为atomics、charges、bonds；杂化类型只记录，因该检查点没有对应头而未注入。键编码遵循原模型去芳香化设置。", "",
        "坐标仍使用原Hungarian条件映射、VJP与FLOWR原生适配；化学硬投影在每步原生类别转移后执行（step10–99），再在原生终态corrector后执行一次。仅投影化学通道，未替换任何坐标、mask或self-conditioning。每个教师90步的投影坐标未写入断言及原生随机流校验全部通过，首次注入前的坐标、梯度增量、噪声、更新结果和映射与已存对照逐值完全一致。", "",
        "终态化学投影后核心特征匹配率为100%，这是硬约束的直接结果，不算独立的优化证据。原生终态投影前也保留，模型upstream affinity在该投影前计算；表中比较仅用投影后的endpoint rescore。", "",
        f"执行代码为 `{audit['code_commit']}`，远端29项工程测试通过（6.51秒）。实验根目录为 `/workspace/agenticgen_chem_inpainting_20261027`，原UG目录与账本未改写、原UG05及原巡检仍暂停。专项账本准确为两笔50次；CPU抽样队列已完成，独立worker在空闲边界停止。", "",
        "## 可复核产物", "",
        "- [协议](protocol.zh-CN.md)", "- [机器可读汇总](summary.json)", "- [四组指标CSV](summary.csv)",
        "- [100个配对槽位](paired_records.csv)", "- [10个种子级变化](seed_deltas.csv)",
        "- [价态错误与原子映射](valence_diagnostics.json)", "- [原始完整审计包](audit.json.gz)", "",
        f"审计包SHA256：`{hashlib.sha256(raw).hexdigest()}`。全部原始SDF、终态和逐步轨迹保留在远端实验根目录；本地结构导出位于 `delivery/20261027_chem_inpainting/`，生成结构未纳入代码版本。", ""]
    (out/'report.zh-CN.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(json.dumps({target:{'delta_pic50':v['changes']['pic50_on_rescore']['mean_delta'],'delta_C':v['changes']['condition_rmsd_A']['mean_delta']} for target,v in derived.items()},ensure_ascii=True))


if __name__=='__main__':main()
