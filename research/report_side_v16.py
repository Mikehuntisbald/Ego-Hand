"""Deliver v16 independent evidence and retain the unresolved severe cases."""
import functools,hashlib,html,json,shutil,tarfile
from pathlib import Path
import cv2,numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hand3d_v8_common import V7,save
from report_dense_adaptive_v14 import picture as base_picture,table
import spatial_rgb_common as s
RUN=V7.parent/'side_native_v16';CODE=Path(__file__).resolve().parent
picture=functools.partial(base_picture,proposal_label='v16 annotation proposal')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out=RUN/'delivery';out.mkdir(exist_ok=True)
    result=json.loads((RUN/'fifth_results.json').read_text());checks=json.loads((RUN/'raw_inference_checks.json').read_text());hard=json.loads((RUN/'hard_results.json').read_text());dev=json.loads((RUN/'development_results.json').read_text())
    assert result['passed'] and checks['passed'];b=result['baseline'];v=result['variants'];f=v['consensus']['final'];hg=v['consensus']['hard'];hs=hard['groups']['severe8']
    for folder in ['code','sealed','evidence']:(out/folder).mkdir(exist_ok=True)
    for p in CODE.glob('*.py'):shutil.copy2(p,out/'code'/p.name)
    for src,name in [(RUN/'consensus/uniform_adaptive/best.pt','dit.pt'),(V7.parent/'side_data_v16/consensus/risk_dense/risk_all.pt','risk.pt'),(V7.parent/'side_data_v16/consensus/risk_dense/calibration.json','risk_calibration.json'),(s.RUN/'sealed/rgb_probe.pt','rgb_probe.pt'),(V7.parent/'natural_reliability_v4/sealed/risk_projection.pt','risk_projection.pt')]:shutil.copy2(src,out/'sealed'/name)
    for name in ['fifth_results.json','fifth_seal.json','deployment_seal.json','development_results.json','hard_results.json','raw_inference_checks.json','example_output.json','dense_engineering_input.json','dense_engineering_prepared.json']:shutil.copy2(RUN/name,out/name)
    for folder,name in [('natural_recovery_v15','development_results.json'),('side_consensus_v16','audit_results.json'),('side_consensus_v16','center_xyz_results.json'),('side_data_v16','ready.json'),('side_data_v16','upstream_results.json'),('side_data_v16','reconstruction_parity.json'),('fifth_dense_v16','fresh_manifest.json'),('fifth_dense_v16','unseen_check.json'),('fifth_dense_v16','side_ready.json'),('native_tail_v12','development_results.json'),('anchor_native_v14_closed','development_results.json')]:
        shutil.copy2(V7.parent/folder/name,out/'evidence'/f'{folder}_{name}')
    for variant in ['control','consensus']:
        for name in ['config.json','history.json','gradient_check.json','done.json']:shutil.copy2(RUN/variant/'uniform_adaptive'/name,out/'evidence'/f'{variant}_{name}')
        shutil.copy2(RUN/variant/'source_provenance.json',out/'evidence'/f'{variant}_source_provenance.json')
    shutil.copytree(V7.parent/'natural_reliability_v4/temporal_clue_audit',out/'temporal_clue_audit',dirs_exist_ok=True)
    cache=torch.load(RUN/'hard_predictions.pt',weights_only=False);old=torch.load(V7/'data.pt',weights_only=False,mmap=True);records,_=s.records_and_index();lookup={int(x):i for i,x in enumerate(cache['indices'])};cases=[]
    for c in hard['cases']:
        if c['category']=='unclassified':continue
        k=lookup[c['window_index']];rec=records[int(old['feature_ids'][c['window_index'],8])-1];name=f'case_{c["id"]}.png';picture(out/name,rec,*[cache[key][k].numpy() for key in ['base','prediction','proposal','gt']]);cases.append({**c,'image':name})
    save(out/'hardcases.json',cases)
    fresh=torch.load(RUN/'fifth_predictions.pt',weights_only=False);pr=fresh['variants']['consensus'];frec=json.loads((V7.parent/'fifth_dense_v16/fresh_rows.json').read_text());mask=fresh['valid'].clone();mask[:,5]=False
    def error(x):return (((x-x[:,5:6])-(fresh['gt']-fresh['gt'][:,5:6])).norm(dim=-1)*1000*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    before=error(fresh['base']);after=error(pr['prediction']);improvement=before-after;indices=list(dict.fromkeys(improvement.argsort(descending=True)[:4].tolist()+after.argsort(descending=True)[:4].tolist()+improvement.argsort()[:4].tolist()));examples=[]
    for k in indices:
        row=fresh['rows'][k];name=f'fresh_{k}.png';rec=frec[int(row['row_index'])];picture(out/name,rec,fresh['base'][k].numpy(),pr['prediction'][k].numpy(),pr['proposal'][k].numpy(),fresh['gt'][k].numpy());examples.append(dict(index=k,row=row,image=name,baseline_relative_mm=float(before[k]),final_relative_mm=float(after[k])))
    save(out/'fresh_examples.json',examples)
    notes=f'''HOT3D 3D DiT v16 开发记录（2026-10-04）

问题 → 处理 → 结果
1. 旧2D不能恢复深度：旧版留档，改为相机20×3 XYZ（米），内部17×20×3。原生RGB192×1280、128维定位支路、4层宽192/6头；10步4采样闭环，非因果双向时间信息。投影/定位/3D头训练，32层主干冻结；最后4块微调对照没有改善手指，未采用。
2. 无标签占位进入扩散latent：清零无效残差、屏蔽缺帧注意力，重新训练；扰动无效标签/缺帧的影响为0。错误试验留档。
3. 近帧太稀、速度下限50毫秒：真实30FPS，最近33/67/100毫秒、远处稀至4秒；相同原始中心做密度对照，速度数值保护改1毫秒。没有人工遮挡；边界缺帧不复制。
4. 固定9.5毫米只能小改：开发集先选逐点风险9.5/50毫米半径、强度0.5，共同手腕下联合限制相机/相对位移；两类准确点损伤门槛都≤1%。旧/原生判断器多次失败、时序锚点失败均保留。v15三组900步困难采样/部署约束训练都选回初始参数，未改善，未采用。
5. 自然困难点有左右手误判：逐帧最高IoU在遮挡下选错。用预测框/侧别置信度做整轨迹投票，支持≥3帧、占比≥0.8，中心置信度<0.7才覆盖；GT不参与选择。重算960个训练/开发观测，重拟合排除自身受试者的风险头，匹配900步3D对照；候选选step900，控制选初始参数。开发困难24窗相对45.65→22.86毫米、恢复18→215点。
6. 重算发现ROI1.3/1.4侧别来源不同：按原WiLoR的1.3ROI重建原始侧别，全部原3D重建与缓存最大差0毫米，才继续。RGB/GT/中心/时槽保持一致，两臂只改侧别XYZ及派生XY。
7. 入口风险投影精度不一致：统一BF16/16帧批处理，避免ROI归一化往返改变边界像素；真实检测→WiLoR→RGB→DiT核对缓存，XYZ最大差{checks['sampler_xyz_max_difference_mm']:.4f}毫米。34确认点、GT投毒、真缺XYZ复核保护通过。
8. 模型/风险/温度/策略/代码/输入先封存，第五批12未用片段{result['windows']}窗/{b['points']}非手腕点：相对{b['relative_mm']:.2f}→{f['relative_mm']:.2f}（{100*(1-f['relative_mm']/b['relative_mm']):.1f}%），相机{b['camera_mm']:.2f}→{f['camera_mm']:.2f}毫米；恢复{f['relative_bad_recovered20']}/{f['relative_bad_points']}坏相对点。准确点损伤相机{f['camera_good_harmed20']}/{f['camera_good_points']}、相对{f['relative_good_harmed20']}/{f['relative_good_points']}。相比同批v14相对25.96→25.56，配对CI支持小幅改善。六来源/受试者已出现，非新受试者/完整OOF验证。
9. 仍有严重失败：旧8例相对81.66→{hs['final']['relative_mm']:.2f}毫米、只恢复{hs['final']['relative_bad_recovered20']}坏点。#41完整候选13.31毫米；#6、#12受约束结果仍退步，B类持续缺直接指节线索仍未可靠恢复。47例、A/B原图和新片段退步均保留。目标继续，不能把整体验收当成全遮挡已解决。

范围：已有手框/轨迹的辅助标注，全部未确认点需复核。整手漏框、真正缺XYZ候选准确性未解决；有限3D标签不是逐指不可见真值。风险是基线错误概率，四次采样分散程度不是校准正确率。大半径保护是实测；确认点保守界。共享视觉监督/上游预训练重叠未知。未证明DiT显著优于同条件3D回归。
'''
    (out/'DEVELOPMENT_NOTES.txt').write_text(notes,encoding='utf-8')
    values=[b,v['v14']['final'],f];labels=['WiLoR','v14','v16'];fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,key,title in zip(axes,['camera_mm','relative_mm'],['Camera MPJPE19','Wrist-relative MPJPE19']):
        y=[m[key] for m in values];ax.bar(labels,y,color=['#8e9eb0','#7da6b5','#2f6fa6']);ax.set_title(title);ax.set_ylabel('mm; lower is better');ax.set_ylim(0,max(y)*1.2)
        for k,n in enumerate(y):ax.text(k,n+.5,f'{n:.2f}',ha='center')
    fig.savefig(out/'metrics.png',dpi=140);plt.close(fig)
    style='body{font:16px/1.7 system-ui;max-width:1160px;margin:32px auto;padding:0 22px;background:#f3f6fa;color:#18324d}section{background:white;padding:24px;border-radius:12px;margin:22px 0}table{border-collapse:collapse;width:100%}td,th{padding:9px;text-align:left;border-bottom:1px solid #dce4ee}img{max-width:100%}.ok{padding:16px;background:#ddf2e7}.warn{padding:16px;background:#fff0d8}a{color:#226eb5}'
    h=f'<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>HOT3D 3D DiT v16</title><style>{style}</style><h1>HOT3D 时序3D DiT v16</h1><p class="ok">第五批未用于选择的片段通过既定恢复与保护验收。输出20×3相机XYZ（米），真实30FPS邻帧＋双向RGB＋离线侧别投票。作为辅助标注；全部未确认点需复核。</p><section><h2>第五批封存验证</h2><p>12未用片段，{result["windows"]}窗，{b["points"]}非手腕点，P0010/P0015和6来源序列已出现，不是新受试者或完整OOF证据。</p>'
    h+=table([('原WiLoR',b),('侧别修正WiLoR',v['consensus']['coarse_input']),('同批v14',v['v14']['final']),('匹配控制',v['control']['final']),('v16完整流程',f),('完整DiT候选，仅复核',v['consensus']['raw'])])
    h+=f'<img src="metrics.png"><p>相对改善{100*(1-f["relative_mm"]/b["relative_mm"]):.1f}%，来源配对95%CI {v["consensus"]["paired_vs_original"]["relative"]["ci95_delta_mm"]} mm。相比v14相对变化CI {result["consensus_comparisons"]["v14"]["relative"]["ci95_delta_mm"]} mm；相比匹配控制，相机差值CI跨0，不宣称所有指标都显著更好。</p><p>准确点原误差≤10 mm，损伤改后&gt;20 mm；恢复原&gt;20 mm改后≤20 mm。困难组原平均相对&gt;40 mm：44.02→{hg["relative_mm"]:.2f} mm，恢复{hg["relative_bad_recovered20"]}/{hg["relative_bad_points"]}点；仍有多数未精确恢复。</p></section><section><h2>结构与已解决的问题</h2><p>YOLO26手框 → WiLoR侧别低置信度时，利用同轨迹预测进行双向一致性判断 → WiLoR20点相机XYZ＋每帧原生192×1280空间RGB → 17帧外参对齐、根/相对手指token → 4层Transformer → 10步4采样 → 联合根/手指约束。</p><p>侧别投票和RGB都保留前后帧；未输入GT侧别、GT形状、GT关键点或可见性。OOF仅排除风险/提案头受试者；上游预训练重叠未知。最近33/67/100毫秒，远处稀疏至±4秒。无人工遮挡。</p><p>32层主干冻结；微调最后4块没有手指收益，已留档。真实采样训练、无效标签屏蔽、ROI/精度/批大小一致性已修复。v15困难采样三组没有收益，未采用。相同条件的开发回归对照尚无DiT显著优势证据。</p><p><a href="DEVELOPMENT_NOTES.txt">简明开发记录</a> · <a href="README.txt">入口与输入</a> · <a href="raw_inference_checks.json">工程核验</a> · <a href="fifth_seal.json">模型/策略封存</a></p></section>'
    h+=f'<section><h2>严重失败继续保留</h2><p class="warn">旧8例：81.66→{hs["final"]["relative_mm"]:.2f} mm；仅恢复{hs["final"]["relative_bad_recovered20"]}坏点，完整候选{hs["raw"]["relative_mm"]:.2f} mm。#6、#12仍退步，不能把通过整体验收当成完全解决遮挡。目标继续优化。</p><p>A：附近有线索；B：整个片段持续缺直接指节线索。旧例已看过，仅作诊断。<a href="temporal_clue_audit/report.html">原始时序图证</a> · <a href="hard_results.json">全部47例</a></p>'
    for c in cases:h+=f'<h3>#{c["id"]} · {c["category"]}</h3><p>{html.escape(c["reason"])} 相对 {c["baseline"]["relative_mm"]:.2f}→{c["final"]["relative_mm"]:.2f} mm；完整候选{c["raw"]["relative_mm"]:.2f} mm；恢复{c["final"]["relative_bad_recovered20"]}点。</p><img loading="lazy" src="{c["image"]}">'
    h+='</section><section><h2>新片段恢复和失败示例</h2><p>按收益最大、残余最大、退步最大各取4例，评估后展示，未用于改模型。绿色GT仅作评估；投影图不能显示深度误差。</p>'
    for c in examples:h+=f'<h3>窗口{c["index"]} · {html.escape(c["row"]["sequence"])}/clip{c["row"]["clip"]}/f{c["row"]["frame"]}</h3><p>相对{c["baseline_relative_mm"]:.2f}→{c["final_relative_mm"]:.2f} mm。</p><img loading="lazy" src="{c["image"]}">'
    h+='</section><section><h2>使用边界</h2><p>现有框/轨迹仍是前提，整手漏框未解决。人工确认点原样复制，带确认点采用保守9.5 mm界。显式缺XYZ的输入保留null与已知值，另给生成候选；该子集准确性没有验证。</p><p>有限3D标签不是逐指不可见真值。风险指基线误差，采样分散程度不是正确率。没有足够观测时，候选只是待核验假设；大半径保护为实测，非每次修改正确保证。</p></section></html>'
    (out/'report.html').write_text(h,encoding='utf-8')
    (out/'README.txt').write_text('''v16服务器原始RGB完整入口：
/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python /mnt/why/hot3d_hand_residual/complete_hand_tracks_v16.py --input tracks.json --output proposals.json --prepared-output wilor_tracks.json --device cuda:2
已有修正WiLoR XYZ的下游入口：infer_hand3d_v16.py --input wilor_tracks.json --output proposals.json --device cuda:2
输入image_size=[1408,1408]；tracks[].frames[]含image、camera（含T_world_from_camera）、timestamp_s、box_xyxy、box_confidence；可含原xyz_camera_m、available_3d、confirmed_3d。轨迹时间递增、建议真实30FPS。已有YOLO26框/轨迹必须提供。
输出HOT3D20 XYZ、手腕5、当前相机、米；内部17×20×3。双向时序侧别投票，GT字段不参与推理。
全部未确认点须复核，candidate与4采样均为假设。人工确认点保留，显式缺XYZ保留null/已有值并给复核候选。不是全遮挡准确恢复承诺。
sealed/包含3D头/风险/校准快照，code/包含源代码，evidence/包含失败和对照。仍依赖服务器WiLoR/MANO/相机toolkit资产，本地目录不是独立Windows应用。
''',encoding='utf-8')
    save(out/'CURRENT_PIPELINE.json',dict(version='native3d_side_consensus_v16',output='20x3 current-camera XYZ meters',internal='17x20x3',spatial_cells=192,native_condition_channels=1280,real_near_interval_s=1/30,locked_acceptance_passed=True,automatic_projection_approved=True,full_candidate_auto_approved=False,severe_occlusion_resolved=False,goal_work_continues=True,raw_adapter=str(CODE/'complete_hand_tracks_v16.py'),run=str(RUN),report=str(out/'report.html')))
    save(out/'manifest.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'})
    archive=RUN/'offline_hand3d_v16_delivery.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(out,arcname='offline_hand3d_v16')
    save(RUN/'delivery_receipt.json',dict(complete=True,files=len(json.loads((out/'manifest.json').read_text())),archive=str(archive),archive_sha256=sha(archive)))
    print((RUN/'delivery_receipt.json').read_text(),flush=True)
if __name__=='__main__':main()
