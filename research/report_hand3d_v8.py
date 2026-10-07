import json,hashlib,shutil,tarfile
from pathlib import Path
import numpy as np,torch,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hand3d_v8_common import RUN,V7,save
from hand3d_temporal_v7 import WRIST,CHAINS
import spatial_rgb_common as s

def main():
    out=RUN/'delivery';out.mkdir(exist_ok=True);fresh=json.loads((RUN/'fresh_results.json').read_text());old=json.loads((RUN/'old_results.json').read_text());seal=json.loads((RUN/'fresh_evaluation_seal.json').read_text());comparison=json.loads((RUN/'fresh_comparisons.json').read_text());assert fresh['passed']
    notes='''3D DiT v8 开发记录（2026-10-04）

1. 单步训练和10步部署采样不一致：改成对完整10步、4次采样均值反向传播；单步去噪保留作对照。
2. 根平移与手指分别门控破坏抵消：改成完整相机XYZ提案统一判断。未受限候选仍会伤害准确点，不能自动覆盖。
3. 学习门控后期趋向全信任：保留开发集早期检查点。自动输出用整手同一系数缩放，绝对/相对位移同时≤9.5mm，保持修正方向；完整候选和每次采样另留给人工复核。
4. 视觉2D投影可能丢失3D线索：分别训练空间投影层和ViT最后4层，记录非零梯度及实际权重更新；开发集未支持替换当前模型，未强行采用。
5. 验证：先冻结开发选择、权重和12个未用过的片段，再评估527窗口/10013非手腕点。相对34.90→30.65mm，绝对43.09→39.69mm，1327坏相对点恢复到20mm内，准确点改坏0/216（相对）、0/542（绝对）。
6. 范围：新片段沿用已出现的受试者/来源序列，WiLoR预训练重叠未知。没有逐指遮挡真值，不能称为不可见手指专用精度或全面泛化。
7. 未解决：8个严重旧案例相对81.66→79.35mm，恢复仍很弱。完整历史信息消融只使开发相对误差21.89→21.95mm，时序利用不足；下一版改成整段3D轨迹联合预测和监督。
8. 当前可用范围：有界3D辅助修正和标注候选；已有手框/轨迹必需，未解决整手漏框。所有未确认点仍需复核。原始2D和v7、失败对照均保留。
'''
    (out/'DEVELOPMENT_NOTES.txt').write_text(notes,encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');labels=['WiLoR','3D regression','3D DiT'];values=[fresh['baseline'],comparison['results']['regression_rollout']['metrics'],fresh['final']]
    for ax,key,title in zip(axes,['camera_mm','relative_mm'],['Camera MPJPE19','Wrist-relative MPJPE19']):
        vals=[x[key] for x in values];ax.bar(labels,vals,color=['#8994a3','#5c9a89','#4b79b2']);ax.set_title(title);ax.set_ylabel('mm, lower is better');ax.set_ylim(0,max(vals)*1.18)
        for j,val in enumerate(vals):ax.text(j,val+.5,f'{val:.2f}',ha='center')
    fig.savefig(out/'metrics.png',dpi=140);plt.close(fig)
    cache=torch.load(RUN/'old_predictions.pt',weights_only=False);data=torch.load(V7/'data.pt',weights_only=False,mmap=True);records,_=s.records_and_index();queue=json.loads((V7.parent/'natural_reliability_v4/delivery/hardcase_review_queue.json').read_text());classification=json.loads((V7.parent/'natural_reliability_v4/temporal_clue_audit/classification.json').read_text());classes={q['id']:q for q in classification['cases']};lookup={int(i):j for j,i in enumerate(cache['indices'])};cases=[]
    for q in queue:
        if q['id'] not in classes:continue
        j=lookup[q['window_index']];fid=int(data['feature_ids'][q['window_index'],8]);rec=records[fid-1];cam=s.common.from_json(rec['camera']);gt=cache['gt'][j].numpy();base=cache['base'][j].numpy();pred=cache['prediction'][j].numpy();raw=cache['output']['raw_xyz_camera_m'][j].numpy();mask=cache['valid'][j].numpy().copy();mask[WRIST]=False
        image=cv2.imread(rec['image'])[:,:,::-1];roi=data['roi'][fid].numpy()*1408;lo=np.maximum(np.floor(roi[:2]).astype(int),0);hi=np.minimum(np.ceil(roi[2:]).astype(int),[1408,1408]);crop=image[lo[1]:hi[1],lo[0]:hi[0]]
        if crop.size==0:crop=image;lo=np.array([0,0])
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for ax,xyz,label,color in zip(axes,[base,pred,raw],['WiLoR','Bounded auto output','Full review candidate'],['#ef9832','#2983be','#9171be']):
            ax.imshow(crop);ax.axis('off');ax.set_title(label)
            for points,c in [(gt,'#39b56b'),(xyz,color)]:
                uv=cam.eye_to_window(points)-lo
                for chain in CHAINS:ax.plot(uv[chain,0],uv[chain,1],lw=1.1,c=c)
                ax.scatter(uv[:,0],uv[:,1],c=c,s=7)
            ax.set_xlim(0,crop.shape[1]);ax.set_ylim(crop.shape[0],0)
        fig.suptitle(f"Case {q['id']} | green GT for review only | true3D points projected to RGB",fontsize=10);name=f"case_{q['id']}.png";fig.savefig(out/name,dpi=120);plt.close(fig)
        def error(x):return float(np.linalg.norm((x-x[WRIST])-(gt-gt[WRIST]),axis=-1)[mask].mean()*1000)
        cases.append(dict(id=q['id'],category=classes[q['id']]['category'],reason=classes[q['id']]['reason'],relative_before_mm=error(base),relative_auto_mm=error(pred),relative_candidate_mm=error(raw),image=name,current_frames=[records[int(f)-1]['frame'] if f else None for f in data['feature_ids'][q['window_index']].tolist()]))
    save(out/'hardcases.json',cases)
    style='body{font:16px/1.75 system-ui;max-width:1120px;margin:32px auto;padding:0 22px;background:#f5f7fb;color:#172b42}section{background:white;padding:22px;border-radius:12px;margin:22px 0}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}img{max-width:100%}.status{padding:14px;background:#e4f3ed}.warn{padding:14px;background:#fff0db}a{color:#1269aa}'
    h=f'<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>3D DiT v8</title><style>{style}</style><h1>3D DiT v8：有界修正通过验证，重遮挡仍需改进</h1><p class="status">直接输出20×3相机XYZ，单位米。自动有界修正通过锁定的12个新片段验证；完整候选需要人工复核。后续优化仍在继续。</p><section><h2>锁定片段结果</h2><p>6个来源序列、12个未用过的片段、527窗口、10013个非手腕3D点。模型与策略由训练/开发集选择，先记录权重和片段哈希再打开这些指标。受试者和来源序列曾经出现，不是跨新人泛化证明。</p><table><tr><th>方法</th><th>绝对误差mm</th><th>手指相对误差mm</th><th>相对坏点恢复≤20mm</th></tr>'
    for label,m in zip(labels,values):h+=f'<tr><td>{label}</td><td>{m["camera_mm"]:.2f}</td><td>{m["relative_mm"]:.2f}</td><td>{m["relative_bad_recovered20"]}</td></tr>'
    h+='</table><img src="metrics.png">'
    f=fresh['final'];ci=fresh['paired_ci'];h+=f'<p>手指相对改善{(1-f["relative_mm"]/f["base_relative_mm"])*100:.1f}%；来源序列配对95%CI：[{ci["relative"]["ci95_delta_mm"][0]:.2f}, {ci["relative"]["ci95_delta_mm"][1]:.2f}] mm。绝对误差变化CI：[{ci["camera"]["ci95_delta_mm"][0]:.2f}, {ci["camera"]["ci95_delta_mm"][1]:.2f}] mm。</p><p>原本≤10mm的点被改到&gt;20mm：绝对 {f["camera_good_harmed20"]}/{f["camera_good_points"]}；相对 {f["relative_good_harmed20"]}/{f["relative_good_points"]}。误差超过20mm的相对点中，{f["relative_bad_recovered20"]}/{f["relative_bad_points"]}恢复到20mm以内。</p><p>DiT相对误差比同输入回归低0.29mm，来源序列CI支持这组数据的小幅优势。闭环与单步去噪在有界输出上差距很小，不能宣称全面优于单步训练。</p></section>'
    h+='<section><h2>实际网络与策略</h2><p>YOLO26手框 → 原始WiLoR XYZ及RGB → 17帧双向近密远疏采样 → 历史XYZ对齐当前相机 → 192个空间RGB单元＋3D时序编码 → 4层/宽192/6头DiT → 10步采样、4次采样均值 → 20×3提案。</p><p>训练对真实采样闭环求梯度，并直接监督相机XYZ、相对姿态和相机投影。自动结果使用完整3D残差的同一缩放系数，绝对及相对位移均限制到9.5mm以内，保持根与手指修正的抵消关系。按三角不等式，输入误差≤10mm的点不能因该界限直接越过20mm；这只保证损伤上限，不保证输出准确。</p><p>当前选中的32层WiLoR主干与原空间投影冻结。另做空间投影3D微调、最后4层3D微调和匹配冻结对照，真实梯度/权重更新均通过；开发结果未支持替换当前检查点。</p><p>GT关节、GT侧别、GT形状和可见性不进入推理。确认点硬锁定；有坐标的模型预测仍可修正。YOLO分数是手框置信度，不是逐指可见性或补全可信度。</p></section>'
    history=json.loads((RUN/'history_diagnosis.json').read_text())['results'];abl=json.loads((RUN/'conditioning_ablation.json').read_text());h+=f'<section><h2>尚未解决的问题</h2><p class="warn">8个最严重旧案例的相对误差仍有81.66→79.35mm，恢复很弱。未受限候选在新片段的绝对准确点损伤为{fresh["raw"]["camera_good_harm_rate"]*100:.1f}%，不能自动采用。手框/轨迹完全缺失也尚未解决。</p><p>开发消融：正常RGB相对误差{abl["normal"]["relative_mm"]:.2f}mm；RGB清零{abl["ablations"]["zero_rgb"]["relative_mm"]:.2f}mm，说明RGB有实际作用。但移除历史XYZ、RGB、历史元数据并重新计算风险后，相对误差只从{history["full"]["relative_mm"]:.2f}变为{history["no_history"]["relative_mm"]:.2f}mm，时序收益不足。下一版改为整段3D轨迹联合预测/监督，优先恢复前后帧存在指节线索的案例。</p><p>没有逐指遮挡真值，当前指标是有限3D标注点误差，不是仅不可见手指误差。整个片段都缺线索的案例仍可能有多解，完整采样候选不是事实标注。</p></section><section><h2>8个严重自然案例</h2><p>A：片段附近有部分可见线索，需进一步利用；B：审阅片段仍缺线索。沿用既有人工审阅分类，并非逐点可见性真值。绿色GT只用于展示。表中相对误差单位mm。</p>'
    for c in cases:h+=f'<details><summary>#{c["id"]} · {c["category"]} · WiLoR {c["relative_before_mm"]:.1f} → 自动 {c["relative_auto_mm"]:.1f}；完整候选 {c["relative_candidate_mm"]:.1f}</summary><p>{c["reason"]}</p><img loading="lazy" src="{c["image"]}"></details>'
    h+='</section><section><h2>文件与证据</h2><a href="DEVELOPMENT_NOTES.txt">简明开发记录</a> · <a href="fresh_results.json">新片段指标</a> · <a href="old_results.json">旧困难组指标</a> · <a href="history_diagnosis.json">完整历史消融</a> · <a href="fresh_evaluation_seal.json">评估锁定哈希</a> · <a href="README.txt">服务器推理说明</a></section></html>'
    (out/'report.html').write_text(h,encoding='utf-8')
    for name in ['fresh_results.json','old_results.json','fresh_comparisons.json','conditioning_ablation.json','history_diagnosis.json','protocol.json','preflight.json','fresh_manifest.json','fresh_manifest_receipt.json','fresh_evaluation_seal.json','development_selection.json','calibration_trials.json','example_output.json','raw_inference_checks.json']:
        shutil.copy2(RUN/name,out/name)
    shutil.copy2(V7.parent/'offline_hand3d_v7_bounded/example_input.json',out/'example_input.json')
    selected=out/seal['folder'];selected.mkdir(exist_ok=True);shutil.copy2(RUN/seal['folder']/'best.pt',selected/'best.pt')
    code=out/'code';code.mkdir(exist_ok=True)
    for path in Path(__file__).resolve().parent.glob('*.py'):shutil.copy2(path,code/path.name)
    for folder in seal['controls']:
        dest=out/'training'/folder;dest.mkdir(parents=True,exist_ok=True)
        for name in ['config.json','history.json','training_done.json','gradient_check.json','weight_update_check.json','initial_parity.json']:
            if (RUN/folder/name).exists():shutil.copy2(RUN/folder/name,dest/name)
    readme=f'''HOT3D offline3D RGB DiT v8\n\n服务器代码：/mnt/why/hot3d_hand_residual/infer_hand3d_v8.py\n实验：{RUN}\n运行：/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python /mnt/why/hot3d_hand_residual/infer_hand3d_v8.py --input {RUN}/delivery/example_input.json --output {RUN}/prediction.json --device cuda:0\n\n输入：1408×1408原始RGB路径、YOLO手框/框分数、相机标定/外参、时间戳、原始WiLoR20×3相机XYZ（米）。available_3d只表示存在坐标；confirmed_3d表示人工锁定。测试示例锁定点仅为工程fixture，不是人工GT。\n输出：xyz_camera_m为通过本批验证的有界3D修正；candidate_xyz_camera_m和candidates_xyz_camera_m为完整候选/各采样。未确认点都需要标注复核。超出有界范围的候选另标needs_special_review。\n\n本地包保留代码、选中权重、记录和报告；完整WiLoR/MANO/风险模型/原RGB依赖仍在服务器。其他试验权重和原2D/v7均保留服务器原实验目录。\n状态：选择性修正达到当前门槛，严重遮挡仍未充分恢复，下一版3D轨迹模型继续优化。\n'''
    (out/'README.txt').write_text(readme,encoding='utf-8');save(out/'CURRENT_PIPELINE.json',dict(version='3d_v8',output='20x3 camera XYZ meters',bounded_correction_approved=True,full_candidate_auto_approved=False,severe_occlusion_resolved=False,goal_work_continues=True,run=str(RUN),report=str(out/'report.html')))
    save(out/'manifest.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'})
    archive=RUN/'offline_hand3d_v8_delivery.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as f:f.add(out,arcname='offline_hand3d_v8')
    save(RUN/'delivery_receipt.json',dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
    print(json.dumps(dict(report=str(out/'report.html'),files=len(json.loads((out/'manifest.json').read_text())),bytes=archive.stat().st_size)),flush=True)

if __name__=='__main__':main()
