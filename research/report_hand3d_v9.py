import json,hashlib,shutil,tarfile
from pathlib import Path
import numpy as np,torch,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hand3d_v8_common import V7,save
from hand3d_temporal_v7 import WRIST,CHAINS
import spatial_rgb_common as s
RUN=V7.parent/'proposal_critic_v9';V9=V7.parent/'offline_hand3d_v9';ROLLOUT=V7.parent/'offline_hand3d_v9_rollout'

def main():
    out=RUN/'delivery';out.mkdir(exist_ok=True);fresh=json.loads((RUN/'fresh_results.json').read_text());old=json.loads((RUN/'old_results.json').read_text());seal=json.loads((RUN/'fresh_evaluation_seal.json').read_text());assert fresh['passed']
    notes='''3D DiT v9 开发记录（2026-10-04）

1. 时序作用不足：v8移除历史后相对误差只变化0.07mm。改为17×20×3整段轨迹联合预测，使用全段3D/速度/骨长监督，17×21根/相对token及每帧192空间单元。v9第一轮完整候选移除历史后相对23.31→31.09mm，证明已使用历史；有界结果的变化仍小。
2. 训练与部署差异：追加真实10步、4采样闭环训练。验证和训练使用同一向量化采样器。
3. 大幅改动风险：用3折排除受试者的真实提案训练整手收益/损伤判断器，根和手指按同一系数修改；拒绝时回到绝对/相对位移≤9.5mm的有界结果。OOF仅涉及提案头和风险头，共享2D编码器/预训练重叠仍存在，不宣称整条流水线OOF。
4. 锁定第二批新片段：另12个未使用片段，550窗口/10450非手腕点。相对33.83→29.18mm、绝对46.32→41.47mm；1476坏相对点恢复≤20mm；准确点改坏0/192（相对）和0/442（绝对）。44手窗口接受较大修正，其他窗口使用有界结果。
5. 旧数据诊断：47困难窗相对40.49→36.62mm，90坏点恢复、0准确点改坏。8个严重窗81.66→78.80mm，仍远未恢复好。完整未筛候选为70.64mm，不可直接自动覆盖。
6. 推理：原始RGB+WiLoR XYZ，内部整段3D、输出当前20×3；锁定点严格保留，未确认点仍需复核，完整候选及4次采样保留。
7. v10发现并修复的训练问题：无标注历史帧占位坐标曾进入扩散加噪，修改这些值导致loss变化0.02460。屏蔽无标签残差并加入缺帧注意力屏蔽后，loss变化和缺帧latent对当前输出的变化均为0。v9推理没有读取GT，已公布实测保留；新的修复试验重新训练，旧试验留档。
8. 后续：直接使用每空间位置1280维原生视觉特征，比较2D压缩128维；两组都采用目标/缺帧修复，并分别训练DiT和回归。严重遮挡改善仍是未完成的目标。
''';(out/'DEVELOPMENT_NOTES.txt').write_text(notes,encoding='utf-8')
    vals=[fresh['baseline'],fresh['fallback'],fresh['final']];labels=['WiLoR','Bounded trajectory DiT','OOF proposal critic + DiT'];fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,key,title in zip(axes,['camera_mm','relative_mm'],['Camera MPJPE19','Wrist-relative MPJPE19']):
        y=[m[key] for m in vals];ax.bar(range(3),y,color=['#8b98a8','#5b9b8e','#447db9']);ax.set_xticks(range(3),labels,rotation=8);ax.set_ylim(0,max(y)*1.18);ax.set_ylabel('mm, lower is better');ax.set_title(title)
        for j,v in enumerate(y):ax.text(j,v+.5,f'{v:.2f}',ha='center')
    fig.savefig(out/'metrics.png',dpi=140);plt.close(fig)
    data=torch.load(V7/'data.pt',weights_only=False,mmap=True);cache=torch.load(RUN/'old_predictions.pt',weights_only=False);records,_=s.records_and_index();queue=json.loads((V7.parent/'natural_reliability_v4/delivery/hardcase_review_queue.json').read_text());classes={q['id']:q for q in json.loads((V7.parent/'natural_reliability_v4/temporal_clue_audit/classification.json').read_text())['cases']};lookup={int(i):j for j,i in enumerate(cache['indices'])};cases=[]
    for q in queue:
        if q['id'] not in classes:continue
        j=lookup[q['window_index']];fid=int(data['feature_ids'][q['window_index'],8]);rec=records[fid-1];cam=s.common.from_json(rec['camera']);base=cache['base'][j].numpy();pred=cache['prediction'][j].numpy();raw=cache['proposal'][j].numpy();gt=cache['gt'][j].numpy();mask=cache['valid'][j].numpy().copy();mask[WRIST]=False
        image=cv2.imread(rec['image'])[:,:,::-1];roi=data['roi'][fid].numpy()*1408;lo=np.maximum(np.floor(roi[:2]).astype(int),0);hi=np.minimum(np.ceil(roi[2:]).astype(int),[1408,1408]);crop=image[lo[1]:hi[1],lo[0]:hi[0]]
        if crop.size==0:crop=image;lo=np.array([0,0])
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for ax,xyz,label,color in zip(axes,[base,pred,raw],['WiLoR','Selected3D output','Full3D review candidate'],['#ea9a36','#237db4','#9975b9']):
            ax.imshow(crop);ax.axis('off');ax.set_title(label)
            for x,c in [(gt,'#34b566'),(xyz,color)]:
                uv=cam.eye_to_window(x)-lo
                for chain in CHAINS:ax.plot(uv[chain,0],uv[chain,1],c=c,lw=1.1)
                ax.scatter(uv[:,0],uv[:,1],c=c,s=7)
            ax.set_xlim(0,crop.shape[1]);ax.set_ylim(crop.shape[0],0)
        name=f"case_{q['id']}.png";fig.suptitle(f"Case {q['id']} | green GT is evaluation only; actual XYZ projected",fontsize=10);fig.savefig(out/name,dpi=120);plt.close(fig)
        def err(x):return float(np.linalg.norm((x-x[WRIST])-(gt-gt[WRIST]),axis=-1)[mask].mean()*1000)
        cases.append(dict(id=q['id'],category=classes[q['id']]['category'],reason=classes[q['id']]['reason'],before_mm=err(base),after_mm=err(pred),candidate_mm=err(raw),large_accepted=bool(cache['accepted'][j]),image=name))
    save(out/'hardcases.json',cases);style='body{font:16px/1.75 system-ui;max-width:1140px;margin:32px auto;padding:0 22px;background:#f5f7fb;color:#152a40}section{background:white;padding:22px;border-radius:12px;margin:22px 0}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}img{max-width:100%}.ok{padding:14px;background:#e3f3eb}.warn{padding:14px;background:#fff0d9}a{color:#176bb0}'
    h=f'<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>3D轨迹DiT v9</title><style>{style}</style><h1>3D轨迹 DiT v9：选择性修正通过，严重遮挡继续优化</h1><p class="ok">内部直接预测17×20×3轨迹，输出当前20×3相机XYZ。原始RGB和前后帧均参与，整手提案判断后选择较大修正或有界回退。所有未确认点仍需标注复核。</p><section><h2>第二批锁定片段</h2><p>6来源序列、另12个此前未用过的片段、550窗口、10450个非手腕点。先锁定模型/判断器/策略哈希再打开指标；受试者和来源序列已出现，预训练重叠未知。</p><table><tr><th>方法</th><th>绝对mm</th><th>手指相对mm</th><th>坏相对点恢复≤20mm</th></tr>'
    for label,m in zip(labels,vals):h+=f'<tr><td>{label}</td><td>{m["camera_mm"]:.2f}</td><td>{m["relative_mm"]:.2f}</td><td>{m["relative_bad_recovered20"]}</td></tr>'
    f=fresh['final'];ci=fresh['paired_ci'];h+=f'</table><img src="metrics.png"><p>相对改善{(1-f["relative_mm"]/f["base_relative_mm"])*100:.1f}%；相对误差变化95%CI [{ci["relative"]["ci95_delta_mm"][0]:.2f}, {ci["relative"]["ci95_delta_mm"][1]:.2f}] mm；绝对CI [{ci["camera"]["ci95_delta_mm"][0]:.2f}, {ci["camera"]["ci95_delta_mm"][1]:.2f}] mm，均按来源序列配对。</p><p>原本≤10mm的点被改到&gt;20mm：绝对0/{f["camera_good_points"]}，相对0/{f["relative_good_points"]}。1476/9098坏相对点恢复到20mm以内。44/550手窗口允许较大修正，其余有界回退。零次损伤是本批实测，不是未来必然零损伤。</p></section><section><h2>实际结构</h2><p>YOLO26手框 → WiLoR20点XYZ＋原始RGB → 17帧近密远疏、双向相机对齐 → 每帧192个空间单元、17×21个根/相对token → 4层Transformer、宽192、6头 → 17×20×3联合轨迹。</p><p>全段3D、实际时间差速度、骨长监督；2D热图仅作定位辅助。追加10步/4采样的完整闭环训练，验证使用同一采样器。当前视觉32层仍冻结，原空间128维特征保留192位置。</p><p>判断器用2526个受试者排除后的实际提案、4档修改幅度训练，预测完整手的相机/相对收益与损伤；不独立拼接根/手指门控。此OOF覆盖提案头和风险头，共享监督2D编码器及上游预训练并未整体排除受试者。</p><p>人工确认点原样保留；部分确认会使提案分布改变，该路径保留有界行为。完整3D候选及4次采样另供复核，不是2D加虚构深度。</p></section>'
    diag=json.loads((V9/'conditioning_diagnosis.json').read_text())['results'];h+=f'<section><h2>仍在解决的问题</h2><p class="warn">8个严重旧窗口的相对误差仍为81.66→78.80mm，恢复不足，不能宣称信息缺失已经补好。完整未筛候选为70.64mm，但会改坏其他准确点，所以不会统一自动覆盖。</p><p>第一轮轨迹候选：正常完整候选相对{diag["full"]["raw"]["relative_mm"]:.2f}mm，去除历史并重新计算风险后{diag["no_history"]["raw"]["relative_mm"]:.2f}mm；时序开始有实际作用。有界输出的差异仍小，说明候选质量和可采用范围仍限制效果。</p><p>v10发现：未标注历史帧的占位XYZ也进入了训练加噪。修改它们会让loss变化0.02460。现在屏蔽无标签残差、将缺帧排除于注意力，修改无标签值/缺帧latent均不改变相关输出。该问题属于训练目标处理，v9推理不读取GT，实测结果仍保留。已重启修复对照，并比较原生1280维视觉与128维压缩分支。</p><p>没有逐指可见性真值；不能把有效3D点误差称为仅不可见手指精度。整段无线索的部分可能多解，输出仍是候选。当前依赖已有手框/轨迹，未解决整手漏框。</p></section><section><h2>严重自然案例</h2><p>沿用旧审阅：A表示附近有部分线索，B表示整个审阅片段线索缺乏；并非逐点遮挡真值。绿色GT仅用于展示，实际绘制预测XYZ的投影。</p>'
    for c in cases:h+=f'<details><summary>#{c["id"]} · {c["category"]} · 相对 {c["before_mm"]:.1f}→{c["after_mm"]:.1f}mm；完整候选 {c["candidate_mm"]:.1f}mm</summary><p>{c["reason"]}</p><img loading="lazy" src="{c["image"]}"></details>'
    h+='</section><section><h2>证据与使用</h2><a href="DEVELOPMENT_NOTES.txt">简明开发记录</a> · <a href="fresh_results.json">新片段结果</a> · <a href="old_results.json">旧困难案例</a> · <a href="fresh_evaluation_seal.json">锁定哈希</a> · <a href="README.txt">服务器推理说明</a></section></html>';(out/'report.html').write_text(h,encoding='utf-8')
    for name in ['fresh_results.json','old_results.json','fresh_evaluation_seal.json','critic_selection.json','critic_config.json','critic_history.json','critic_done.json','example_output.json']:
        shutil.copy2(RUN/name,out/name)
    shutil.copy2(V9/'conditioning_diagnosis.json',out/'conditioning_diagnosis.json');shutil.copy2(V9/'track_identity_audit.json',out/'track_identity_audit.json');shutil.copy2(V7.parent/'offline_hand3d_v7_bounded/example_input.json',out/'example_input.json')
    shutil.copy2(RUN/'critic.pt',out/'critic.pt');shutil.copy2(ROLLOUT/'rgb_dit/best.pt',out/'trajectory_dit.pt');code=out/'code';code.mkdir(exist_ok=True)
    for p in Path(__file__).resolve().parent.glob('*.py'):shutil.copy2(p,code/p.name)
    for stage,parent,arms in [('trajectory_initial',V9,['rgb_dit','rgb_regression']),('trajectory_rollout',ROLLOUT,['rgb_dit','rgb_regression']),('actual_oof',RUN,['fold0','fold1','fold2'])]:
        for arm in arms:
            dest=out/'training'/stage/arm;dest.mkdir(parents=True,exist_ok=True)
            for name in ['config.json','history.json','training_done.json','done.json','rollout_preflight.json']:
                if (parent/arm/name).exists():shutil.copy2(parent/arm/name,dest/name)
    (out/'README.txt').write_text(f'''HOT3D3D轨迹DiT v9\n\n代码：/mnt/why/hot3d_hand_residual/infer_hand3d_v9.py\n检查点：{ROLLOUT}/rgb_dit/best.pt + {RUN}/critic.pt\n运行：/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python /mnt/why/hot3d_hand_residual/infer_hand3d_v9.py --input {RUN}/delivery/example_input.json --output {RUN}/prediction.json --device cuda:0\n\n输入原始RGB、YOLO框/框分数、相机标定/外参、时间戳、WiLoR20×3 XYZ米。available_3d只表示存在坐标；confirmed_3d为人工锁定。输出xyz_camera_m为选择性3D结果；完整candidate_xyz_camera_m和4次采样保留。未确认点需复核，例子34锁点仅工程fixture。\n本地包保存模型/源码/报告；完整WiLoR/MANO/风险模型/RGB依赖在服务器，不是无依赖Windows程序。\n状态：两次新片段验证已支持选择性修正；严重遮挡仍未训好，v10修复/原生视觉对照继续。\n''',encoding='utf-8')
    save(out/'CURRENT_PIPELINE.json',dict(version='3d_trajectory_v9',output='20x3 camera XYZ meters',internal='17x20x3 trajectory',selective_correction_approved=True,full_candidate_auto_approved=False,severe_occlusion_resolved=False,goal_work_continues=True,run=str(RUN),report=str(out/'report.html')))
    save(out/'manifest.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'});archive=RUN/'offline_hand3d_v9_delivery.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as f:f.add(out,arcname='offline_hand3d_v9')
    save(RUN/'delivery_receipt.json',dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()));print(json.dumps(dict(report=str(out/'report.html'),bytes=archive.stat().st_size)),flush=True)

if __name__=='__main__':main()
