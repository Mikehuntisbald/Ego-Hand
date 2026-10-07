import json,hashlib,shutil,tarfile,html
from pathlib import Path
import numpy as np,torch,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hand3d_v8_common import V7,save
from hand3d_temporal_v7 import WRIST,CHAINS
from calibrate_hand3d_v8 import paired_ci
import spatial_rgb_common as s
RUN=V7.parent/'native_projection_v11';PROPOSAL=V7.parent/'offline_hand3d_v10_rollout';CODE=Path(__file__).resolve().parent

def main():
    out=RUN/'delivery';out.mkdir(exist_ok=True);fresh=json.loads((RUN/'fresh_results.json').read_text());old=json.loads((RUN/'old_results.json').read_text());control=json.loads((RUN/'fresh_regression_results.json').read_text());assert fresh['passed'];assert json.loads((RUN/'raw_inference_checks.json').read_text())['passed']
    a=torch.load(RUN/'fresh_predictions.pt',weights_only=False);b=torch.load(RUN/'fresh_regression_predictions.pt',weights_only=False);assert torch.equal(a['indices'],b['indices'])
    comparisons=paired_ci(a['prediction'],b['prediction'],a['gt'],a['valid'],a['rows']);save(RUN/'fresh_dit_vs_regression.json',comparisons)
    notes='''3D DiT v11 开发记录（2026-10-04）

问题 → 处理 → 结果：
1. 无标签历史帧的占位XYZ进入扩散加噪，扰动它们让loss变化0.0246006。清零无标签残差、屏蔽缺帧注意力后，两项扰动影响均为0。旧试验完整留档，v10重新训练。
2. 128维2D压缩可能丢失3D语义。v10改用WiLoR原生192×1280空间条件，另保留128维定位分支；RGB投影和定位层得到训练。32层原生视觉主干在v11仍冻结。四臂对照和失败记录保留。
3. 训练单次去噪与部署采样不一致。追加真实10步、4采样、17×20×3轨迹闭环训练，并保护原本准确的相机/相对点；选中DiT step900。
4. 迁移旧判断器、重新训练native OOF判断器、加入原生RGB及逐点监督的判断器，都未可靠选出大修正；均未采用。OOF排除的是提案/风险头的训练受试者，共享2D训练与上游预训练不是完整OOF。
5. 整手单系数被一个大偏移压小。改为共同手腕下同时投影到绝对位移和手腕相对位移≤9.5mm的交集；两种约束同时成立。开发集相对25.32→21.69mm，比单系数22.59mm更好。
6. 模型、策略、代码哈希先封存，再评估第三批12未用片段：474窗口/9006非手腕点，相对34.34→28.42mm（17.2%），绝对44.41→40.89mm；1470/7419坏相对点恢复≤20mm；准确点损伤绝对0/597、相对0/268。六来源序列配对CI支持改善，但受试者/来源序列已出现。
7. 仍未解决严重缺信息：旧8窗相对81.66→77.65mm，完整候选67.12mm，依然很差；只恢复3个坏相对点。全候选会损伤其他准确点，继续作为复核候选。整手漏框未解决，推理需要现有手框/轨迹。
8. 原始RGB推理保留34个确认点，GT/可见性/侧别/形状字段投毒不改变输出；128个极端提案/锁点场景同时满足两种位移界。若输入XYZ真的缺失，该手只给复核候选，不在零占位附近伪造自动完成。
9. v12完成原生1280维条件下最后4个WiLoR块的3D联合微调，冻结对照采用相同输入、步数、种子。4块梯度/权重更新已通过，但开发集相对21.47mm，冻结对照21.42mm，差值CI[0.02,0.08]mm，不采用该微调。v13已开始86个既有训练/开发片段的真实30FPS观测准备，中心仍每5帧，邻帧更密、远处稀疏。未加入人工遮挡。目标继续active。

范围：以上有效点是有限3D标注，不是逐指不可见真值；预训练重叠未知。9.5mm界只保证原≤10mm不越过20mm，不保证每次修改有益。未确认点均需标注复核，采样方差不是校准可靠性。
'''
    (out/'DEVELOPMENT_NOTES.txt').write_text(notes,encoding='utf-8')
    labels=['WiLoR','Native temporal regression','Native trajectory DiT'];vals=[fresh['baseline'],control['final'],fresh['final']]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,key,title in zip(axes,['camera_mm','relative_mm'],['Camera MPJPE19','Wrist-relative MPJPE19']):
        y=[v[key] for v in vals];ax.bar(range(3),y,color=['#91a2b4','#72a4ac','#386fb0']);ax.set_xticks(range(3),labels,rotation=8);ax.set_ylim(0,max(y)*1.2);ax.set_title(title);ax.set_ylabel('mm, lower is better')
        for j,v in enumerate(y):ax.text(j,v+.5,f'{v:.2f}',ha='center')
    fig.savefig(out/'metrics.png',dpi=140);plt.close(fig)
    data=torch.load(V7/'data.pt',weights_only=False,mmap=True);cache=torch.load(RUN/'old_predictions.pt',weights_only=False);records,_=s.records_and_index();classes={q['id']:q for q in json.loads((V7.parent/'natural_reliability_v4/temporal_clue_audit/classification.json').read_text())['cases']};lookup={int(i):j for j,i in enumerate(cache['indices'])};cases=[]
    for case in old['cases']:
        if case['id'] not in classes:continue
        j=lookup[case['window_index']];fid=int(data['feature_ids'][case['window_index'],8]);rec=records[fid-1];cam=s.common.from_json(rec['camera']);base=cache['base'][j].numpy();pred=cache['prediction'][j].numpy();raw=cache['proposal'][j].numpy();gt=cache['gt'][j].numpy()
        image=cv2.imread(rec['image'])[:,:,::-1];roi=data['roi'][fid].numpy()*1408;lo=np.maximum(np.floor(roi[:2]).astype(int),0);hi=np.minimum(np.ceil(roi[2:]).astype(int),[1408,1408]);crop=image[lo[1]:hi[1],lo[0]:hi[0]]
        if crop.size==0:crop=image;lo=np.array([0,0])
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for ax,xyz,label,color in zip(axes,[base,pred,raw],['WiLoR','Safe3D output','Full3D review candidate'],['#ec9a34','#2d70ae','#a276bc']):
            ax.imshow(crop);ax.axis('off');ax.set_title(label)
            for points,c in [(gt,'#37b975'),(xyz,color)]:
                uv=cam.eye_to_window(points)-lo
                for chain in CHAINS:ax.plot(uv[chain,0],uv[chain,1],c=c,lw=1.)
                ax.scatter(uv[:,0],uv[:,1],c=c,s=6)
            ax.set_xlim(0,crop.shape[1]);ax.set_ylim(crop.shape[0],0)
        name=f"case_{case['id']}.png";fig.suptitle(f"Case {case['id']} | green GT: evaluation only | projected XYZ does not show depth error",fontsize=9);fig.savefig(out/name,dpi=120);plt.close(fig)
        cases.append(dict(**case,image=name,reason=classes[case['id']]['reason']))
    save(out/'hardcases.json',cases)
    style='body{font:16px/1.7 system-ui;max-width:1150px;margin:32px auto;padding:0 22px;background:#f3f6fa;color:#17304b}section{background:white;padding:24px;border-radius:12px;margin:22px 0}table{border-collapse:collapse;width:100%}td,th{padding:9px;text-align:left;border-bottom:1px solid #dce4ee}img{max-width:100%}.ok{padding:16px;background:#ddf2e7}.warn{padding:16px;background:#fff0d8}a{color:#226eb5}'
    h=f'<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>原生3D轨迹DiT v11</title><style>{style}</style><h1>原生3D轨迹 DiT v11</h1><p class="ok">第三批封存片段通过有界补全验收。输出20×3相机XYZ，使用原始RGB和双向前后帧。严重信息缺失继续优化，全部未确认预测仍需复核。</p><section><h2>第三批12片段实测</h2><p>474窗口、9006个非手腕点；先封存模型/策略/代码哈希，再读取指标。6个来源序列、P0010/P0015受试者已出现，不是新受试者验证。</p><table><tr><th>方法</th><th>绝对mm</th><th>手指相对mm</th><th>坏相对点恢复≤20mm</th><th>准确点损伤：绝对/相对</th></tr>'
    for label,m in zip(labels,vals):h+=f'<tr><td>{label}</td><td>{m["camera_mm"]:.2f}</td><td>{m["relative_mm"]:.2f}</td><td>{m["relative_bad_recovered20"]}</td><td>{m["camera_good_harmed20"]}/{m["camera_good_points"]}；{m["relative_good_harmed20"]}/{m["relative_good_points"]}</td></tr>'
    ci=fresh['paired_ci'];comp=comparisons['relative']['ci95_delta_mm'];h+=f'</table><img src="metrics.png"><p>DiT相对改善17.2%；来源序列配对95%CI：相对变化[{ci["relative"]["ci95_delta_mm"][0]:.2f}, {ci["relative"]["ci95_delta_mm"][1]:.2f}]mm，绝对变化[{ci["camera"]["ci95_delta_mm"][0]:.2f}, {ci["camera"]["ci95_delta_mm"][1]:.2f}]mm。</p><p>同条件回归相对28.60mm，DiT28.42mm；DiT减回归的相对CI [{comp[0]:.2f}, {comp[1]:.2f}]mm。差距较小，不能把改善全归因于diffusion。两种方法均直接输出3D。</p><p>困难组（原相对&gt;40mm）168窗口：48.73→41.83mm，27个坏相对点恢复≤20mm；仍未达到精确标注水平。</p></section><section><h2>当前结构与修正方式</h2><p>YOLO26框/轨迹 → WiLoR20点XYZ＋原始RGB → 17帧近密远疏双向对齐 → 17×21个根/相对token、每帧192×1280原生视觉位置 → 4层/宽192/6头Transformer → 17×20×3轨迹，当前输出20×3米。</p><p>32层WiLoR视觉主干冻结；原生1280→192投影、128维定位层和3D轨迹头已训练。定位热图只作辅助；主要目标是3D轨迹、速度、骨长与真实10步4采样输出。</p><p>当前自动使用共同手腕下的双约束投影：每点绝对位移和相对手腕位移均≤9.5mm。一个大偏移不再把所有手指的修正一起压小。开发集策略先固定；原生OOF判断器未通过，未采用。</p><p>该界保证原≤10mm点不会改到&gt;20mm，但不是每个修改都有益的保证。人工确认点原样复制。真的缺XYZ时保留null并另给复核候选，已有坐标保持原样；未宣称该分布已获得准确性验证。</p></section><section><h2>保留的失败与下一步</h2><p class="warn">旧8个严重窗口：81.66→77.65mm，完整候选67.12mm；只恢复3个坏相对点，严重信息缺失仍没补好。旧47困难窗口和A/B病例保留在下方。</p><p>GT占位进入扩散latent的问题已修复并重新训练。迁移旧判断器、新native OOF判断器、原生RGB逐点判断器均未通过收益与损伤联合要求，失败配置和指标已留档。只排除了提案/风险头受试者，共享2D和上游预训练不构成完整OOF。</p><p>v12正对原生1280维分支一起微调WiLoR最后4块，并保留同种子/输入/步数冻结对照。梯度和真实权重更新已通过，效果尚未验证。最近帧间隔仍1/6秒，没有将其称为30FPS。</p><p>没有逐指不可见真值；有限3D标签误差不等于不可见点专属精度。当前依赖已有手框/轨迹，整手漏检未解决；多次采样间的离散程度也不是校准正确率。</p></section><section><h2>自然严重案例：A附近有线索，B片段缺线索</h2><p>图为真实XYZ投影，绿色是仅供评估的真值。投影重合也可能有很大的深度误差。</p>'
    for case in cases:
        h+=f'<h3>案例 {case["id"]} · {case["category"]} · {html.escape(case["reason"])}</h3><p>手指相对：WiLoR {case["baseline"]["relative_mm"]:.2f} → 安全输出 {case["final"]["relative_mm"]:.2f}mm；完整复核候选 {case["raw"]["relative_mm"]:.2f}mm。坏相对点恢复 {case["final"]["relative_bad_recovered20"]} 个。</p><img loading="lazy" src="{case["image"]}">'
    h+='<section><h2>后续试验更新</h2><p>v12最后4块3D微调已完成且未采用：开发集相对21.47mm，对照21.42mm；微调减冻结CI[0.02,0.08]mm，主要手指指标未改善。主干参数确实得到梯度并更新，不能把完成训练当作效果提升。</p><p>v13正在准备86个既有训练/开发片段的真实30FPS观测。中心仍每5帧，近处1帧间隔、远处稀疏；尚未得到训练/评估收益。上方v12描述保留为v11交付生成时的试验记录。</p><p><a href="native_tail_v12/development_results.json">v12完整开发对照</a></p></section>'
    h+='</section><section><h2>开发文档与证据</h2><p><a href="DEVELOPMENT_NOTES.txt">简明开发记录：问题、解决办法、效果</a> · <a href="fresh_results.json">完整封存实测</a> · <a href="old_results.json">旧困难病例诊断</a> · <a href="raw_inference_checks.json">原始RGB、锁点、GT投毒检查</a> · <a href="README.txt">运行方式</a></p></section></html>'
    (out/'report.html').write_text(h,encoding='utf-8')
    for name in ['fresh_results.json','fresh_regression_results.json','fresh_dit_vs_regression.json','old_results.json','old_regression_results.json','fresh_evaluation_seal.json','development_selection.json','calibration_trials.json','raw_inference_checks.json','example_output.json']:
        shutil.copy2(RUN/name,out/name)
    shutil.copy2(V7.parent/'offline_hand3d_v7_bounded/example_input.json',out/'example_input.json');shutil.copy2(V7/'risk_all.pt',out/'risk_all.pt');shutil.copy2(V7/'risk_calibration.json',out/'risk_calibration.json')
    for arm in ['dit','regression']:
        shutil.copy2(PROPOSAL/f'rgb_{arm}/best.pt',out/f'{arm}.pt')
        for name in ['config.json','history.json','training_done.json','gradient_check.json']:shutil.copy2(PROPOSAL/f'rgb_{arm}'/name,out/f'{arm}_{name}')
    source=out/'code';source.mkdir(exist_ok=True)
    for path in CODE.glob('*.py'):shutil.copy2(path,source/path.name)
    failure=out/'failed_judges';failure.mkdir(exist_ok=True)
    for folder in [V7.parent/'native_critic_v10',V7.parent/'point_critic_native_v11/native',V7.parent/'point_critic_native_v11/geometry']:
        label=folder.name if folder.parent.name!='point_critic_native_v11' else 'point_'+folder.name;target=failure/label;target.mkdir(exist_ok=True)
        for path in folder.glob('*.json'):shutil.copy2(path,target/path.name)
    for name,folder in [('native_initial',V7.parent/'offline_hand3d_v10_native'),('native_oof',V7.parent/'native_oof_v10')]:
        target=out/name;target.mkdir(exist_ok=True)
        for path in folder.rglob('*.json'):
            if 'observations' in str(path) or 'delivery' in str(path):continue
            # Keep concise lineage/config/check evidence; bulky frame manifests stay remote.
            if path.stat().st_size>300000:continue
            dest=target/path.relative_to(folder);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    folder=V7.parent/'native_tail_v12';target=out/'native_tail_v12';target.mkdir(exist_ok=True)
    for path in folder.rglob('*.json'):
        dest=target/path.relative_to(folder);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    (out/'README.txt').write_text(f'''服务器运行（依赖仍在服务器，不是独立Windows应用）：
cd /mnt/why/hot3d_hand_residual
/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python infer_hand3d_v11.py --input {RUN}/delivery/example_input.json --output {RUN}/prediction.json --device cuda:2

输入原始RGB、已有YOLO手框/置信度/轨迹、相机标定/外参、时间戳、WiLoR20×3相机XYZ米。confirmed_3d只用于人工锁定；available_3d表示坐标是否存在，不代表可见或正确。
输出xyz_camera_m是有界结果，candidate_xyz_camera_m和4次采样另供复核。缺少XYZ的输入只暴露复核假设、null和已有坐标，不自动当成已恢复。
第三批12未用片段验证通过有界修正；严重缺信息尚未做好，目标继续active。v12原生3D主干微调不优于冻结对照，未采用；v13真实30FPS近密远疏观测准备中。
''',encoding='utf-8')
    save(out/'CURRENT_PIPELINE.json',dict(version='native3d_projection_v11',output='20x3 current-camera XYZ meters',internal='17x20x3 trajectory',native_condition_channels=1280,spatial_cells=192,automatic_bounded_approved=True,full_candidate_auto_approved=False,severe_occlusion_resolved=False,goal_work_continues=True,run=str(RUN),report=str(out/'report.html')))
    save(out/'manifest.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'})
    archive=RUN/'offline_hand3d_v11_delivery.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as f:f.add(out,arcname='offline_hand3d_v11')
    save(RUN/'delivery_receipt.json',dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()));print(json.dumps(dict(report=str(out/'report.html'),bytes=archive.stat().st_size,comparisons=comparisons)),flush=True)

if __name__=='__main__':main()
