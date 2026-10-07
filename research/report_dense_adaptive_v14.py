"""Deliver fixed acceptance evidence, natural failures and deployable sources."""
import hashlib,html,json,shutil,tarfile
from pathlib import Path
import cv2,numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hand3d_v8_common import V7,save,metrics
from hand3d_temporal_v7 import CHAINS
import spatial_rgb_common as s

RUN=V7.parent/'adaptive_projection_v14/dit_dense';CODE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def picture(dest,row,base,pred,raw,gt,proposal_label='v14 annotation proposal'):
    cam=s.common.from_json(row['camera']);image=cv2.imread(row['image'])[:,:,::-1]
    box=np.asarray(row['box']);size=np.maximum(box[2:]-box[:2],40);center=(box[:2]+box[2:])/2
    lo=np.maximum(np.floor(center-size*.8).astype(int),0);hi=np.minimum(np.ceil(center+size*.8).astype(int),[1408,1408])
    crop=image[lo[1]:hi[1],lo[0]:hi[0]]
    if crop.size==0:crop=image;lo=np.array([0,0])
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,xyz,label,color in zip(axes,[base,pred,raw],['WiLoR',proposal_label,'Full diffusion review candidate'],['#d98b25','#236dad','#a56bc0']):
        ax.imshow(crop);ax.set_title(label);ax.axis('off')
        for points,c in [(gt,'#1da86e'),(xyz,color)]:
            uv=cam.eye_to_window(points)-lo
            for chain in CHAINS:ax.plot(uv[chain,0],uv[chain,1],c=c,lw=1)
            ax.scatter(uv[:,0],uv[:,1],c=c,s=8)
        ax.set_xlim(0,crop.shape[1]);ax.set_ylim(crop.shape[0],0)
    fig.suptitle('Green: GT for evaluation only. Projected XYZ does not reveal depth error.',fontsize=9)
    fig.savefig(dest,dpi=130);plt.close(fig)

def table(items):
    out='<table><tr><th>方法</th><th>相机误差 mm</th><th>手指相对 mm</th><th>坏相对点恢复至≤20 mm</th><th>准确点损伤：相机 / 相对</th></tr>'
    for label,m in items:
        out+=f'<tr><td>{html.escape(label)}</td><td>{m["camera_mm"]:.2f}</td><td>{m["relative_mm"]:.2f}</td><td>{m["relative_bad_recovered20"]}</td><td>{m["camera_good_harmed20"]}/{m["camera_good_points"]} / {m["relative_good_harmed20"]}/{m["relative_good_points"]}</td></tr>'
    return out+'</table>'

def main():
    out=RUN/'delivery';out.mkdir(exist_ok=True)
    fresh=json.loads((RUN/'fourth_results.json').read_text());checks=json.loads((RUN/'raw_inference_checks.json').read_text());hard=json.loads((RUN/'hard_results.json').read_text())
    assert fresh['passed'] and checks['passed'] and hard['complete']
    assert fresh['groups']['baseline_relative_over40mm']['final']['relative_bad_recovered20']>0
    head=V7.parent/'native_density_v13/dit_dense';risk=V7.parent/'aligned_density_v13/risk_dense'
    for folder in ['code','sealed','evidence']: (out/folder).mkdir(exist_ok=True)
    for p in CODE.glob('*.py'):shutil.copy2(p,out/'code'/p.name)
    for src,name in [(head/'best.pt','dit.pt'),(risk/'risk_all.pt','risk.pt'),(risk/'calibration.json','risk_calibration.json'),(s.RUN/'sealed/rgb_probe.pt','rgb_probe.pt'),(V7.parent/'natural_reliability_v4/sealed/risk_projection.pt','risk_projection.pt')]:shutil.copy2(src,out/'sealed'/name)
    for name in ['fourth_results.json','fourth_seal.json','deployment_seal.json','selection.json','raw_inference_checks.json','projection_checks.json','hard_results.json','example_output.json','dense_engineering_input.json']:
        shutil.copy2(RUN/name,out/name)
    for folder,name in [('native_density_v13','development_selection.json'),('native_tail_v12','development_results.json'),('anchor_native_v14_closed','development_results.json'),('hardcase_alignment_v13','results.json'),('aligned_density_v13','ready.json'),('fourth_dense_v14','fresh_manifest.json'),('fourth_dense_v14','unseen_check.json'),('fourth_dense_v14','prepared.json')]:
        src=V7.parent/folder/name
        if src.exists():shutil.copy2(src,out/'evidence'/f'{folder}_{name}')
    for arm in ['dit_sparse','dit_dense','regression_dense']:
        for name in ['config.json','history.json','gradient_check.json','done.json']:
            shutil.copy2(V7.parent/'native_density_v13'/arm/name,out/'evidence'/f'{arm}_{name}')
    for arm in ['blend0','blend0.5','blend1']:
        for name in ['config.json','history.json','done.json']:
            src=V7.parent/'anchor_native_v14_closed'/arm/name
            if src.exists():shutil.copy2(src,out/'evidence'/f'anchor_{arm}_{name}')
    clue=V7.parent/'natural_reliability_v4/temporal_clue_audit'
    shutil.copytree(clue,out/'temporal_clue_audit',dirs_exist_ok=True)
    original,record_index=s.records_and_index();lookup={int(i):j for j,i in enumerate(torch.load(RUN/'hard_predictions.pt',weights_only=False)['indices'])}
    old_data=torch.load(V7/'data.pt',weights_only=False,mmap=True);cache=torch.load(RUN/'hard_predictions.pt',weights_only=False);severe=[]
    for case in hard['cases']:
        if case['category']=='unclassified':continue
        k=lookup[case['window_index']];fid=int(old_data['feature_ids'][case['window_index'],8]);row=original[fid-1]
        name=f'case_{case["id"]}.png';picture(out/name,row,*[cache[key][k].numpy() for key in ['base','prediction','proposal','gt']]);severe.append({**case,'image':name})
    save(out/'hardcases.json',severe)
    fc=torch.load(RUN/'fourth_predictions.pt',weights_only=False);records=json.loads((V7.parent/'fourth_dense_v14/fresh_rows.json').read_text())
    m=fc['valid'].clone();m[:,5]=False
    def errors(x):return (((x-x[:,5:6])-(fc['gt']-fc['gt'][:,5:6])).norm(dim=-1)*1000*m).sum(-1)/m.sum(-1).clamp_min(1)
    before=errors(fc['base']);after=errors(fc['prediction']);change=before-after
    best=change.argsort(descending=True)[:4];worst=after.argsort(descending=True)[:4];regressed=change.argsort()[:4]
    selected=list(dict.fromkeys(best.tolist()+worst.tolist()+regressed.tolist()));examples=[]
    for k in selected:
        row=fc['rows'][k];rec=records[int(row['row_index'])];name=f'fresh_{k}.png'
        picture(out/name,rec,*[fc[key][k].numpy() for key in ['base','prediction','proposal','gt']])
        examples.append(dict(index=k,row=row,image=name,baseline_relative_mm=float(before[k]),final_relative_mm=float(after[k]),selection='Post-evaluation illustrative mining: strongest improvement, largest remaining error, largest regression; never used to change model/policy'))
    save(out/'fresh_examples.json',examples)
    b=fresh['baseline'];f=fresh['final'];ci=fresh['paired_ci'];g=fresh['groups']['baseline_relative_over40mm'];sv=hard['groups']['severe8']
    notes=f'''HOT3D 时序3D DiT v14 开发记录（2026-10-04）

遇到的问题 → 解决办法 → 结果
1. 旧架构只输出2D，无法恢复深度。旧2D已留档，改为当前相机20×3 XYZ（米），内部预测17×20×3轨迹；训练直接监督3D、手指相对位置、速度、骨长，2D热图仅辅助。
2. 128维定位特征不足以保留3D语义。增加WiLoR原生192×1280空间条件，保留128维定位支路；投影、定位和3D头训练。最后4块视觉主干实际微调的v12对照相对误差更差，未采用。最终32层主干冻结。
3. 无标签GT占位混入扩散latent。清零无标签残差并屏蔽缺帧注意力，扰动无效标签/缺帧后影响为0；错误试验留档，模型重新训练。
4. 去噪训练与部署多步采样不同。追加实际10步、4采样闭环训练、准确点保护；v13密帧DiT选中step300，全部1800步完成。只用dev_select选权重、dev_calibrate定运行策略。
5. 原来最近帧仅1/6秒，速度损失还截断到50毫秒。准备86个训练/开发片段的真实30FPS；3417个原始中心XYZ/RGB/标签逐字一致，两臂只改变邻帧密度，风险头重新按排除自身受试者方式训练；速度使用真实时间差，数值下限1毫秒。密帧比稀帧的开发相对CI支持小幅改善，但相机误差略差，不夸大成整体提案更好。
6. 时序线性先验在旧严重8窗有帮助、整体较差。保持全部RGB输入的0/0.5/1先验对照都完成真实采样阶段；先验增加原始提案整体误差，未采用。风险判断器多次失败记录仍保留。
7. 固定9.5毫米界限制可恢复幅度。基于基线相机/相对错误风险设置逐点9.5或50毫米半径，在共同手腕下联合求绝对与相对约束；开发集筛选同时要求两类准确点损伤≤1%及相机收益。先封存权重/策略/代码，再评估第四批12未用片段，未用该批重调参数。
8. 第四批{fresh['windows']}窗/{fresh['points']}非手腕点：相对{b['relative_mm']:.2f}→{f['relative_mm']:.2f}毫米（{100*(1-f['relative_mm']/b['relative_mm']):.1f}%），相机{b['camera_mm']:.2f}→{f['camera_mm']:.2f}；坏相对点恢复{f['relative_bad_recovered20']}/{f['relative_bad_points']}。两类准确点损伤{f['camera_good_harmed20']}/{f['camera_good_points']}、{f['relative_good_harmed20']}/{f['relative_good_points']}。来源序列配对CI支持改善，困难{g['windows']}窗相对{g['baseline']['relative_mm']:.2f}→{g['final']['relative_mm']:.2f}毫米，恢复{g['final']['relative_bad_recovered20']}坏点。
9. 原始RGB入口与新缓存的风险分数不一致。统一BF16投影及16帧批处理，核验时刻、坐标、特征、风险分数、实际采样XYZ；最大3D差{checks['sampler_xyz_max_difference_mm']:.4f}毫米。34个确认点锁定、GT字段投毒、真缺XYZ保留null及已有坐标、256个极端提案双约束检查通过。带确认点整手使用保守界，缺XYZ仅给复核候选。
10. 47个旧自然失败保留中心不变，重新准备真实密帧。严重8窗相对{sv['baseline']['relative_mm']:.2f}→{sv['final']['relative_mm']:.2f}毫米，旧v11{sv['v11_same_center']['relative_mm']:.2f}，原始候选{sv['raw']['relative_mm']:.2f}，恢复{sv['final']['relative_bad_recovered20']}坏点；仍有严重失败，不能宣称完全遮挡精确补全。A附近帧有线索与B片段持续缺线索的图证、逐例结果保留。

适用范围：已有手框/轨迹的辅助3D标注，未确认点需要复核。未解决整手漏框。有限3D标注不是逐指不可见真值；第四批受试者/来源序列已出现，上游预训练重叠未知，OOF只覆盖风险/提案头。较大修正的保护是实测结果，非每点正确保证。四采样差异不是校准置信度。相同条件开发集回归略优于DiT，不宣称DiT有显著优势。旧2D、v7–v13和失败试验均保留。
'''
    (out/'DEVELOPMENT_NOTES.txt').write_text(notes,encoding='utf-8')
    labels=['WiLoR','Conservative','v14 adaptive'];values=[b,fresh['fallback'],f]
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,key,title in zip(axes,['camera_mm','relative_mm'],['Camera MPJPE19','Wrist-relative MPJPE19']):
        vals=[v[key] for v in values];ax.bar(labels,vals,color=['#8e9eb0','#76a9b6','#326fa7']);ax.set_title(title);ax.set_ylabel('mm; lower is better');ax.set_ylim(0,max(vals)*1.2)
        for j,v in enumerate(vals):ax.text(j,v+.6,f'{v:.2f}',ha='center')
    fig.savefig(out/'metrics.png',dpi=140);plt.close(fig)
    style='body{font:16px/1.7 system-ui;max-width:1160px;margin:32px auto;padding:0 22px;background:#f3f6fa;color:#18324d}section{background:white;padding:24px;border-radius:12px;margin:22px 0}table{border-collapse:collapse;width:100%}td,th{padding:9px;text-align:left;border-bottom:1px solid #dce4ee}img{max-width:100%}.ok{padding:16px;background:#ddf2e7}.warn{padding:16px;background:#fff0d8}a{color:#226eb5}'
    h=f'<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>HOT3D 3D DiT v14</title><style>{style}</style><h1>HOT3D 时序3D DiT v14</h1><p class="ok">第四批未用于选择的片段通过既定恢复与保护验收。输出相机20×3 XYZ，真实30FPS近邻、双向RGB。用于辅助标注，全部未确认点需复核。</p><section><h2>独立片段验证</h2><p>12未用片段，6来源序列，{fresh["windows"]}窗，{fresh["points"]}个非手腕点。受试者P0010/P0015和来源序列已出现，不是新受试者泛化验证。模型、代码、风险头、温度、参数和清单均先封存。</p>'
    h+=table([('WiLoR',b),('同一DiT保守9.5 mm',fresh['fallback']),('v14自适应联合约束',f),('完整DiT候选，仅复核',fresh['raw'])])
    h+=f'<img src="metrics.png"><p>相对改善{100*(1-f["relative_mm"]/b["relative_mm"]):.1f}%；来源序列配对95%CI：相对变化{ci["relative"]["ci95_delta_mm"]} mm，绝对变化{ci["camera"]["ci95_delta_mm"]} mm。准确点定义原误差≤10 mm、损伤定义改后&gt;20 mm；恢复定义原&gt;20 mm、改后≤20 mm。</p><p>困难组（基线平均相对&gt;40 mm）{g["windows"]}窗：{g["baseline"]["relative_mm"]:.2f}→{g["final"]["relative_mm"]:.2f} mm，{g["final"]["relative_bad_recovered20"]}/{g["final"]["relative_bad_points"]}坏相对点恢复。大多数仍不精确；比保守版平均误差更低，但困难组跨过20 mm的恢复数更少（保守版{g["fallback"]["relative_bad_recovered20"]}），两种证据都保留。</p></section>'
    h+='<section><h2>结构与部署</h2><p>YOLO26手框 → WiLoR相机XYZ → 每帧原生192×1280空间RGB＋128维定位支路 → 相机外参对齐17帧 → 根/相对手指token＋带射线与时间的位置编码 → 4层、宽192、6头Transformer → 10步4次diffusion采样 → 内部17×20×3、当前20×3米。</p><p>主干冻结，RGB投影、定位、3D头已训练。实际最近1/30秒、远处至±4秒；片段边界或轨迹断裂的帧保留缺失，禁止重复填充。相机风险阈值0.95、相对风险阈值0.65；半径9.5/50毫米，候选强度0.5，根与手指共同约束。可靠性表示基线误差风险，不表示逐指可见性或改后正确率。</p><p>人工确认点原样保留，带确认点使用保守界；真缺XYZ保留null与已有坐标，仅另给候选。当前服务器入口 <code>infer_hand3d_v14.py --input tracks.json --output proposals.json --device cuda:2</code>。<a href="README.txt">使用说明</a> · <a href="DEVELOPMENT_NOTES.txt">简明开发记录</a> · <a href="raw_inference_checks.json">入口核验</a> · <a href="fourth_seal.json">封存记录</a></p></section>'
    h+=f'<section><h2>保留的自然严重失败</h2><p class="warn">旧严重8窗相对{sv["baseline"]["relative_mm"]:.2f}→{sv["final"]["relative_mm"]:.2f} mm；同中心旧v11为{sv["v11_same_center"]["relative_mm"]:.2f} mm。完整候选{sv["raw"]["relative_mm"]:.2f} mm。旧例已看过，仅作诊断。不能声称绝对信息缺失已经准确恢复。</p><p>A：附近帧有可用线索；B：片段持续缺乏直接指节线索。<a href="temporal_clue_audit/report.html">原始时序图证</a> · <a href="hard_results.json">完整47例结果</a></p>'
    for c in severe:
        h+=f'<h3>#{c["id"]} · {c["category"]}</h3><p>{html.escape(c["reason"])} 相对 {c["baseline"]["relative_mm"]:.2f}→{c["final"]["relative_mm"]:.2f} mm，候选{c["raw"]["relative_mm"]:.2f} mm；恢复{c["final"]["relative_bad_recovered20"]}点。有效上下文{c["valid_context_count"]}/17。</p><img loading="lazy" src="{c["image"]}">'
    h+='</section><section><h2>新片段恢复与失败示例</h2><p>按收益最大、残余误差最大和退步最大各取4例，去重后展示；这是评估后的说明性挖掘，没有据此改模型。绿色GT只作评估，2D投影看不出深度误差。</p>'
    for c in examples:h+=f'<h3>窗口{c["index"]} · {html.escape(c["row"]["sequence"])}/clip{c["row"]["clip"]}/f{c["row"]["frame"]}</h3><p>手指相对{c["baseline_relative_mm"]:.2f}→{c["final_relative_mm"]:.2f} mm。</p><img loading="lazy" src="{c["image"]}">'
    h+='</section><section><h2>失败对照与限制</h2><p>视觉最后4块微调、RGB时序锚点、旧与原生判断器均有失败记录，不采用。密帧相同条件的开发集直接3D回归21.07 mm、DiT21.12 mm，差值CI跨0；没有证据DiT显著优于回归。当前收益包含时序RGB与安全修正，不能全部归因于diffusion。</p><p>未解决整手漏框；真缺XYZ候选缺少准确性验证；逐指不可见真值未提供；预训练重叠未知。保护较好不等于每次修正有益，四次采样的一致性不等于真值唯一可恢复。</p></section></html>'
    (out/'report.html').write_text(h,encoding='utf-8')
    readme='''v14 用于已有手框/轨迹的离线辅助3D标注；输出HOT3D20 XYZ、手腕索引5、当前相机坐标、米。
入口：/mnt/why/hot3d_hand_residual/infer_hand3d_v14.py --input tracks.json --output proposals.json --device cuda:2
Python：/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python
输入image_size=[1408,1408]、tracks[].frames[]，每帧含image、camera（含T_world_from_camera）、timestamp_s、box_xyxy、box_confidence、xyz_camera_m[20][3]，可含available_3d、confirmed_3d；track内时间递增。
提供原始实际30FPS轨迹；按真实时间选17个近密远疏观测，外侧±4秒，中心密33/67/100毫秒。边界缺帧不复制。
人工确认点锁定；含确认点的整手采用9.5mm保守界；真缺XYZ仅给复核候选，保留null和已有坐标。全部未确认点均需复核。candidate与4次采样是待核验假设。
当前模型：native_density_v13/dit_dense/best.pt；风险头aligned_density_v13/risk_dense/risk_all.pt；策略adaptive_projection_v14/dit_dense/fourth_seal.json。
sealed/为权重与校准快照，code/为源码快照，evidence/为对照训练记录。入口仍依赖服务器既有WiLoR权重、MANO资产及相机toolkit，本地目录不是独立Windows应用。
report.html包含第四批封存结果、全部47例和严重8例图证；DEVELOPMENT_NOTES.txt简述问题、处理与失败。
'''
    (out/'README.txt').write_text(readme,encoding='utf-8')
    pointer=dict(version='native3d_dense_adaptive_v14',output='20x3 current-camera XYZ meters',internal='17x20x3 trajectory',native_condition_channels=1280,spatial_cells=192,real_near_interval_s=1/30,automatic_projection_approved=True,full_candidate_auto_approved=False,severe_occlusion_resolved=False,locked_acceptance_passed=True,raw_adapter=str(CODE/'infer_hand3d_v14.py'),run=str(RUN),report=str(out/'report.html'),scope='Reviewed offline annotation with existing hand boxes/tracks; encountered subjects/source sequences')
    save(out/'CURRENT_PIPELINE.json',pointer)
    save(out/'manifest.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'})
    archive=RUN/'offline_hand3d_v14_delivery.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(out,arcname='offline_hand3d_v14')
    save(RUN/'delivery_receipt.json',dict(complete=True,files=len(json.loads((out/'manifest.json').read_text())),archive=str(archive),archive_sha256=sha(archive)))
    print((RUN/'delivery_receipt.json').read_text(),flush=True)

if __name__=='__main__':main()
