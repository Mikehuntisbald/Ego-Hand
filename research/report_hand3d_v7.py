import json,shutil,hashlib,tarfile
from pathlib import Path
import numpy as np,torch,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hand3d_data_v7 import RUN as BASE
from evaluate_hand3d_bounded_v7 import RUN,PARENT
from hand3d_temporal_v7 import WRIST,CHAINS
from hand3d_protected_v7 import ProtectedHand3D
from evaluate_offline_kp import bootstrap
import spatial_rgb_common as s

def main():
    O=RUN/'delivery';O.mkdir(exist_ok=True);result=json.loads((RUN/'test_results.json').read_text());data=torch.load(BASE/'data.pt',weights_only=False,mmap=True);rows=data['rows'];records,_=s.records_and_index()
    report=dict(result);report['raw_intervals']={};report['raw_rgb_comparisons']={}
    arrays={a:np.load(RUN/f'{a}_predictions.npz') for a in result['methods']}
    for arm,c in arrays.items():
        indices=c['indices'];gt=c['gt'];base=c['base'];pred=c['raw'];mask=c['valid'].copy();mask[:,WRIST]=False;clusters=np.array([rows[i]['sequence'] for i in indices]);camera=(np.linalg.norm(pred-gt,axis=-1)-np.linalg.norm(base-gt,axis=-1))*1000
        relative=(np.linalg.norm((pred-pred[:,WRIST:WRIST+1])-(gt-gt[:,WRIST:WRIST+1]),axis=-1)-np.linalg.norm((base-base[:,WRIST:WRIST+1])-(gt-gt[:,WRIST:WRIST+1]),axis=-1))*1000
        report['raw_intervals'][arm]=dict(camera_ci95_mm=bootstrap(camera,mask,clusters),relative_ci95_mm=bootstrap(relative,mask,clusters))
    for kind in ['regression','dit']:
        a=arrays['rgb_'+kind];b=arrays['tracks_'+kind];mask=a['valid'].copy();mask[:,WRIST]=False;clusters=np.array([rows[i]['sequence'] for i in a['indices']]);delta=(np.linalg.norm(a['raw']-a['gt'],axis=-1)-np.linalg.norm(b['raw']-b['gt'],axis=-1))*1000
        rel=(np.linalg.norm((a['raw']-a['raw'][:,WRIST:WRIST+1])-(a['gt']-a['gt'][:,WRIST:WRIST+1]),axis=-1)-np.linalg.norm((b['raw']-b['raw'][:,WRIST:WRIST+1])-(b['gt']-b['gt'][:,WRIST:WRIST+1]),axis=-1))*1000
        report['raw_rgb_comparisons'][kind]=dict(camera_delta_mm=float(delta[mask].mean()),camera_ci95_mm=bootstrap(delta,mask,clusters),relative_delta_mm=float(rel[mask].mean()),relative_ci95_mm=bootstrap(rel,mask,clusters))
    model=ProtectedHand3D();report['temporal_head_parameters']=sum(p.numel() for p in model.parameters())
    # An annotation candidate is selected by the predeclared development metric.
    # Automatic approval remains false; annotation candidate and auto output differ.
    cal=json.loads((PARENT/'selection.json').read_text())['calibration']
    annotation_arm=min(['rgb_regression','rgb_dit'],key=lambda a:cal[a]['raw']['camera_mm']+.5*cal[a]['raw']['relative_mm'])
    selection=json.loads((RUN/'selection.json').read_text());selection['annotation_arm']=annotation_arm;selection['annotation_selected_on']='dev_calibrate raw camera +0.5 relative metric; automatic gate stays unapproved'
    for where in [RUN/'selection.json',RUN/'sealed/selection.json']:where.write_text(json.dumps(selection,indent=2))
    manifest=json.loads((RUN/'sealed/manifest.json').read_text());manifest['selection.json']=hashlib.sha256((RUN/'sealed/selection.json').read_bytes()).hexdigest();(RUN/'sealed/manifest.json').write_text(json.dumps(manifest,indent=2))
    report['annotation_arm']=annotation_arm
    (O/'test_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    seal=O/'sealed';shutil.copytree(RUN/'sealed',seal,dirs_exist_ok=True)
    code=O/'code';code.mkdir(exist_ok=True)
    names=['hand3d_temporal_v7.py','hand3d_data_v7.py','hand3d_risk_v7.py','train_hand3d_v7.py','check_hand3d_v7.py','run_hand3d_v7.py',
        'hand3d_protected_v7.py','train_hand3d_protected_v7.py','run_hand3d_protected_v7.py','evaluate_hand3d_v7.py','evaluate_hand3d_protected_v7.py','evaluate_hand3d_bounded_v7.py','infer_hand3d_v7.py','make_hand3d_example_v7.py','report_hand3d_v7.py']
    for name in names:shutil.copy2(Path(__file__).parent/name,code/name)
    for name in ['pose_residual_dit.py','spatial_rgb_common.py','spatial_rgb_model.py','wilor_eval_common.py','offline_rgb_encoder.py','cache_dit_v3.py','infer_natural_reliability.py','infer_sampling_v6.py','temporal_sampling_v6.py','natural_corrector.py','spatial_temporal_model.py','offline_kp_model.py','natural_reliability.py','natural_policy.py']:
        shutil.copy2(Path(__file__).parent/name,code/name)
    for phase,root in [('initial_3d',BASE),('protected_3d',PARENT)]:
        dest=O/'training'/phase;dest.mkdir(parents=True,exist_ok=True)
        for name in ['protocol.json','preflight.json','data_checks.json','training_status.json','selection.json']:
            if (root/name).exists():shutil.copy2(root/name,dest/name)
        for arm in result['methods']:
            (dest/arm).mkdir(exist_ok=True)
            for name in ['config.json','history.json','training_done.json']:shutil.copy2(root/arm/name,dest/arm/name)
    for name in ['protocol.json','selection.json','policies.json','raw_inference_checks.json','example_input.json','example_output.json','example_dit_output.json']:shutil.copy2(RUN/name,O/name)
    archive=BASE.parent/'archive_2d_before_3d_v7_20261003';shutil.copy2(archive/'archive_receipt.json',O/'two_dimensional_archive_receipt.json')
    hard=json.loads((BASE.parent/'natural_reliability_v4/delivery/hardcase_review_queue.json').read_text());severe=[q for q in hard if q['after_px']>40]
    candidate=arrays[annotation_arm];lookup={v:i for i,v in enumerate(candidate['indices'])};examples=[]
    for q in severe:
        j=lookup[q['window_index']];row=rows[q['window_index']];fid=int(data['feature_ids'][q['window_index'],8]);rec=records[fid-1];cam=s.common.from_json(rec['camera'])
        base=candidate['base'][j];pred=candidate['raw'][j];gt=candidate['gt'][j];image=cv2.imread(rec['image'])[:,:,::-1]
        # Review-only crop/labels do not enter inference. Detector ROI, not GT ROI.
        roi=data['roi'][fid].numpy()*1408;lo=np.maximum(np.floor(roi[:2]).astype(int),0);hi=np.minimum(np.ceil(roi[2:]).astype(int),[1408,1408]);crop=image[lo[1]:hi[1],lo[0]:hi[0]]
        if crop.size==0:crop=image;lo=np.array([0,0])
        fig=plt.figure(figsize=(11,4.8),layout='constrained');ax=fig.add_subplot(121)
        ax.imshow(crop);ax.axis('off')
        for xyz,color,label in [(gt,'#4fb05b','GT'),(base,'#f49733','WiLoR'),(pred,'#3091e5','3D candidate')]:
            uv=cam.eye_to_window(xyz)-lo
            for chain in CHAINS:
                points=uv[chain];ax.plot(points[:,0],points[:,1],color=color,lw=1.4,alpha=.8)
            ax.scatter(uv[:,0],uv[:,1],s=7,c=color,label=label)
        ax.set_xlim(0,crop.shape[1]);ax.set_ylim(crop.shape[0],0);ax.set_title('Projection of actual XYZ');ax.legend(fontsize=8,loc='lower left')
        ax3=fig.add_subplot(122,projection='3d')
        for xyz,color,label in [(gt,'#4fb05b','GT'),(base,'#f49733','WiLoR'),(pred,'#3091e5','3D candidate')]:
            pose=(xyz-xyz[WRIST])*1000
            for chain in CHAINS:ax3.plot(pose[chain,0],pose[chain,1],pose[chain,2],c=color,lw=1.5)
        ax3.set_xlabel('X mm');ax3.set_ylabel('Y mm');ax3.set_zlabel('Z mm');ax3.set_title('Wrist-relative 3D pose');ax3.set_box_aspect([1,1,1])
        mask=np.ones(20,bool);mask[WRIST]=False
        cm=float(np.linalg.norm(pred-gt,axis=-1)[mask].mean()*1000);rm=float(np.linalg.norm((pred-pred[WRIST])-(gt-gt[WRIST]),axis=-1)[mask].mean()*1000)
        fig.suptitle(f"Case {q['id']} | true 3D candidate, not automatically approved | camera {cm:.1f} mm / relative {rm:.1f} mm",fontsize=11);fig.savefig(O/f"case_{q['id']}_3d.png",dpi=130);plt.close(fig)
        examples.append(dict(id=q['id'],camera_mm=cm,relative_mm=rm))
    (O/'example_metrics.json').write_text(json.dumps(examples,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');labels=['WiLoR','3D regression','3D DiT'];values=[result['baseline'],result['methods']['rgb_regression']['raw'],result['methods']['rgb_dit']['raw']]
    for ax,key,title in zip(axes,['camera_mm','relative_mm'],['Camera-space MPJPE19','Wrist-relative MPJPE19']):
        ax.bar(labels,[v[key] for v in values],color=['#7a899e','#2d8c86','#4d72aa']);ax.set_ylabel('mm, lower is better');ax.set_title(title);ax.set_ylim(0,45)
    fig.savefig(O/'metrics.png',dpi=150);plt.close(fig)
    architecture='''<svg viewBox="0 0 1050 470" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="3D时序网络结构"><defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3z" fill="#607187"/></marker></defs><g fill="#e9f1f8" stroke="#8ba3bb"><rect x="10" y="20" width="225" height="75" rx="10"/><rect x="290" y="20" width="310" height="75" rx="10"/><rect x="10" y="135" width="225" height="75" rx="10"/><rect x="290" y="135" width="310" height="75" rx="10"/><rect x="680" y="60" width="340" height="150" rx="10"/><rect x="680" y="265" width="340" height="70" rx="10"/><rect x="680" y="380" width="340" height="70" rx="10"/></g><g font-family="system-ui" font-size="17" fill="#172333" text-anchor="middle"><text x="123" y="50">RGB 手框裁剪</text><text x="123" y="78">WiLoR ViT：32层</text><text x="445" y="50">16×12×1280 → 16×12×128</text><text x="445" y="78">192个空间位置全部保留</text><text x="123" y="165">逐帧 WiLoR XYZ</text><text x="123" y="193">相机位姿、时间差</text><text x="445" y="165">历史XYZ对齐到当前相机</text><text x="445" y="193">17帧近密远疏，双向</text><text x="850" y="95">空间视觉注意力＋3D时序编码</text><text x="850" y="130">4层Transformer，宽192，6头</text><text x="850" y="165">回归 / DiT：预测3D残差</text><text x="850" y="192">手腕平移＋相对手腕姿态</text><text x="850" y="295">提案信任门控</text><text x="850" y="321">20×3 XYZ，单位米</text><text x="850" y="408">3D标注候选：需要复核</text><text x="850" y="435">自动覆盖验收：未通过</text></g><g stroke="#607187" stroke-width="2" marker-end="url(#arrow)" fill="none"><path d="M235 57 H285"/><path d="M235 172 H285"/><path d="M600 57 H640 V95 H675"/><path d="M600 172 H675"/><path d="M850 210 V260"/><path d="M850 335 V375"/></g></svg>'''
    h='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>3D时序补全 v7</title><style>body{font:16px/1.7 system-ui;max-width:1160px;margin:36px auto;padding:0 24px;background:#f5f7fa;color:#172333}section{background:white;border-radius:12px;padding:24px;margin:24px 0}h1{font-size:29px}img,svg{max-width:100%}table{border-collapse:collapse;width:100%}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:left}.note{color:#536174}.status{background:#fff0dc;border-left:4px solid #b77818;padding:14px}a{color:#1269aa}</style><h1>3D时序补全 v7：结构已修正，训练已完成</h1><p class="status"><b>交付的是3D标注候选；自动覆盖验收尚未通过。</b>上一分支只输出2D，不能据此判断3D补全上限。现已留档，新的回归和DiT直接预测真实XYZ，不通过2D坐标补深度。</p>'''+architecture+'''<section><h2>真实3D候选的测试结果</h2><p>2082个窗口、39558个非手腕关节；误差单位mm。下面是模型候选，未经自动门控拒绝/保留原值，不是自动覆盖后的结果。</p><table><tr><th>模型</th><th>绝对相机误差↓</th><th>手指相对误差↓</th><th>手腕误差↓</th><th>相对PCK20↑</th></tr>'''
    for label,v in zip(labels,values):h+=f"<tr><td>{label}</td><td>{v['camera_mm']:.2f}</td><td>{v['relative_mm']:.2f}</td><td>{v['wrist_mm']:.2f}</td><td>{v['pck20_relative']*100:.1f}%</td></tr>"
    h+='</table><img src="metrics.png"><p>两阶段分别完成6000步3D初训和4000步保护训练，检查点按开发集选择，并非直接取最后一步。视觉主干保持冻结，3D时序头、热图解码器和提案信任网络参与训练。</p>'
    for arm in ['rgb_regression','rgb_dit']:
        ci=report['raw_intervals'][arm];h+=f"<p>{arm} 相对WiLoR的误差变化95%CI：相机 [{ci['camera_ci95_mm'][0]:+.2f}, {ci['camera_ci95_mm'][1]:+.2f}] mm；手指相对 [{ci['relative_ci95_mm'][0]:+.2f}, {ci['relative_ci95_mm'][1]:+.2f}] mm。按来源序列配对bootstrap，负数表示改善。</p>"
    h+='</section><section><h2>为什么还没有批准自动覆盖</h2><table><tr><th>候选</th><th>原本准确的绝对点改坏</th><th>原本准确的相对点改坏</th><th>相对坏点恢复至20mm内</th></tr>'
    for arm in ['rgb_regression','rgb_dit']:
        r=result['methods'][arm]['raw'];h+=f"<tr><td>{arm}</td><td>{r['camera_good_harmed20']}/{r['camera_good_points']} ({r['camera_good_harm_rate']*100:.1f}%)</td><td>{r['relative_good_harmed20']}/{r['relative_good_points']} ({r['relative_good_harm_rate']*100:.2f}%)</td><td>{r['relative_bad_recovered20']}/{r['relative_bad_points']}</td></tr>"
    h+='''</table><p>“原本准确”指WiLoR误差≤10mm，“改坏”指候选误差&gt;20mm。相对姿态的收益没有消除绝对位置的风险，尤其是手腕修正带动全手移动。</p><p>尝试了错误风险筛选、提案信任训练，以及根平移/相对姿态各≤5mm的有界修正。后者能限制单点绝对位移≤10mm，但开发校准集仍未满足声明的恢复门槛：坏绝对点平均误差至少减少5%、整体至少改善0.2mm，同时保护准确点。保护本身不计作恢复成功。</p><p>推理文件分别返回 <code>candidate_xyz_camera_m</code>（实际3D网络候选）和 <code>xyz_camera_m</code>（自动策略输出）。当前自动策略为review_only，后者保留原始WiLoR XYZ。不能把它当成模型已经完成可靠修正。</p></section><section><h2>RGB对照</h2>'''
    for kind,c in report['raw_rgb_comparisons'].items():h+=f"<p>{kind}：RGB候选相对坐标-only候选，相机误差变化 {c['camera_delta_mm']:+.2f}mm，95%CI [{c['camera_ci95_mm'][0]:+.2f}, {c['camera_ci95_mm'][1]:+.2f}]；手指相对变化 {c['relative_delta_mm']:+.2f}mm，95%CI [{c['relative_ci95_mm'][0]:+.2f}, {c['relative_ci95_mm'][1]:+.2f}]。</p>"
    h+='</section><section><h2>严重案例：真实3D候选与GT</h2><p>沿用先前8个严重2D失败窗口，作为诊断图，不是重新选择的3D测试集。绿色GT仅用于评估/展示，橙色WiLoR，蓝色3D候选。左图投影真实XYZ，右图画相对手腕的XYZ；自动输出仍保留原值。</p>'
    for e in examples:h+=f"<details><summary>案例 #{e['id']}：相机 {e['camera_mm']:.1f}mm / 手指相对 {e['relative_mm']:.1f}mm</summary><img loading='lazy' src='case_{e['id']}_3d.png'></details>"
    h+='''</section><section><h2>实现与验证</h2><p>手腕索引5，HOT3D的20关节顺序。3D残差拆为手腕平移和相对手腕姿态；历史XYZ利用相机外参对齐到当前相机。17个时间位置使用近密远疏，最近约0.167秒、最远±4秒；不存在的同轨迹帧保持缺失。没有人工遮挡。</p><p>输出头实际宽192、4层、6头；DiT使用100步训练噪声日程，10步推理、4次采样均值。相机位置、相对姿态、骨长与保护损失均在3D监督；2D热图只是RGB定位辅助损失。输入不读取GT关节、GT侧别、GT形状或可见性标签。</p><p>相机变换往返误差约0.00024mm。前向/反向、20×3输出、确认点锁定通过；17帧真实RGB的回归与DiT CLI均跑通，34个确认点保持原值；添加伪造GT输入字段后结果完全一致。</p><p class="note">P0010/P0015序列此前已经查看，属于开发诊断验证，不是新的独立跨人泛化证据。WiLoR预训练是否覆盖HOT3D仍未知；没有逐指可见性真值。</p><a href="README.txt">推理使用说明</a> · <a href="DEVELOPMENT_NOTES.txt">简明开发记录</a> · <a href="test_results.json">完整指标</a> · <a href="two_dimensional_archive_receipt.json">2D留档校验</a></section></html>'''
    (O/'report.html').write_text(h,encoding='utf-8')
    (O/'DEVELOPMENT_NOTES.txt').write_text('3D时序补全 v7 开发记录\n\n1. 结构偏离：旧分支只有20x2残差，不能代表3D补全。处理：旧代码/权重/报告留档，新版直接输出20x3 XYZ。\n2. 相机运动混入手运动：不同帧XYZ坐标系不同。处理：用相机外参对齐至当前相机，往返误差0.00024mm。\n3. 根平移掩盖手指姿态：处理：独立手腕根token和相对姿态token，分别评估相机/相对/手腕误差。\n4. 候选改善均值却破坏准确点：处理：追加4000步提案信任与保护训练，保留初训失败结果；未放宽验收标准。\n5. 保守门控只能保护、恢复不足：处理：校准有界修正并单列结果；未通过自动恢复验收，不宣称训好或自动可用。\n6. 结果：模型3D候选的平均相对误差有改善，绝对准确点仍受损；作为人工标注候选交付。\n7. 验证：原始RGB、两种3D输出、34点锁定、GT字段投毒不改变输出。测试集已查看，不作新的泛化证明。\n',encoding='utf-8')
    readme=f'''3D temporal hand completion v7\n\nServer: ssh -p 11123 root@111.230.4.68\nCode: /mnt/why/hot3d_hand_residual/infer_hand3d_v7.py\nModels: {RUN}/sealed\nRuntime: /mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python\n\nCommand on server:\n/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python /mnt/why/hot3d_hand_residual/infer_hand3d_v7.py --input {RUN}/example_input.json --output {RUN}/prediction.json --kind dit --device cuda:0\n\nInput: original RGB path, predicted YOLO box/score, calibrated camera metadata, timestamps, native WiLoR xyz_camera_m (20x3 meters). Optional xy_px should be the original WiLoR projection, not the archived 2D corrector output. available_3d is coordinate presence; confirmed_3d is an explicit lock. Example locks are a test fixture, not human-validated GT.\n\nOutput: candidate_xyz_camera_m is the actual trained 3D model candidate. xyz_camera_m is the automatic policy result; current automatic policy did not pass and retains native XYZ. Every automatic candidate needs review. Neither output is 2D with fabricated depth.\n\nTwo-dimensional branch archive: {archive}/snapshot.tar.gz; SHA256 in two_dimensional_archive_receipt.json. Original 2D runs preserved.\n\nBest annotation candidate selected on development calibration only: {annotation_arm}. Native frozen WiLoR weights and 2D spatial stem dependencies remain on the server; checkpoint/source hashes are provided.\n'''
    (O/'README.txt').write_text(readme,encoding='utf-8')
    summary_file=dict(output='20x3 camera XYZ meters',annotation_candidate=annotation_arm,automatic_approved=False,run=str(RUN),report=str(O/'report.html'),old_2d_archived=str(archive))
    (O/'CURRENT_PIPELINE.json').write_text(json.dumps(summary_file,indent=2));(BASE.parent/'CURRENT_HAND_COMPLETION.json').write_text(json.dumps(summary_file,indent=2))
    (O/'manifest.json').write_text(json.dumps({str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in O.rglob('*') if p.is_file() and p.name!='manifest.json'},indent=2))
    tar=RUN/'offline_hand3d_v7_delivery.tar.gz'
    with tarfile.open(tar,'w:gz',compresslevel=1) as f:f.add(O,arcname='offline_hand3d_v7')
    (RUN/'delivery_receipt.json').write_text(json.dumps(dict(archive=str(tar),bytes=tar.stat().st_size,sha256=hashlib.sha256(tar.read_bytes()).hexdigest()),indent=2))
    print(json.dumps(dict(annotation_arm=annotation_arm,raw_intervals=report['raw_intervals'],raw_rgb=report['raw_rgb_comparisons'],parameters=report['temporal_head_parameters']),indent=2))

if __name__=='__main__':main()
