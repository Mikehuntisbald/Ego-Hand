"""Read-only result audit: no further model/rotation selection after test."""
import hashlib
import html
import json
from pathlib import Path
from wilor_eval_common import RUN,ROOT,from_json,SOURCE
import cv2
import numpy as np
from compare_detectors import save
from export_hand_labels import EDGES
from metrics_3d import pose_metrics,EVAL_INDICES

samples=json.loads((RUN/'samples.json').read_text())
result=json.loads((RUN/'results.json').read_text())
observations=json.loads((RUN/'observation_audit.json').read_text())
base=np.load(RUN/'coarse_predictions.npy')
pred=np.load(RUN/'wilor_predictions.npy')
ids=np.array([i for i,s in enumerate(samples) if s['split']=='test' and s['matched'] and s['right']>=0])
gt=np.array([samples[i]['gt'] for i in ids])
before=np.linalg.norm((base[ids]-base[ids,5:6])-(gt-gt[:,5:6]),axis=-1)[:,EVAL_INDICES].mean(1)*1000
after=np.linalg.norm((pred[ids]-pred[ids,5:6])-(gt-gt[:,5:6]),axis=-1)[:,EVAL_INDICES].mean(1)*1000

def metric(p,g):
    r=pose_metrics(p,g);r.pop('sample_mpjpe19_mm');return r

groups={}
for key in sorted({samples[i]['sequence'] for i in ids}):
    mask=np.array([samples[i]['sequence']==key for i in ids])
    groups[key]=dict(samples=int(mask.sum()),coarse=metric(base[ids][mask],gt[mask]),wilor=metric(pred[ids][mask],gt[mask]))
pipeline={}
for name,p in [('coarse',base),('wilor',pred)]:
    er=np.linalg.norm(p[ids]-gt,axis=-1)[:,EVAL_INDICES]*1000
    rel=np.linalg.norm((p[ids]-p[ids,5:6])-(gt-gt[:,5:6]),axis=-1)[:,EVAL_INDICES]*1000
    pipeline[name]=dict(camera_pck10_all_gt=float((er<=10).sum()/(observations['test']['gt_hands']*19)),
                         relative_pck10_all_gt=float((rel<=10).sum()/(observations['test']['gt_hands']*19)))
audit=dict(per_sequence=groups,paired_common_samples=len(ids),test_total_gt_hands=observations['test']['gt_hands'],
           detected_without_side=observations['test']['side_unavailable'],test_missed_detector=observations['test']['missed'],
           end_to_end_on_common_pipeline=pipeline,note='Missing detections/side assignments count as failures in all-GT PCK; coarse uses same common sample denominator here.')
diagnostics={}
for name,p in [('coarse',base[ids]),('wilor',pred[ids])]:
    # Post-hoc diagnosis separates articulation from wrist-reference and translation errors.
    palm=[8,11,14,17]
    pr=p-p[:,palm].mean(1,keepdims=True);gr=gt-gt[:,palm].mean(1,keepdims=True)
    oracle_offset=(gt[:,EVAL_INDICES]-p[:,EVAL_INDICES]).mean(1,keepdims=True)
    aligned=p+oracle_offset
    diagnostics[name]=dict(palm_mcp_relative_mpjpe19_mm=float(np.linalg.norm(pr-gr,axis=-1)[:,EVAL_INDICES].mean()*1000),
        gt_translation_aligned_mpjpe19_mm=float(np.linalg.norm(aligned-gt,axis=-1)[:,EVAL_INDICES].mean()*1000),
        wrist_disagreement_after_gt_translation_mm=float(np.linalg.norm(aligned[:,5]-gt[:,5],axis=-1).mean()*1000),
        note='Post-hoc diagnostic; GT translation alignment is not deployable accuracy and does not replace primary metrics.')
audit['post_hoc_reference_diagnostics']=diagnostics
save(RUN/'post_eval_audit.json',audit)

# Select representative outcome quantiles across clips; visualize true native fisheye projections.
order=np.argsort(after-before);chosen=[];clips=set()
for pos in [0,len(order)//4,len(order)//2,3*len(order)//4,len(order)-1]:
    for j in order[max(0,pos-10):min(len(order),pos+11)]:
        i=int(ids[j])
        if samples[i]['clip'] not in clips:
            chosen.append(i);clips.add(samples[i]['clip']);break
panels=[]
for i in chosen:
    s=samples[i];im=cv2.imread(s['image']);cam=from_json(s['camera']);target=np.array(s['gt'])
    b=np.asarray(s['box']);c=(b[:2]+b[2:])/2;side=max(b[2:]-b[:2])*1.8
    x0,y0=np.floor(c-side/2);scale=384/side
    affine=np.array([[scale,0,-x0*scale],[0,scale,-y0*scale]],np.float32)
    view=cv2.warpAffine(im,affine,(384,384))
    columns=[]
    for name,p in [('GT',target),('Old coarse',base[i]),('WiLoR',pred[i])]:
        image=view.copy()
        for source,color,width in [(target,(50,230,50),1),(p,(0,140,255),2)]:
            uv=cam.eye_to_window(source)
            uv=(uv-np.array([x0,y0]))*scale
            uv=np.clip(uv,-10000,10000).astype(int)
            for a,b in EDGES:
                if source[a,2]>0 and source[b,2]>0:cv2.line(image,tuple(uv[a]),tuple(uv[b]),color,width)
        cv2.rectangle(image,(0,0),(384,43),(20,20,20),-1)
        relative=float(np.linalg.norm((p-p[5])-(target-target[5]),axis=-1)[EVAL_INDICES].mean()*1000)
        cv2.putText(image,f'{name}: rel {relative:.1f}mm',(8,17),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        cv2.putText(image,f'{s["clip"]}:{s["frame"]} side={s["right"]}',(8,35),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        columns.append(image)
    panels.append(np.concatenate(columns,axis=1))
cv2.imwrite(str(RUN/'pose_examples.jpg'),np.concatenate(panels,axis=0))
save(RUN/'example_samples.json',[dict(index=i,sequence=samples[i]['sequence'],clip=samples[i]['clip'],frame=samples[i]['frame']) for i in chosen])

summary=dict(detector='Keep YOLO26 HOT3D checkpoint based on same-image measured comparison',
             wilor='No overall 3D improvement established by this frozen zero-shot adapter',
             dit='Do not start DiT training yet; stabilize calibrated 3D estimator and side prediction first',
             test=result['test'],observations=observations,reference_diagnostics=diagnostics,
             implementation='YOLO boxes + auxiliary WiLoR-detector side classification + calibrated perspective crop + full WiLoR reconstruction; no DiT',
             upstream_overlap='Released model config includes HOT3D-TRAIN; these are reused test subjects, upstream overlap not audited')
save(RUN/'summary.json',summary)
rows=[]
for key,label in [('all','全部'),('occlusion_lt_0_5','遮挡，可见比例<0.5'),('severe_lt_0_25','严重遮挡，可见比例<0.25'),('occlusion_in_view','遮挡且≥18点在视场内')]:
    g=result['test']['groups'][key]
    rows.append(f'<tr><td>{label}</td><td>{g["samples"]}</td><td>{g["coarse"]["mpjpe19_mm"]:.2f}</td><td>{g["wilor"]["mpjpe19_mm"]:.2f}</td><td>{g["coarse"]["wrist_relative_mpjpe19_mm"]:.2f}</td><td>{g["wilor"]["wrist_relative_mpjpe19_mm"]:.2f}</td></tr>')
markup='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>YOLO26 → WiLoR: HOT3D 3D evaluation</title>
<style>body{font-family:system-ui;max-width:1160px;margin:35px auto;padding:20px;color:#172334}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccd}img{max-width:100%}pre{white-space:pre-wrap;background:#f3f5f7;padding:16px}.note{padding:20px;background:#fff3df}</style>
<h1>YOLO26 手框 → WiLoR 3D 重建</h1>
<p class="note">当前零样本接入未建立整体 3D 优势，暂不进入 DiT 训练。检测器继续采用已经在相同图像上胜出的 YOLO26 HOT3D 权重。</p>
<p>720 张测试帧、1329 个有效 GT 手框；YOLO 匹配1307个，其中5个缺失预测左右手信息，共1302个共同样本用于3D对照。左右手预测准确率96.39%；主结果包括左右手判错的样本。GT侧别和GT姿态仅用于评分。</p>
<table><tr><th>分组</th><th>手数</th><th>旧模型相机/mm</th><th>WiLoR相机/mm</th><th>旧模型腕相对/mm</th><th>WiLoR腕相对/mm</th></tr>'''+''.join(rows)+'''</table>
<p>整体腕相对改善量为 -3.75 mm，按6条源序列配对bootstrap的95%区间为[-12.12, 4.72] mm（正数代表WiLoR更好）；未建立整体改善。全视场遮挡子集平均有改善，不能扩展为全部恶劣情况有效。</p>
<p>事后参考点诊断：以四个手指MCP关节平均位置作为掌心，旧模型/WiLoR的掌心相对误差为19.84/17.93 mm；仅使用GT平移对齐后的诊断误差为17.47/13.37 mm。这提示指形存在收益，但腕部参考点和全局平移误差掩盖了收益。GT对齐值不可作为部署精度，不替代主指标。下一步应专项核查腕部定义与绝对位置，再考虑残差补全。</p>
<p>测试集原本腕相对误差≤10 mm的1118个点中，95.26%在替换模型后误差增加超过1 mm。直接整体替换不能满足原有正确点保护目标。</p>
<p>接入：预测框和标定生成透视手部crop；逐点反投影闭环误差&lt;0.01px；使用官方canonical20映射；固定90°旋转由P0003的384个开发样本选择，测试前冻结。使用WiLoR原始完整网络FP32，未跳层，未训练DiT。附加WiLoR检测器仅提供左右手预测，未替换YOLO框。</p>
<p>限制：WiLoR发布配置包含HOT3D训练来源；上游重叠未审计。当前为已有数据上的部署对照。MANO与UmeTrack存在模型/解剖定义差异。绝对深度通过虚拟相机焦距与预测camera参数换算，仍需专项校准验证。</p>
<p>以下按误差差异分位选取示例；绿色为GT，橙色为预测，投影使用原始鱼眼标定。非随机总体展示。</p><img src="pose_examples.jpg" alt="Pose projections">
<p><a href="results.json">完整指标</a> · <a href="post_eval_audit.json">分序列与端到端指标</a> · <a href="protocol.json">评估协议</a> · <a href="selection.json">开发集选择</a> · <a href="provenance.json">权重及代码来源</a></p></html>'''
(RUN/'report.html').write_text(markup)

def hash_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(8<<20),b''):h.update(data)
    return h.hexdigest()
code=Path('/mnt/why/hot3d_hand_residual')
save(RUN/'provenance.json',dict(github_revision='fcb911312a38fa8badd30d9656a167485d61b8f9',
     huggingface_revision='99fe3d7acff8104ecca1055df7467709506c2fa6',
     checkpoint_sha256=hash_file(RUN/'assets/wilor_final.mirror.ckpt'),
     code={n:hash_file(code/n) for n in ['wilor_eval_common.py','prepare_wilor_eval.py','run_wilor_eval.py','check_wilor_geometry.py','report_wilor_eval.py']},
     renderer_disabled=True,precision='FP32',fast_skip_blocks=False,weights_finetuned=False,
     geometry_check='Canonical mapping, rotations and actual calibrated ray roundtrips passed',
     no_gt_used_in_inference=True,test_read_only_after_rotation_selection=True))
print(json.dumps(dict(report=str(RUN/'report.html'),reference_diagnostics=diagnostics,end_to_end=pipeline),indent=2))
