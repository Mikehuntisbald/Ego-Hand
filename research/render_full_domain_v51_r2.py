"""Render actual full model outputs, keeping evidence scopes explicit."""
import json,html,collections
from pathlib import Path
import cv2,numpy as np
from cache_instance_conditions_v51 import RUN

EDGES=[(6,7),(7,0),(5,8),(8,9),(9,10),(10,1),(5,11),(11,12),(12,13),(13,2),(5,14),(14,15),(15,16),(16,3),(5,17),(17,18),(18,19),(19,4)]
COLORS=[(60,210,255),(255,170,80),(90,220,120),(220,80,220),(110,100,250)]

def main():
    folder=RUN/'natural_nail_care_video_memory_r3';out=RUN/'review';out.mkdir(exist_ok=True)
    prediction=json.loads((folder/'prediction.json').read_text());source=json.loads((folder/'input_tracks.json').read_text())
    lookup={};frames={}
    for i,tr in enumerate(source['tracks']):
        for f in tr['frames']:lookup[tr['id'],round(f['timestamp_s'],6)]=f
    for i,tr in enumerate(prediction['tracks']):
        for f in tr['frames']:frames.setdefault(round(f['timestamp_s'],6),[]).append((i,tr,f))
    images=json.loads((RUN/'natural_nail_care/input.json').read_text())['frames']
    writer=cv2.VideoWriter(str(out/'nail_care_full_model.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),29.970, (1024,1024))
    bone=[];motion=[]
    for tr in prediction['tracks']:
        xyz=np.asarray([f['candidate_xyz_camera_m'] for f in tr['frames']]);times=np.asarray([f['timestamp_s'] for f in tr['frames']])
        lengths=np.stack([np.linalg.norm(xyz[:,a]-xyz[:,b],axis=-1) for a,b in EDGES[1:]],-1)
        bone.append(dict(id=tr['id'],frames=len(xyz),max_edge_length_CV=float((lengths.std(0)/np.maximum(lengths.mean(0),1e-9)).max()),scope='Observed decoded landmark edge variation; not a physical bone measurement or correct-pose proof'))
        if len(times)>1:
            jumps=np.linalg.norm(np.diff(xyz,axis=0),axis=-1)
            motion.append(dict(id=tr['id'],max_observed_joint_step_m=float(jumps.max()),p95_observed_joint_step_m=float(np.quantile(jumps,.95)),fast_motion_accuracy_GT_available=False,constraints=tr['constraints']))
    selected=set(np.linspace(0,len(images)-1,6).round().astype(int).tolist())
    for index,original in enumerate(images):
        t=round(original['timestamp_s'],6);present=frames.get(t,[])
        image=cv2.imread(str(folder/'normalized'/f'{index:06d}.jpg'));assert image is not None
        for i,tr,f in present:
            inp=lookup[tr['id'],t];color=COLORS[i%len(COLORS)];box=np.asarray(inp['box_xyxy']).round().astype(int)
            cv2.rectangle(image,tuple(box[:2]),tuple(box[2:]),color,3)
            xyz=np.asarray(f['candidate_xyz_camera_m']);fx,fy,cx,cy=inp['camera']['calibration']['projection_params'][:4]
            uv=xyz[:,:2]/np.maximum(xyz[:,2:],1e-6)*[fx,fy]+[cx,cy]
            for a,b in EDGES:
                if np.isfinite(uv[[a,b]]).all() and (uv[[a,b]]>=0).all() and (uv[[a,b]]<1408).all():cv2.line(image,tuple(uv[a].round().astype(int)),tuple(uv[b].round().astype(int)),color,3,cv2.LINE_AA)
            for xy in uv:
                if np.isfinite(xy).all() and (xy>=0).all() and (xy<1408).all():cv2.circle(image,tuple(xy.round().astype(int)),4,color,-1)
            cv2.putText(image,tr['id']+' 3D estimate',(max(0,box[0]),max(24,box[1]-8)),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
        cv2.putText(image,f'{original["timestamp_s"]:.3f}s | uncalibrated 3D | review required',(24,1370),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2)
        writer.write(cv2.resize(image,(1024,1024)))
        if index in selected:cv2.imwrite(str(out/f'nail_case_{index:03d}.jpg'),cv2.resize(image,(1000,1000)))
    writer.release()
    (out/'video_checks.json').write_text(json.dumps(dict(constraints_passed=prediction['constraint_checks_passed'],tracks=len(prediction['tracks']),observed_lengths=[len(t['frames']) for t in prediction['tracks']],association=prediction['association_summary'],bone_consistency=bone,motion=motion,GT_3D_available=False,person_identity_GT_available=False,fast_motion_retention_accuracy_unverified=True,uncalibrated_camera=True),indent=2))
    detector=json.loads((RUN/'paired_protocol/heldout_detector/evaluation.json').read_text());pose=json.loads((RUN/'surgical_pose/evaluation.json').read_text())
    refinement=json.loads((RUN/'paired_protocol/localizer_refinement_r1/core_r1/dit_joint/done.json').read_text())
    rows=''
    for dataset in ['egohands','cppe5','surgical_hands']:
        old=detector['statistics']['reference'][dataset];new=detector['statistics']['candidate'][dataset];ci=detector['paired_intervals'][dataset]
        rows+=f'<tr><td>{dataset}</td><td>{new["images"]}</td><td>{new["hands"]}</td><td>{old["matched"]}/{old["hands"]} ({old["box_coverage_IoU50"]:.1%})</td><td>{new["matched"]}/{new["hands"]} ({new["box_coverage_IoU50"]:.1%})</td><td>{ci["groups"]}</td><td>{ci["paired_coverage_CI95"][0]:+.1%} ～ {ci["paired_coverage_CI95"][1]:+.1%}</td></tr>'
    panels=''.join(f'<figure><img src="{p.name}"><figcaption>{p.name}: 模型估计，非真值</figcaption></figure>' for p in sorted(out.glob('nail_case_*.jpg')))
    checks=json.loads((out/'video_checks.json').read_text())
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>v51 完整模型：真实手套与多人多手</title><style>body{{font:16px system-ui;max-width:1180px;margin:32px auto;padding:0 20px;color:#243040}}h1,h2{{color:#172b4d}}p{{line-height:1.75}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d3dce8;padding:10px;text-align:left}}figure{{margin:20px 0}}img,video{{max-width:100%;border-radius:8px}}.note{{background:#fff2d5;padding:16px}}</style><h1>v51：完整模型的真实场景适配</h1><p>YOLO26 → SAM2 手实例 mask / 离线身份关联 → WiLoR → 强 RGB + mask 条件时序 DiT → 34 参数 / UmeTrack FK → 整段稳定化。输出仍为 3D；2D 人工标签只监督投影与 RGB 定位头。非因果 ±1.6 秒，速度不变、软硬加速度 ×2，无人工遮挡。</p><p class="note">已证实的主要提升在漏手检测。遮挡指尖的 3D 准确性、跨人物身份以及真实美甲泛化尚未证明。默认 v42 没有替换。</p><h2>独立留出检测：冻结输出后评分</h2><table><tr><th>数据</th><th>图像</th><th>标注手实例</th><th>原完整流程前端</th><th>适配前端</th><th>独立组</th><th>组配对覆盖差 95% CI</th></tr>{rows}</table><p>IoU ≥0.5，一对一最大匹配。EgoHands 按同步交互对划分；手术数据按源视频划分；CPPE 为原官方测试。CPPE 仅标注手套，不能把未标注裸手算成误检。EgoHands 精确率 {detector['statistics']['reference']['egohands']['precision_IoU50']:.1%} → {detector['statistics']['candidate']['egohands']['precision_IoU50']:.1%}。这是完整流程的检测阶段结果。</p><h2>手套指尖：条件姿态诊断</h2><p>{pose['instances']} 手实例 / {pose['groups']} 源视频组，{pose['candidate']['all_labelled']['points']} 人工 2D 点。人工框条件下，3D 投影 PCK10 {pose['reference']['all_labelled']['PCK10']:.1%} → {pose['candidate']['all_labelled']['PCK10']:.1%}；标注为遮挡点 {pose['reference']['annotated_occluded']['PCK10']:.1%} → {pose['candidate']['annotated_occluded']['PCK10']:.1%}。坏点恢复 {pose['bad_recovered_PCK10']}/{pose['bad_points']}，原正确点受损 {pose['correct_points_harmed_PCK10']}/{pose['correct_points']}。误差差值 95% CI {pose['source_video_paired_CI95']} 跨零，不能宣称明确提高。该实验没有 3D GT，不是完整检出后姿态准确率。</p><h2>真实护理：戴手套操作者与患者裸手重叠</h2><p>真实视频 148.022–151.992 秒、120 帧，原始 PTS；相机内参未测量，深度和尺度只供复核。当前用预测掌指关节点作种子，通过 SAM2 视频记忆前后传播，输出 {checks['tracks']} 条候选轨迹，长度 {checks['observed_lengths']}。低置信短轨迹及替代检测全部保留，见关联文件。硬约束通过={checks['constraints_passed']}；这只说明同一预测 ID 内的保存轨迹符合约束，不能证明未串手或指尖准确。快速动作真实幅度暂无 GT 验证。本片段已参与真实背景 presence 弱标签训练，只是场景适配回放。mask ID 固定不等于物理身份必然正确。当前素材是护理，不能等同于真实商业美甲。</p><video controls preload="metadata" src="nail_care_full_model.mp4"></video>{panels}<h2>完整流程独立留出测试</h2><p>12 张真实留出手术图 / 6 源视频组 / 31 标注手 / 420 标注点。真实检测框开始，漏手计入失败：PCK10 34.3% → 37.1%，组配对差 95% CI [-1.58,+10.10] 个百分点，跨零。恢复 20 个原失败点，但丢失 8/144 个原 PCK10 成功点。额外 ROI / 复核器版尚未达到全面替换标准，默认没有改变。</p><p><a href='complete_glove_test.json'>完整留出统计与正确点损伤</a> · <a href='spatial_solver.json'>强图像引导的受控参数采样/FK诊断</a></p><h2>失败与修复</h2><p>原试验 BF16/FP32 分支改变了零初始化输出：修复残差 dtype 后重新通过一致性检查。首次数据划分遗漏同步反视角：按同一交互重分组并从未见外部数据的权重重训。低分框产生大量短假轨迹：保留所有候选，先用前后帧接轨，再把短暂目标列为复核。加强 2D 监督的 240 步联训没有通过原正确 3D 点保护，选中步数 {refinement['selected_step']}，因此未接入其训练末权重。</p><p><a href="detector_test.json">检测留出统计</a> · <a href="pose_diagnostic.json">条件姿态统计</a> · <a href="video_checks.json">轨迹、骨长与约束</a> · <a href="association_summary.json">候选与身份关联</a> · <a href="../../DEVELOPMENT_V51.md">开发说明</a></p><p>来源：<a href="https://github.com/MichiganCOG/Surgical_Hands_RELEASE">Surgical Hands</a>、<a href="https://github.com/Rishit-dagli/CPPE-Dataset">CPPE-5</a>、<a href="https://commons.wikimedia.org/wiki/File:Nail_Care.webm">真实 Nail Care 视频</a>。人物关系没有独立 GT，美甲素材没有足够标注。</p></html>'''
    (out/'report.html').write_text(page,encoding='utf-8')
    for name,path in [('complete_glove_test.json',RUN/'complete_glove_test/evaluation_with_preservation.json'),('spatial_solver.json',RUN/'paired_protocol/full_spatial_solver_rgb35/evaluation.json')]:
        (out/name).write_text(path.read_text())
    for name,data in [('detector_test.json',detector),('pose_diagnostic.json',pose),('association_summary.json',prediction['association_summary'])]:(out/name).write_text(json.dumps(data,indent=2))
    print(json.dumps(dict(report=str(out/'report.html'),tracks=checks['tracks'],constraints=checks['constraints_passed'])),flush=True)

if __name__=='__main__':main()
