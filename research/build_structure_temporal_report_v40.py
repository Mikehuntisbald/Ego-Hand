"""Package actual development evidence; no model selection or training here."""
import json,hashlib,zipfile,html,collections
from pathlib import Path
import numpy as np,torch,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hand3d_v8_common import V7,save
from hand3d_rollout_v8 import project_fisheye624
from joint_mano_model_v29 import EDGES
from calibrate_hand3d_v8 import paired_ci

ROOT=V7.parent;OUT=ROOT/'structure_temporal_review_v40'
RUNS=['joint_mano_v28','joint_mano_v29','joint_kinematic_v30','parameter_kinematic_v31','parameter_pose_v32',
      'parameter_trajectory_v33','joint_feasibility_v34','joint_refinement_v35','semantic_parameter_v36',
      'semantic_parameter_v36_incorrect_token_mapping','semantic_parameter_trajectory_v37','semantic_joint_refinement_v37',
      'observation_ik_v38','fitted_parameter_v39','fitted_parameter_trajectory_v39','fitted_joint_refinement_v39',
      'parameter_dit_trajectory_v37','dit_joint_refinement_v37']


def read(path):return json.loads(path.read_text())


def training_result(run):
    done=read(run/'done.json');history=read(run/'history.json')
    if done['selected_step']==0:return read(run/'matched_start.json')
    return next(x['metrics'] for x in history if x['step']==done['selected_step'])


def compact(name,report,tracks=None):
    m=report['metrics'];c=report.get('coherence',{});hard=report.get('hard',{})
    return dict(name=name,camera_mm=m['camera_mm'],relative_mm=m['relative_mm'],
        camera_good_harmed=m['camera_good_harmed20'],camera_good_total=m['camera_good_points'],
        relative_good_harmed=m['relative_good_harmed20'],relative_good_total=m['relative_good_points'],
        recovered_relative=m['relative_bad_recovered20'],bad_relative_total=m['relative_bad_points'],
        hard_relative_mm=hard.get('relative_mm'),hard_recovered=hard.get('relative_bad_recovered20'),
        jump_pairs=c.get('spurious_jump_pairs'),bone_extreme_frames=c.get('bone_extreme_frames'),bone_flicker=c.get('bone_flicker_gt_under1_pred_over10'),tracks=tracks)


def illustrations(summary):
    source=ROOT/'joint_mano_v28';allrows=read(source/'rows.json');ids=torch.tensor([i for i,r in enumerate(allrows) if r['role']=='dev_select']);rows=[allrows[i] for i in ids]
    cache=torch.load(source/'candidates.pt',weights_only=False,mmap=True);cache={k:v[ids] for k,v in cache.items() if torch.is_tensor(v) and len(v)==len(allrows)}
    lab=torch.load(source/'evaluation_labels.pt',weights_only=False,mmap=True);gt,valid=lab['gt'][ids],lab['valid'][ids]
    reg=torch.load(ROOT/'fitted_parameter_trajectory_v39/observations.pt',weights_only=False)['xyz']
    dit=torch.load(ROOT/'parameter_dit_trajectory_v37/observations.pt',weights_only=False)['xyz']
    constrained=torch.load(ROOT/'fitted_joint_refinement_v39/results.pt',weights_only=False)
    v16=cache['baseline'];base=cache['base'];variants=[('GT',gt),('v16',v16),('Parameterreg v39',reg),('ParameterDiT v31',dit),('Jointconstrained v39',constrained['prediction'])]
    i,j=zip(*EDGES);gl=(gt[:,i]-gt[:,j]).norm(dim=-1);vl=(v16[:,i]-v16[:,j]).norm(dim=-1)
    ev=valid[:,i]&valid[:,j];bone=(((vl/gl.clamp_min(1e-7)<.5)|(vl/gl.clamp_min(1e-7)>2))&ev).sum(-1)
    center=torch.tensor([r['window_index'] is not None for r in rows]);mask=valid.clone();mask[:,5]=False
    relative=lambda x:(((x-x[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)*mask).sum(-1)/mask.sum(-1).clamp_min(1)*1000
    er,eb,ej=relative(reg),relative(base),relative(constrained['prediction'])
    good=mask&((base-gt).norm(dim=-1)<.01);damage=((dit-gt).norm(dim=-1)*good).max(-1).values
    hardmask=valid.all(-1)&center&(eb>40)&constrained['frame_accepted']
    if not hardmask.any():hardmask=valid.all(-1)&center&(eb>40)
    assert hardmask.any()
    pick=[('bone',(bone.float()+relative(v16)*.001).masked_fill(~valid.all(-1),-1).argmax().item()),
          ('damage',damage.masked_fill(~(valid.all(-1)&center),-1).argmax().item()),
          ('hard',(ej-relative(v16)).masked_fill(~hardmask,-1e9).argmax().item())]
    colors=['#23984d','#e68b1e','#3d76d6','#9062c7','#ce4f61'];cases=[]
    for serial,(tag,n) in enumerate(pick,1):
        r=rows[n];image=cv2.imread(r['image']);assert image is not None
        box=np.asarray(r['box'],float);centerxy=(box[:2]+box[2:])/2;side=max(box[2:]-box[:2])*1.7
        lo=np.maximum(0,np.floor(centerxy-side/2).astype(int));hi=np.minimum(np.array(image.shape[:2][::-1]),np.ceil(centerxy+side/2).astype(int))
        fig=plt.figure(figsize=(17,6.5));zs=[]
        for k,(name,x) in enumerate(variants):
            uv=project_fisheye624(x[n:n+1],cache['camera_params'][n:n+1])[0].numpy()
            ax=fig.add_subplot(2,5,k+1);ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            for a,b in EDGES:
                if np.isfinite(uv[[a,b]]).all():ax.plot(uv[[a,b],0],uv[[a,b],1],color=colors[k],lw=1.8)
            okay=np.isfinite(uv).all(-1);ax.scatter(uv[okay,0],uv[okay,1],c=colors[k],s=8)
            ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.set_title(name);ax.axis('off')
            pose=(x[n]-x[n,5:6]).numpy()*1000;zs.append(pose)
            ax3=fig.add_subplot(2,5,k+6,projection='3d')
            for a,b in EDGES:ax3.plot(pose[[a,b],0],pose[[a,b],1],pose[[a,b],2],color=colors[k],lw=2)
            ax3.scatter(*pose.T,c=colors[k],s=7);ax3.view_init(elev=22,azim=-65);ax3.set_box_aspect([1,1,1])
            ax3.set_xlabel('X(mm)');ax3.set_ylabel('Y(mm)');ax3.set_zlabel('Z(mm)')
            camera=float((x[n]-gt[n]).norm(dim=-1)[mask[n]].mean()*1000);rel=float(relative(x)[n])
            ax3.set_title(f'Camera {camera:.1f}mm / Relative {rel:.1f}mm',fontsize=10)
        low=np.min(np.concatenate(zs),0)-5;high=np.max(np.concatenate(zs),0)+5;mid=(low+high)/2;radius=(high-low).max()/2
        for ax in [z for z in fig.axes if hasattr(z,'set_zlim')]:
            ax.set_xlim(mid[0]-radius,mid[0]+radius);ax.set_ylim(mid[1]-radius,mid[1]+radius);ax.set_zlim(mid[2]-radius,mid[2]+radius)
        accepted=bool(constrained['frame_accepted'][n]);fig.suptitle(f'{r["clip"]} frame{r["frame"]} track{r["track_id"]} | diagnostic{tag} | jointaccepted={accepted}',fontsize=13)
        fig.tight_layout();name=f'case{serial}.png';fig.savefig(OUT/name,dpi=140,bbox_inches='tight');plt.close(fig)
        cases.append(dict(name=name,tag=tag,sequence=r['sequence'],clip=r['clip'],frame=r['frame'],track_id=r['track_id'],accepted=accepted,
                          scope='Illustration picked using labels after inference only; never used to select runtime points/parameters.'))
    return cases


def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    assert (ROOT/'parameter_kinematic_v31/dit/done.json').exists()
    assert (ROOT/'parameter_dit_trajectory_v37/selection.json').exists()
    reference=read(ROOT/'parameter_trajectory_v33/selection.json')['baseline'];table=[compact('当前v16',reference)]
    for label,path in [('参数回归v31','parameter_trajectory_v33'),('参数查询回归v36','semantic_parameter_trajectory_v37'),('参数DiT v31','parameter_dit_trajectory_v37'),('拟合初始化回归v39','fitted_parameter_trajectory_v39')]:
        x=read(ROOT/path/'selection.json');table.append(compact(label+'（原始候选）',x['generator_review_only']))
    for label,path in [('联合可行性v34','joint_feasibility_v34'),('联合精度细化v35','joint_refinement_v35'),('参数查询联合v37','semantic_joint_refinement_v37'),('DiT联合v37','dit_joint_refinement_v37'),('拟合初始化联合v39','fitted_joint_refinement_v39')]:
        if (ROOT/path/'selection.json').exists():
            x=read(ROOT/path/'selection.json');table.append(compact(label+'（含回退）',x['final'],f"{x['accepted_tracks']}/{x['tracks']}"))
    summary=dict(complete=True,default='v16 unchanged',approved=False,scope='352 developmentselection windows, 6688 fingerpoints; 2673 dense observations, 2309 adjacent sameGT-handpairs; 14 highinitialrelativeerror>40mm windows.4sequences fromP0003. No independent fresh verification.',
        table=table,training=[dict(run=path,metrics=training_result(ROOT/path)) for path in ['parameter_kinematic_v31/regression','parameter_kinematic_v31/dit','parameter_pose_v32','semantic_parameter_v36/regression','fitted_parameter_v39/regression']],
        limitations=['GT UmeTrack and MANO wrist/landmarks differ; parameterdecoder aligns UmeTrack20 output.',
                     'Fullnative192x1280RGB remains; handpretrained32layerencoder frozen; projections/localization/temporalparameterhead trained.',
                     'Window17frame shape sharing does not ensure tracksharing; wholetracksolver adds sharedshape and chirality.',
                     'Acceptedtracks meet saved structure/point/speed/acceleration checks. Rejectedtracks retain v16; no wholeoutput guarantee.',
                     'No individual XYZ clipping after decoder. Exact manualpoints reject incompatible tracks.',
                     'GT IK label capacity (.39mmtrain/3.96mmdev) is an oracle diagnostic, not predicted accuracy.',
                     'No artificial occlusion, no newcamera, no dev_calibrate or sixthbatch assessment. Highinitialerror is not direct invisibility truth.'],
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*v3[1-9].py')})
    source=ROOT/'joint_mano_v28';allrows=read(source/'rows.json');ids=torch.tensor([i for i,r in enumerate(allrows) if r['role']=='dev_select']);rows=[allrows[i] for i in ids]
    centers=torch.tensor([r['window_index'] is not None for r in rows]);labels=torch.load(source/'evaluation_labels.pt',weights_only=False,mmap=True)
    baseline=torch.load(source/'candidates.pt',weights_only=False,mmap=True)['baseline'][ids]
    summary['paired_vs_v16']={}
    for name in ['joint_refinement_v35','semantic_joint_refinement_v37','dit_joint_refinement_v37','fitted_joint_refinement_v39']:
        predicted=torch.load(ROOT/name/'results.pt',weights_only=False)['prediction']
        summary['paired_vs_v16'][name]=paired_ci(predicted[centers],baseline[centers],labels['gt'][ids][centers],labels['valid'][ids][centers],[r for r in rows if r['window_index'] is not None])
    summary['cases']=illustrations(summary);save(OUT/'summary.json',summary)
    cells=lambda t:''.join(f'<td>{html.escape(str(x))}</td>' for x in t)
    rows=[]
    for x in table:
        rows.append('<tr>'+cells([x['name'],f"{x['camera_mm']:.2f}",f"{x['relative_mm']:.2f}",f"{x['camera_good_harmed']}/{x['camera_good_total']}",f"{x['relative_good_harmed']}/{x['relative_good_total']}",f"{x['hard_recovered']}" if x['hard_recovered'] is not None else '—',x['jump_pairs'],x['bone_extreme_frames'],x['bone_flicker'],x['tracks'] or '—'])+'</tr>')
    captions={'bone':'骨架异常：手模型输出更规整，定位仍可能不准确。','damage':'准确点损伤：完整候选需要复核，不能以平均误差改善自动接受。','hard':'困难例：初始相对误差>40mm；结构连贯不代表恢复正确。'}
    casehtml=''.join(f'<article><h3>{captions[c["tag"]]}</h3><p>{c["sequence"]}/{c["clip"]}，帧{c["frame"]}，轨迹{c["track_id"]}；联合候选接受={c["accepted"]}。标注仅用于事后展示。</p><a href="{c["name"]}"><img src="{c["name"]}" alt="actualRGB and3Dskeleton comparisons"></a></article>' for c in summary['cases'])
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HOT3D结构与整段时序实验</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1450px;margin:32px auto;padding:0 24px;color:#182433;background:#f6f8fb}h1{font-size:28px}section,article{background:white;padding:24px;border-radius:12px;margin:20px 0}table{border-collapse:collapse;min-width:1100px}td,th{text-align:left;padding:9px 12px;border-bottom:1px solid #dfe5ed}th{background:#e9eef6}img{width:100%;height:auto}.warning{background:#fff3d6;padding:18px;border-radius:8px}.flow{white-space:pre-wrap;background:#edf3f9;padding:18px;border-radius:8px}.scroll{overflow:auto}.small{font-size:14px;color:#4b5b6b}</style><h1>HOT3D：手模型结构约束＋整段时序约束</h1><p>已实现并训练、评估。结构异常减少，但本轮还未达到替换v16的精度与困难恢复门槛。</p><div class="warning"><strong>当前默认仍是v16。</strong>新模型和联合拟合结果都是研究候选。通过结构、边界检查只能说明候选可行，不能说明关键点恢复正确。</div><section><h2>约束接在哪里</h2><div class="flow">YOLO26手框/预测轨迹 → WiLoR观测＋17帧双向空间RGB\n→ 参数回归 / 参数DiT：根XYZ＋全局旋转＋20关节角＋5手型参数\n→ 与UmeTrack标签一致的运动学解码：20×3相机XYZ\n→ 整条轨迹共享手型与侧别，在世界坐标联合优化\n→ 显式求解并检查点位、速度、加速度边界 → 接受轨迹 / 保留v16</div><p>结构约束在输出解码器内；时序约束在整段联合求解内。解码后不逐点裁XYZ。与人工确认点不相容时拒绝整条候选，确认点保持原值。回退结果继承v16的限制。</p></section><section><h2>相同输入上的结果</h2><p class="small">开发选择集352窗、6688手指点；逐帧2673观测，2309同手相邻帧对。4条序列均来自P0003；是已用于开发的材料，不能当新主体或独立验证。困难组14窗定义为初始相对误差>40mm，不是逐指不可见真值。</p><div class="scroll"><table><thead><tr>'''+cells(['方法','相机误差mm','相对误差mm','准确相机点损伤','准确相对点损伤','困难坏点恢复/237','异常跳变对','骨长极端帧','骨长闪烁次','接受轨迹'])+'''</tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div><p class="small">准确点损伤：原本≤10mm变为>20mm。跳变：标注腕相对世界位移<10mm而预测>30mm；骨长极端：小于标注一半或大于两倍；闪烁：标注骨长变化<1mm而预测>10mm。全部联合结果包含回退；完整候选没有经过自动采用门槛。</p></section><section><h2>遇到的问题与处理</h2><ol><li><strong>MANO与标签腕点不一致：</strong>用成对标注核验，改为UmeTrack兼容的显式运动学。保留MANO失败试验。</li><li><strong>参数初始化太粗：</strong>手指角度事后替换诊断显示姿态预测是主要误差源；加强角度监督未获可接受收益。改参数查询关联实际关节，并用WiLoR预测XYZ做无GT逆运动学初始化。</li><li><strong>软惩罚仍越界：</strong>新增增广拉格朗日可行性求解，原接受边界保持不变；保存参数后重新解码核验。</li><li><strong>正常手型仍可能错位：</strong>同时评估相对/绝对误差、准确点损伤、困难恢复、跳变和骨长。通过保护或平滑不算恢复成功。</li><li><strong>工程对齐：</strong>修复追加bank按窗口顺序误映射、WiLoR XYZ误用BF16、参数查询忘记root附加槽、乘子原位更新导致反传错误。XYZ维持FP32，RGB维持BF16/16帧；失败记录保留。</li></ol><p>标签拟合残差约0.39mm（训练）、3.96mm（开发），仅证明表示能力。推理不输入标注姿态、主体手型或标注侧别。视觉主干32层冻结，完整192×1280空间特征保留，投影/定位与时序参数头训练。没有人工遮挡或额外相机。</p></section>'''+casehtml+'''<section><h2>证据与复现</h2><p><a href="summary.json">汇总JSON</a>；每个实验目录保留配置、预检、结果、损失与失败记录。代码与模型保存在服务器/mnt/why/hot3d_hand_residual/和/mnt/why/HOT3D/experiments/。本轮未打开开发校准集或第六批。</p></section></html>'''
    ci=summary['paired_vs_v16']['fitted_joint_refinement_v39']['relative']['ci95_delta_mm']
    note=f'最好联合v39相比v16的相对误差增加1.41mm；按来源序列配对的95%区间为+{ci[0]:.2f}至+{ci[1]:.2f}mm，支持精度退步。相机差异区间跨0。原始参数DiT逐帧采样种子按批起点分配，中心指标与训练选择缓存有小幅采样差异，均未达到保护门槛。'
    note+='通过边界的1732帧中骨长极端/闪烁为0，但仍有12/1647相邻对符合异常跳变诊断。声明的速度/加速度边界并不保证零跳变，不能称已解决运动不连续。'
    page=page.replace('全部联合结果包含回退；完整候选没有经过自动采用门槛。</p>','全部联合结果包含回退；完整候选没有经过自动采用门槛。</p><p>'+note+'</p>')
    page=page.replace('本轮未打开开发校准集或第六批。','本轮没有报告开发校准性能或使用第六批；既有缓存及标签预处理包含校准角色，不能称其数据未接触。参数模型训练/选择/性能评估仍限定train/dev_select。')
    (OUT/'report.html').write_text(page,encoding='utf-8')
    # Zip only human-readable evidence and figures, never massive native banks.
    archive=ROOT/'structure_temporal_review_v40.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.iterdir():
            if p.is_file():z.write(p,p.name)
        for name in RUNS:
            directory=ROOT/name
            if not directory.exists():continue
            for p in directory.rglob('*.json'):z.write(p,'evidence/'+str(p.relative_to(ROOT)))
    print(json.dumps(dict(complete=True,archive=str(archive),bytes=archive.stat().st_size,table=table)),flush=True)


if __name__=='__main__':main()
