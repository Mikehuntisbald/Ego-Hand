"""Finite natural-video integration, checkpoint audit and local-ready report."""
import os,json,time,subprocess,hashlib,fcntl
from pathlib import Path
import torch,numpy as np,cv2
B=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53B_20261007')
A=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53_20261007')
CODE=Path('/mnt/why/hot3d_hand_residual')
RF='/mnt/why/HOT3D/experiments/rfdetr_hand_instance_v52_20261007/venv/bin/python'
BASE='/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python'
EDGES=[(6,7),(7,0),(5,8),(8,9),(9,10),(10,1),(5,11),(11,12),(12,13),(13,2),(5,14),(14,15),(15,16),(16,3),(5,17),(17,18),(18,19),(19,4)]
COLORS=[(60,210,255),(255,170,80),(90,220,120),(220,80,220),(110,100,250)]
def free_gpu():return 'GPU-2f45f698-b39e-0f82-204d-f3a7d925baa9' not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader']).decode()
def render_natural():
    raw=json.loads((B/'nail_RF/predictions.json').read_text());source=json.loads((B/'nail_complete/input_tracks.json').read_text());prediction=json.loads((B/'nail_complete/prediction.json').read_text());out=B/'review';out.mkdir(exist_ok=True);by_time={};lookup={};bone=[];motion=[]
    for tr in source['tracks']:
        for f in tr['frames']:lookup[tr['id'],round(f['timestamp_s'],6)]=f
    for i,tr in enumerate(prediction['tracks']):
        xyz=np.asarray([f['candidate_xyz_camera_m'] for f in tr['frames']]);times=np.asarray([f['timestamp_s'] for f in tr['frames']]);assert np.isfinite(xyz).all()
        lengths=np.stack([np.linalg.norm(xyz[:,a]-xyz[:,b],axis=-1) for a,b in EDGES],-1);bone.append(dict(id=tr['id'],frames=len(xyz),max_edge_length_CV=float((lengths.std(0)/np.maximum(lengths.mean(0),1e-9)).max())))
        if len(times)>1:
            steps=np.linalg.norm(np.diff(xyz,axis=0),axis=-1);motion.append(dict(id=tr['id'],max_joint_step_m=float(steps.max()),p95_joint_step_m=float(np.quantile(steps,.95)),constraints=tr.get('constraints'),fast_motion_GT_available=False))
        for f in tr['frames']:by_time.setdefault(round(f['timestamp_s'],6),[]).append((i,tr,f))
    writer=cv2.VideoWriter(str(out/'natural_video_raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),29.97002997,(1024,512));counts=[]
    from render_rfdetr_v52 import overlay
    for i,frame in enumerate(raw['frames']):
        d=np.load(frame['output']);image=cv2.imread(frame['image']);mask_view=overlay(image,d['masks'],d['boxes'],d['scores'],f'RF boxes + masks | frame {i}');pose_view=image.copy();counts.append(len(d['scores']))
        for j,tr,f in by_time.get(round(frame['timestamp_s'],6),[]):
            inp=lookup[tr['id'],round(f['timestamp_s'],6)];xyz=np.asarray(f['candidate_xyz_camera_m']);fx,fy,cx,cy=inp['camera']['calibration']['projection_params'][:4];uv=xyz[:,:2]/np.maximum(xyz[:,2:],1e-6)*[fx,fy]+[cx,cy];color=COLORS[j%len(COLORS)]
            for a,b in EDGES:
                points=uv[[a,b]]
                if np.isfinite(points).all() and (points>=0).all() and (points<1408).all():cv2.line(pose_view,tuple(uv[a].round().astype(int)),tuple(uv[b].round().astype(int)),color,4,cv2.LINE_AA)
            box=np.asarray(inp['box_xyxy']).round().astype(int);cv2.rectangle(pose_view,tuple(box[:2]),tuple(box[2:]),color,3);cv2.putText(pose_view,tr['id'],(max(0,box[0]),max(20,box[1]-8)),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
        cv2.putText(pose_view,'Same full 3D core | uncalibrated, review required',(20,40),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
        tile=np.concatenate([cv2.resize(mask_view,(512,512)),cv2.resize(pose_view,(512,512))],axis=1);writer.write(tile)
        if i in [0,24,48,69,95,119]:cv2.imwrite(str(out/f'natural_{i:03d}.jpg'),tile)
    writer.release()
    check=dict(frames=len(counts),fps=29.97002997,instances_per_frame=counts,tracks=len(prediction['tracks']),track_lengths=[len(t['frames']) for t in prediction['tracks']],constraints_passed=prediction['constraint_checks_passed'],bone_consistency=bone,motion=motion,GT_3D=False,identity_GT=False,true_fast_motion_retention_unverified=True,uncalibrated_camera=True,diagnostic_replay=True,default_replaced=False)
    (out/'natural_checks.json').write_text(json.dumps(check,indent=2));return check
def verify(selection):
    torch.set_num_threads(4);initial=torch.load(A/'checkpoint_metadata_fix/best_admitted_ema.pth',map_location='cpu',weights_only=False)['model'];checkpoint=Path(selection['checkpoint']);state=torch.load(checkpoint,map_location='cpu',weights_only=False);trained=state['model'];updates={k:float((v.float()-trained[k].float()).abs().max()) for k,v in initial.items() if k in trained and v.shape==trained[k].shape and 'encoder' in k and any(s in k for s in ['query.weight','key.weight','value.weight','layernorm.weight'])};assert updates and all(v>0 for v in updates.values())
    from rfdetr import RFDETR
    reload=RFDETR.from_checkpoint(str(checkpoint),device='cpu',trust_checkpoint=True);actual=reload.model.model.state_dict();assert set(actual)==set(trained) and all(torch.equal(actual[k],v) for k,v in trained.items());del reload
    full=torch.load(B/'full/checkpoint_7.ckpt',map_location='cpu',weights_only=False);assert full['optimizer_states'][0]['state'] and full['lr_schedulers'] and all(k in full['experiment_rng'] for k in ['python','numpy','torch','cuda']);done=json.loads((B/'full/done.json').read_text());assert done['visual_and_mask_updates_passed']
    out=dict(selected_checkpoint=str(checkpoint),selected_sha256=hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest(),actual_encoder_updates=updates,actual_attention_norm_updates=True,all_tensors_equal_after_actual_reload=True,live_gradients=done['gradient_norm_max'],actual_head_and_projection_updates=done['weight_max_abs_update'],optimizer_entries=len(full['optimizer_states'][0]['state']),RNG_and_scheduler_saved=True,exact_resume_execution_tested=False,GT_free_test_predictions=True,test_used_for_training=False,default_replaced=False)
    (B/'training_verification.json').write_text(json.dumps(out,indent=2));return out
def main():
    B.mkdir(exist_ok=True);lock=(B/'finalize.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);env=dict(os.environ,CUDA_VISIBLE_DEVICES='3')
    def status(phase,**kwargs):(B/'finalize_status.json').write_text(json.dumps(dict(phase=phase,pid=os.getpid(),time=time.time(),**kwargs),indent=2))
    def run(name,script,*args,base=False,gpu=True):
        if gpu:
            while not free_gpu():status('waiting_GPU3',next=name);time.sleep(30)
        status(name)
        with (B/(name+'.log')).open('a') as log:
            child=subprocess.Popen([BASE if base else RF,str(CODE/script),*map(str,args)],env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT);status(name,child_pid=child.pid);ret=child.wait()
        if ret:raise RuntimeError(f'{name} exited {ret}; preserve logs and repair separately')
    try:
        while True:
            p=B/'replacement_eval/status.json'
            if p.exists():
                phase=json.loads(p.read_text())['phase']
                if phase=='failed':raise RuntimeError('replacement evaluation failed')
                if phase=='complete_pending_report':break
            status('waiting_replacement_evaluation');time.sleep(30)
        selection=json.loads((B/'replacement_eval/selection.json').read_text());status('verify');verification=verify(selection)
        if not (B/'nail_RF/freeze.json').exists():run('nail_instances','predict_instances_v53.py','--input',B/'nail_rgb.json','--output',B/'nail_RF','--checkpoint',selection['checkpoint'],'--threshold',selection['threshold'],'--policy',selection['policy'])
        if not (B/'nail_complete/freeze.json').exists():run('nail_3d','complete_rfdetr_v52.py','--predictions',B/'nail_RF/predictions.json','--output',B/'nail_complete/prediction.json','--threshold',selection['threshold'],'--video',base=True)
        status('render');natural=render_natural();run('report','render_replacement_v53.py',gpu=False)
        out=B/'review';page=(out/'report.html').read_text();extra='<h2>真实护理：全120帧完整3D</h2><p>RF前端接同一WiLoR/参数DiT/FK/整段优化；保持原始29.97fps时间戳。轨迹长度 '+str(natural['track_lengths'])+'，保存轨迹硬约束通过='+str(natural['constraints_passed'])+'。没有身份/指尖3D真值，骨长稳定也不能证明没有串手或猜错。既有素材诊断回放。</p><video controls src="natural_video.mp4"></video>'+''.join(f'<figure><img src="{p.name}"></figure>' for p in sorted(out.glob('natural_*.jpg')))
        page=page.replace('<h2>自然恢复与失败</h2>',extra+'<h2>自然恢复与失败</h2>');(out/'report.html').write_text(page,encoding='utf-8')
        ffmpeg=__import__('shutil').which('ffmpeg')
        if ffmpeg:
            subprocess.run([ffmpeg,'-y','-i',str(out/'natural_video_raw.mp4'),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'natural_video.mp4')],check=True,stdout=subprocess.DEVNULL,stderr=(B/'video_encode.log').open('w'))
        else:__import__('shutil').copy2(out/'natural_video_raw.mp4',out/'natural_video.mp4')
        (out/'training_verification.json').write_bytes((B/'training_verification.json').read_bytes());status('complete',report=str(out/'report.html'),default_replaced=False)
    except Exception as e:status('failed',error=repr(e));raise
if __name__=='__main__':main()
