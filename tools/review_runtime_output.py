"""Post-seal FK/JSON/H5 review and real RGB overlays, WITHOUT any GT labels."""
import argparse
from pathlib import Path
import sys
import html
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ego_hand_runtime.io import read_json,atomic_json,verify_seal,sha256
from ego_hand_runtime.geometry import project,pixel_mask
from ego_hand_runtime.schema import Camera
from ego_hand_runtime.qa import EDGES


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--images-root',required=True)
    p.add_argument('--mano',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    root=Path(args.run);receipt=verify_seal(root)
    config=read_json(root/'expanded_config.json');data=read_json(root/'joints.json');episode=read_json(root/'input_contract.json')
    if sha256(args.mano)!=config['assets']['mano_right']['sha256']:raise RuntimeError('MANO SHA differs')
    import torch
    torch.set_num_threads(4)
    from ego_hand_runtime.native import NativeMano
    codec=NativeMano(args.mano,'cpu')
    import h5py
    errors=[];points=0
    with h5py.File(root/'bundle.h5','r') as f:
        for _,g in f['tracks'].items():
            valid=g['validmask'][:];state=g['state_world51'][:][valid];beta=g['betas'][:][valid]
            reflection=g['reflection'][:][valid];right=reflection[:,0]>0
            if not len(state):continue
            with torch.inference_mode():
                a,b=codec.raw(torch.tensor(g['global_orient'][:][valid][:,0]),torch.tensor(g['hand_pose'][:][valid]),torch.tensor(beta))
                independent=(a.numpy()+g['transl'][:][valid,None])*reflection[:,None]
            xyz=g['joints_world_m'][:][valid]
            errors.extend(np.linalg.norm(independent-xyz,axis=-1).reshape(-1).tolist());points+=len(state)*21
            if float(np.abs(independent-xyz).max())>1e-6:raise RuntimeError('Exported parameters do not reproduce delivered skeleton')
            if float(np.abs(codec(state,beta,right)-xyz).max())>1e-6:raise RuntimeError('Stored wrist states disagree with delivered skeleton')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);(out/'images').mkdir()
    valid=[i for i,f in enumerate(data['frames']) if f['head_valid'] and any(h['valid'] for h in f['hands'])]
    chosen={valid[0],valid[len(valid)//2],valid[-1]}
    motion=[]
    for a,b in zip(valid[:-1],valid[1:]):
        old={str(h['track_id']):h for h in data['frames'][a]['hands'] if h['valid']}
        new={str(h['track_id']):h for h in data['frames'][b]['hands'] if h['valid']}
        dt=(data['frames'][b]['timestamp_ns']-data['frames'][a]['timestamp_ns'])/1e9
        if 0<dt<.1 and old.keys()&new.keys():
            speed=max(np.linalg.norm(np.array(new[k]['joints_world_m'])[0]-np.array(old[k]['joints_world_m'])[0])/dt for k in old.keys()&new.keys())
            motion.append((speed,b))
    chosen.update(i for _,i in sorted(motion,reverse=True)[:3])
    import cv2
    previews=[]
    for index in sorted(chosen):
        frame=data['frames'][index];source=episode['frames'][index]
        cams=source.get('cameras',episode['cameras']);camera=Camera.from_dict(next(c for c in cams if c['role']=='rgb'))
        image=source['images'][camera.id];path=Path(args.images_root)/Path(image['path']).name
        pixels=cv2.imread(str(path));T=np.array(frame['world_from_device'])@camera.T_device_from_camera
        for hand in frame['hands']:
            if not hand['valid']:continue
            xyz=(np.array(hand['joints_world_m'])-T[:3,3])@T[:3,:3]
            uv=project(xyz,camera);mask=pixel_mask(xyz,uv,camera);color=(60,230,60) if hand['right'] else (0,155,255)
            for a,b in EDGES:
                if mask[a] and mask[b]:cv2.line(pixels,tuple(np.rint(uv[a]).astype(int)),tuple(np.rint(uv[b]).astype(int)),color,3,cv2.LINE_AA)
            for point,good in zip(uv,mask):
                if good:cv2.circle(pixels,tuple(np.rint(point).astype(int)),4,color,-1,cv2.LINE_AA)
        size=(900,round(pixels.shape[0]*900/pixels.shape[1]));pixels=cv2.resize(pixels,size)
        cv2.putText(pixels,'Predictions only / no GT / frame '+str(frame['source_index']),(14,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
        name='images/frame_'+str(frame['source_index'])+'.jpg';cv2.imwrite(str(out/name),pixels)
        previews.append(dict(source_index=frame['source_index'],timestamp_ns=frame['timestamp_ns'],image=name))
    result=read_json(root/'result.json')
    summary=dict(source_run=str(root.resolve()),prediction_binding=receipt['binding'],program_success=result['program_success'],
                 independent_CPU_FK_points=points,FK_max_euclidean_m=float(max(errors)),FK_mean_euclidean_m=float(np.mean(errors)),
                 quality_admitted=False,absolute_accuracy_evidence=False,GT_used=False,previews=previews,
                 sample_selection='first/middle/last valid and largest raw predicted wrist velocities; not GT failures')
    atomic_json(out/'review.json',summary)
    cards=''.join('<figure><img src="'+x['image']+'"><figcaption>源帧 '+str(x['source_index'])+' · '+str(x['timestamp_ns'])+' ns</figcaption></figure>' for x in previews)
    page='<!doctype html><meta charset="utf-8"><title>Ego-Hand runtime 验证</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;background:#101922;color:#eee}p{line-height:1.7}img{width:100%}figure{margin:20px 0}pre{white-space:pre-wrap;background:#1e2b37;padding:18px}</style>'
    page+='<h1>真实 RGB 与原生 MANO 交付检查</h1><p>绿色为右手，橙色为左手。这些叠图只有预测，没有 GT；包含预测运动较大的帧，不能据此称绝对精度或跳变修复通过。</p>'
    page+='<pre>'+html.escape(__import__('json').dumps(result,ensure_ascii=False,indent=2))+'</pre><p>独立 CPU FK 核验 '+str(points)+' 个点，最大重建差 '+format(max(errors)*1000,'.6f')+' mm。这是交付内部一致性，不是真值误差。</p>'+cards
    (out/'report.html').write_text(page,encoding='utf-8');print(summary)


if __name__=='__main__':main()
