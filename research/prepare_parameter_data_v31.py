"""Observation-only coarse kinematics and supervised parameter targets.

Training hand-calibrations learn a generic PCA shape space. Evaluated actors'
shape/pose appear only in target files, never inference conditioning.
"""
import dataclasses,json,tarfile,collections,time
from pathlib import Path
import numpy as np,torch
from scipy.spatial.transform import Rotation
from hand3d_v8_common import V7,save
from prepare_joint_mano_v28 import records_bank,DATA
from compare_detectors import iou
from hand_tracking_toolkit.hand_models.umetrack_hand_model import from_json

OUT=V7.parent/'parameter_kinematic_v31'

def geometry_vector(model):
    return torch.cat([model.joint_rest_positions.flatten()/.05,model.landmark_rest_positions.flatten()/.05,model.joint_rotation_axes.flatten()/.2])

def load_shape(record,cache):
    sp,seq,clip=Path(record['image']).relative_to('/mnt/why/HOT3D/export/images').parts[:3];key=(sp,seq,clip)
    if key not in cache:
        with tarfile.open(V7.parent.parent/'rgb_clips'/sp/seq/(clip+'.tar')) as archive:raw=json.load(archive.extractfile('__hand_shapes.json__'))['umetrack']
        cache[key]=from_json({k:v for k,v in raw.items() if k not in ['mesh_vertices','mesh_triangles','dense_bone_weights']})
    return cache[key]

def learn_shape_space():
    path=OUT/'shape_space.pt'
    if path.exists():return torch.load(path,weights_only=False)
    roles=json.loads((V7.parent/'dense_sampling_v13/source_roles.json').read_text());subjects=collections.defaultdict(list);cache={}
    for r in roles:
        if r['role']!='train':continue
        rec=dict(image=f"/mnt/why/HOT3D/export/images/{r['split']}/{r['sequence']}/clip-{r['clip']:06d}/000000.jpg")
        subjects[r['subject']].append(geometry_vector(load_shape(rec,cache)))
    X=torch.stack([torch.stack(subjects[s]).mean(0) for s in sorted(subjects)]);mean=X.mean(0);U,S,V=torch.linalg.svd(X-mean,full_matrices=False)
    basis=V[:5];std=(S[:5]/np.sqrt(len(X)-1)).clamp_min(1e-5)
    template=torch.load(V7.parent/'joint_kinematic_v30/template.pt',weights_only=False)
    template.update(geometry_mean=mean,geometry_basis=basis,geometry_std=std,shape_dimensions=5)
    torch.save(template,path);save(OUT/'shape_provenance.json',dict(training_subjects=sorted(subjects),training_clips=sum(len(v) for v in subjects.values()),shape_dimensions=5,
        normalized_blocks='Jointrest/50mm, landmarkrest/50mm, unitjointaxes/.2',inference_subject_calibration=False,
        target_shapes='Per-subject GT hand calibrations provide supervised labels only; never put them in inference batch'))
    return template

def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(4);shape=learn_shape_space();d=torch.load(DATA/'dense_data.pt',weights_only=False,mmap=True);primary,full=records_bank(d)
    n=len(full)+1;right=torch.ones(n,dtype=torch.long);choices={int(r['fid']):r for r in json.loads((V7.parent/'side_data_v16/prediction_only_changes.json').read_text())}
    for fid,r in enumerate(full,1):
        box=np.array(r['box']);center=(box[:2]+box[2:])/2;size=max(float((box[2:]-box[:2]).max())*1.3,24.);roi=np.r_[center-size/2,center+size/2]
        side=r['side_predictions'];overlap=iou([roi],side['boxes'])[0]
        if len(overlap) and overlap.max()>=.1:right[fid]=int(side['classes'][int(overlap.argmax())])
        if fid in choices:right[fid]=choices[fid]['candidate_right']
    torch.save(right,OUT/'predicted_right_bank.pt')
    # Only target assignment uses annotation identity. Original input windows
    # and all observation banks stay byte-for-byte unchanged.
    gt_side={r['window_index']:r['GT_side_diagnostic'] for r in json.loads((V7.parent/'side_consensus_v16/audit_results.json').read_text())['sets']['train_development']['rows']}
    labels=torch.load(V7.parent/'context_data_v17/control/trajectory_labels.pt',weights_only=False,mmap=True)
    states=torch.zeros(len(d['roles']),17,21,3);mask=torch.zeros(len(d['roles']),17,dtype=torch.bool);hand_side=torch.zeros(len(d['roles']),dtype=torch.long)
    annotations={};shape_cache={};outofrange=0;anglecount=0
    lo,hi=shape['joint_limits'][:20].unbind(-1)
    for i,row in enumerate(d['rows']):
        side=gt_side[int(d['source_window_indices'][i])];assert side in ['left','right'];hand_side[i]=int(side=='right')
        current=int(d['feature_ids'][i,8]);R=d['rotation'][current];rec=full[current-1]
        subject_shape=load_shape(rec,shape_cache);vec=geometry_vector(subject_shape)
        coef=((vec-shape['geometry_mean'])@shape['geometry_basis'].T)/shape['geometry_std']
        for t,fid in enumerate(d['feature_ids'][i].tolist()):
            if not fid:continue
            record=full[fid-1];sp,seq,clip=Path(record['image']).relative_to('/mnt/why/HOT3D/export/images').parts[:3];path=V7.parent.parent/'export/annotations'/sp/seq/(clip+'.jsonl')
            if path not in annotations:annotations[path]=[json.loads(x) for x in path.read_text().splitlines()]
            hands=[h for h in annotations[path][record['frame']]['hands'] if h['side']==side]
            if not hands or not labels['valid'][i,t].all():continue
            assert len(hands)==1;h=hands[0];pose=h['umetrack_pose'];q=pose['T_world_from_wrist']['quaternion_wxyz']
            W=torch.tensor(Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix(),dtype=torch.float32);eye=R.T@W
            a=torch.tensor(pose['joint_angles'][:20]);outofrange+=int(((a<lo-1e-4)|(a>hi+1e-4)).sum());anglecount+=20
            x=states[i,t].flatten();x[:3]=labels['gt'][i,t,5]/.1;x[3:9]=eye[:,:2].T.flatten();x[9:29]=(2*(a-lo)/(hi-lo)-1).clamp(-1,1);x[29:34]=coef
            mask[i,t]=True
        assert mask[i,8]
    torch.save(dict(state=states,mask=mask,right=hand_side,**labels),OUT/'targets.pt')
    save(OUT/'target_ready.json',dict(complete=True,windows=len(d['roles']),valid_parameter_frames=int(mask.sum()),angle_labels=anglecount,
        angles_clipped_to_explicit_limits=outofrange,target_state='RootXYZ/0.1m, 6Drotation,20normalizedangles,5PCAshapes in21x3carrier; inactive29components zero',
        inference='No GT or per-subject calibration enters observations; original3417centers/RGB/XYZ retained',
        shape_label_range=[float(states.flatten(2)[:,:,29:34][mask].min()),float(states.flatten(2)[:,:,29:34][mask].max())],
        roles={r:d['roles'].count(r) for r in set(d['roles'])},no_fresh_or_retained_cases=True))
    print((OUT/'target_ready.json').read_text(),flush=True)

if __name__=='__main__':main()
