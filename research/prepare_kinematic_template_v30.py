"""Learn a generic UmeTrack-compatible hand from training subjects only."""
import json,tarfile,collections
import numpy as np,torch
from hand3d_v8_common import V7,save
from hand_tracking_toolkit.hand_models.umetrack_hand_model import from_json

def main():
    root=V7.parent.parent;out=V7.parent/'joint_kinematic_v30';out.mkdir(exist_ok=True)
    clips=json.loads((V7.parent/'dense_sampling_v13/source_roles.json').read_text());models=collections.defaultdict(list);angles=[]
    for r in clips:
        if r['role']!='train':continue
        split,seq,cid=r['split'],r['sequence'],r['clip']
        with tarfile.open(root/'rgb_clips'/split/seq/f'clip-{cid:06d}.tar') as t:raw=json.load(t.extractfile('__hand_shapes.json__'))['umetrack']
        model=from_json({k:v for k,v in raw.items() if k not in ['mesh_vertices','mesh_triangles','dense_bone_weights']})
        models[r['subject']].append(model)
        for x in (root/'export/annotations'/split/seq/f'clip-{cid:06d}.jsonl').read_text().splitlines()[::10]:
            for h in json.loads(x)['hands']:angles.append(h['umetrack_pose']['joint_angles'])
    subjects=sorted(models);reference=models[subjects[0]][0];values={}
    for k in ['joint_rotation_axes','joint_rest_positions','landmark_rest_positions']:
        per=torch.stack([torch.stack([getattr(m,k) for m in models[s]]).mean(0) for s in subjects]);values[k]=per.mean(0)
    values['joint_rotation_axes']=torch.nn.functional.normalize(values['joint_rotation_axes'],dim=-1)
    for k in ['joint_frame_index','joint_parent','joint_first_child','joint_next_sibling','landmark_rest_bone_weights','landmark_rest_bone_indices','joint_limits']:
        v=getattr(reference,k)
        if k!='joint_limits':
            for ms in models.values():
                for m in ms:assert torch.equal(v,getattr(m,k)),k
        values[k]=v.clone()
    values['hand_scale']=torch.tensor(1.)
    values['neutral_angles']=torch.tensor(angles).median(0).values
    values['training_subjects']=subjects
    torch.save(values,out/'template.pt');save(out/'template_provenance.json',dict(complete=True,training_subjects=subjects,training_clips=sum(len(v) for v in models.values()),
        pose_prior_samples=len(angles),geometry='Equal-subject mean; subject geometry averaged across training clips. No dev/test subject model read.',
        convention='UmeTrack canonical20 landmarks,22jointangles. Last2wristangles fixedzero. Explicitjointlimits from training model.',
        inference_shape='One6scale palm/finger shape per predictedtrack, optimized fromRGB andpredictedXYZ only; no subjectcalibration supplied.',natural_only=True))
    print((out/'template_provenance.json').read_text(),flush=True)

if __name__=='__main__':main()
