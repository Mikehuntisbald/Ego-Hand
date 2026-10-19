"""Lock six native-head validation clips without reading any hand annotations."""
import hashlib,json,tarfile,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
R=Path('/mnt/why/native_mano_multiseq_v3_20261019');PUBLIC=Path('/mnt/why/public_headhand_h20_20261019')
V1=Path('/mnt/why/native_mano_dit_20261019');V2=Path('/mnt/why/native_mano_dit_v2_20261019')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def transform(value):
    q=value['quaternion_wxyz'];out=np.eye(4);out[:3,:3]=Rotation.from_quat(q[1:]+q[:1]).as_matrix();out[:3,3]=value['translation_xyz'];return out.tolist()

def main():
    assert not (R/'protocol.json').exists()
    train=json.loads((V1/'data/selection.json').read_text());old_groups={(x[2],int(x[3])) for x in train['groups']}
    clips=[]
    for sequence in sorted(Path('/mnt/why/HOT3D/rgb_clips/val').iterdir()):
        choices=sorted(sequence.glob('clip-*.tar'));archive=choices[len(choices)//2];clip=int(archive.stem.split('-')[1])
        assert (sequence.name,clip) not in old_groups
        view=R/'clips'/archive.stem;view.mkdir(parents=True,exist_ok=False)
        for name in ['outputs','data','receipts']:(view/name).mkdir()
        for name in ['assets','vendor']:(view/name).symlink_to(PUBLIC/name,target_is_directory=True)
        # Legacy aliases only provide filesystem compatibility; source clip IDs
        # remain explicit in protocol and all new output manifests.
        images=view/'data/clip-002102';images.mkdir();frames=[];head=[]
        with tarfile.open(archive) as t:
            infos=sorted(m.name for m in t.getmembers() if m.name.endswith('.info.json'))
            for name in infos:
                info=json.load(t.extractfile(name));prefix=name.split('.')[0]
                cameras=json.load(t.extractfile(f'{prefix}.cameras.json'))
                sanitized={k:dict(calibration=v['calibration']) for k,v in cameras.items()}
                (images/name).write_text(json.dumps(info));(images/f'{prefix}.cameras.json').write_text(json.dumps(sanitized))
                picture=f'{prefix}.image_214-1.jpg';(images/picture).write_bytes(t.extractfile(picture).read())
                frames.append(dict(id=f'{archive.stem}_{prefix}',image=str(images/picture),group=sequence.name,dataset='HOT3D',split='native_head_validation',timestamp_s=info['ref_timestamp_ns']/1e9))
                head.append(dict(timestamp_ns=int(info['ref_timestamp_ns']),T_world_from_rgb=transform(cameras['214-1']['T_world_from_camera']),valid=True,segment=0))
        (view/'data/frames.json').write_text(json.dumps(dict(frames=frames)))
        pose=view/'outputs/head_clip2102';pose.mkdir()
        (pose/'headpose.json').write_text(json.dumps(dict(frames=head,trajectory_source='provided dataset reference camera, not inferred headpose',headpose_end_to_end=False,hand_GT_used=False)))
        clips.append(dict(clip=clip,sequence=sequence.name,subject=sequence.name.split('_')[0],archive=str(archive),archive_sha256=sha(archive),frames=len(frames),view=str(view),hand_GT_extracted=False,camera_trajectory='provided dataset reference; identical for all arms'))
    models={'regression':V1/'runs/regression/best_development.pt','linear_dit':V1/'runs/dit/best_development.pt','cosine_dit':V2/'runs/cosine_dit/best_development.pt','cosine_protected':V2/'runs/cosine_protected/best_development.pt'}
    protocol=dict(complete=True,locked_unix=time.time(),clips=clips,selection_rule='middle cached clip in every validation recording; filenames/counts only, no hand GT or pose error',
        models={name:dict(path=str(p),sha256=sha(p)) for name,p in models.items()},draws=4,DDIM_steps=10,seed=190019,context='17 noncausal slots, maximum +/-1.6s',
        arms=['WiLoR',*models,'cosine_shared','cosine_protected_displacement'],no_training_or_tuning_on_these_labels=True,
        prior_native_training='304 windows from six train subjects; selected 52 other P0003 windows; none of the six chosen recordings used',
        historical_exposure='public validation recordings existed in earlier project; this is new native-head validation, not fresh overall-project test',
        scope='native hand completion before BA; provided camera trajectory; not complete predicted headpose/depth/BA pipeline',
        safety='keep production default; report native/relative accuracy, recovery, harm, timestamp motion, coverage and paired sequence/subject uncertainty')
    (R/'protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(protocol,indent=2))

if __name__=='__main__':main()
