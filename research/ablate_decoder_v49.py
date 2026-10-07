"""Separate hand-fit and network contributions using fixed observation-only IK."""
import hashlib,json,shutil,time
from pathlib import Path
import torch
import ablate_modules_v49 as review
from parameter_codec_v31 import ParameterCodec
from parameter_trajectory_model_v33 import ParameterTrajectoryDecoder
from complete_hand_tracks_acceleration_v46 import solve_profile

RUN=review.V7.parent/'decoder_ablation_v49_20261007'
FIRST=review.V7.parent/'module_ablation_v49_20261007'
def main():
    torch.set_num_threads(4);RUN.mkdir(parents=True,exist_ok=True);snap=RUN/'code_snapshot';snap.mkdir(exist_ok=True)
    for src in Path(__file__).parent.glob('*.py'):
        dst=snap/src.name
        if dst.exists():assert dst.read_bytes()==src.read_bytes()
        else:shutil.copy2(src,dst)
    review.RUN=RUN;review.NAMES=dict(full='完整保护DiT＋求解器',wilor='原WiLoR（已有前端手模型）',
        coarse_fk='原WiLoR的200步手模型拟合，无时序网络',coarse_solver='手模型拟合＋整段求解，无时序网络')
    (RUN/'protocol.json').write_text(json.dumps(dict(fixed='Same natural RGB/tracks and original200-step observation-only IK. Same decoder,350-step solver,accelerationx2.',
        scope='Same10 previously evaluated clips/5sequences/2known actors. Diagnostic replay only.',
        caveat='coarse_fk vs WiLoR measures this additional hand-model fitting, not absence vs presence of all hand priors; WiLoR itself already uses MANO. Full vscoarse_solver measures temporal network plus its candidates under shared solver.'),indent=2))
    for name in ['full','wilor']:
        folder=RUN/name;folder.mkdir(exist_ok=True)
        for file in ['result.pt','sealed.json']:
            src=FIRST/name/file;dst=folder/file
            if dst.exists():assert dst.read_bytes()==src.read_bytes()
            else:shutil.copy2(src,dst)
    if not (RUN/'coarse_solver/sealed.json').exists():
        cache=torch.load(review.INPUT/'cache.pt',weights_only=False,map_location='cpu');rows=json.loads((review.INPUT/'rows.json').read_text())
        data=torch.load(review.INPUT/'inputs.pt',weights_only=False,mmap=True);right=torch.load(review.INPUT/'right.pt',weights_only=False)
        coarse=torch.load(review.INPUT/'coarse.pt',weights_only=False);ids=data['feature_ids'][:,8];state=coarse[ids];side=right[ids]
        xyz=ParameterCodec('cpu').decode(state[:,None],side,shared=False)[:,0]
        review.seal('coarse_fk',dict(prediction=xyz,parameter_state=state,right=side),dict(GT_free=True,source='Frozen observation-only200stepIK bank',source_sha256=review.sha(review.INPUT/'coarse.pt')))
        obs=dict(xyz=xyz,parameter_state=state,right=side,parameter_side_outlier=torch.zeros(len(rows),dtype=torch.bool))
        start=time.time();result=solve_profile(ParameterTrajectoryDecoder('cpu'),cache,obs,rows,profile='acc_x2',progress=lambda x:print(json.dumps(dict(**x,seconds=time.time()-start)),flush=True))
        assert result['check']['passed'];review.seal('coarse_solver',result,dict(GT_free=True,original_limit_check=result['check'],network=False,seconds=time.time()-start))
    if not (RUN/'summary.json').exists():review.evaluate()
    (RUN/'controller_status.json').write_text(json.dumps(dict(stage='complete',review=str(RUN/'review/report.html'))))
if __name__=='__main__':main()
