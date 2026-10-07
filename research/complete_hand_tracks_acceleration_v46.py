"""Runnable fixed v44 DiT with synchronized acceleration limits x2."""
import argparse,json
from pathlib import Path
import complete_hand_tracks_dit_v43 as generator
from complete_hand_tracks_dit_v44 import complete as complete_short
from stability_trajectory_v43 import DEFAULT,fit_stable
import encode_hand3d_dense_v14 as sampling
import torch


def profile_config(profile='acc_x2',config=None):
    assert profile in ['strict','acc_x2']
    result=dict(DEFAULT,**(config or {}))
    for name in ['root_speed','pose_speed','wrist_speed','joint_speed']:
        assert result[name]==DEFAULT[name], 'Speed limits must remain unchanged'
    for name in ['root_acc','pose_acc','wrist_acc','joint_acc']:
        result[name]=DEFAULT[name]*(2 if profile=='acc_x2' else 1)
    return result


def solve_profile(decoder,cache,obs,rows,config=None,profile='acc_x2',progress=None):
    return fit_stable(decoder,cache,obs,rows,profile_config(profile,config),progress)


def bounded_sample(z,device,sampler):
    sampled,chosen=sampler(z,device)
    # Nearest-frame tolerance must not extend the declared context horizon.
    outside=torch.zeros_like(sampled['rgb_valid'])
    for i,row in enumerate(chosen):
        for j,k in enumerate(row):
            if k is not None and abs(z['times'][k]-z['times'][i])>1.600002:outside[i,j]=True
    sampled['rgb_valid'][outside]=False;sampled['ids'][outside]=0
    for name in ['xy','observed_2d','roi','positions','scores']:sampled[name][outside]=0
    nominal=torch.tensor(sampling.OFFSETS,device=device,dtype=sampled['dt'].dtype)/30
    sampled['dt']=torch.where(outside,nominal[None],sampled['dt'])
    for i,j in outside.nonzero().cpu().tolist():chosen[i][j]=None
    return sampled,chosen


def complete(source,device='cuda:0',prepared=False,profile='acc_x2'):
    original=generator.fit_stable;original_sample=sampling.sample
    def solve(decoder,cache,obs,rows,config):
        return solve_profile(decoder,cache,obs,rows,config,profile)
    try:
        generator.fit_stable=solve;sampling.sample=lambda z,d:bounded_sample(z,d,original_sample)
        result=complete_short(source,device,prepared,1.6)
    finally:generator.fit_stable=original;sampling.sample=original_sample
    config=profile_config(profile,result['config']);result['config']=config
    limits={k:config[k] for k in ['root_speed','pose_speed','wrist_speed','joint_speed','root_acc','pose_acc','wrist_acc','joint_acc']}
    for tr in result['tracks']:
        assert tr['constraints']['limits']==limits and tr['constraints']['passed']
    result.update(mode='experimental_acceleration_v46',motion_profile=profile,
        acceleration_multiplier=2 if profile=='acc_x2' else 1,speed_multiplier=1,default_replaced=False,
        selector_motion_costs_changed=False)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--prepared',action='store_true')
    ap.add_argument('--profile',choices=['strict','acc_x2'],default='acc_x2');a=ap.parse_args()
    result=complete(json.loads(Path(a.input).read_text()),a.device,a.prepared,a.profile)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(output=str(path),profile=a.profile,passed=result['constraint_checks_passed'])),flush=True)


if __name__=='__main__':main()
