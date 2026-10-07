"""Instrument the unchanged v43 solver; defaults must reproduce v44 exactly.

Only diagnostic stages and explicitly supplied motion/smoothing settings are
added. Original production files and structural decoding are untouched.
"""
from pathlib import Path

path=Path(__file__).parent/'stability_trajectory_v43.py'
source=path.read_text()


def once(before,after):
    global source
    assert source.count(before)==1,(before,source.count(before))
    source=source.replace(before,after)


once("    theta0=torch.tensor(smooth_theta,device=device,dtype=torch.float32)",'''    def stage_from_angles(angle,rotation,position):
        angle=torch.as_tensor(angle,device=device,dtype=torch.float32)
        rotation=torch.as_tensor(rotation,device=device,dtype=torch.float32)
        position=torch.as_tensor(position,device=device,dtype=torch.float32)
        raw=torch.logit(((angle-lo)/(hi-lo)).clamp(.001,.999))
        with torch.no_grad():return decoder(raw,rotation_to_six(rotation),position,beta,group,sign).detach()
    stage_world={'side_repaired_before_smoothing':stage_from_angles(dense_theta,rotation0,dense_root)}
    theta0=torch.tensor(smooth_theta,device=device,dtype=torch.float32)''')
once("    history=[]\n    for step in range(cfg['steps']):",'''    with torch.no_grad():
        stage_world['after_initial_smoothing']=decoder(local,rotation_to_six(anchor@exp_rotation(rv)),root,beta,group,sign).detach()
    history=[]
    for step in range(cfg['steps']):''')
once("(cfg[name+'_acc']*.4)","(cfg.get('temporal_motion_limits',cfg.get('optimization_motion_limits',cfg))[name+'_acc']*.4)")
once("(cfg[name+'_'+suffix]*.9)","(cfg.get('optimization_motion_limits',cfg)[name+'_'+suffix]*.9)")
once("loss=data+prior+.08*temporal+100*penalty+cfg['rgb_weight']*rgb", "loss=data+prior+cfg.get('temporal_weight',.08)*temporal+cfg.get('penalty_weight',100.)*penalty+cfg['rgb_weight']*rgb")
once("        restoration=[]",'''        stage_world['optimized_before_hard_restoration']=decoder(local,rotation_to_six(rot),root,beta,group,sign).detach()
        restoration=[]''')
once("        return dict(prediction=prediction,world=world,parameters=parameters,layout=layout,check=check,",'''        stages={name:torch.einsum('njc,nck->njk',value[source]-cache['translation'][:,None],cache['rotation']) for name,value in stage_world.items()}
        return dict(prediction=prediction,world=world,stages=stages,parameters=parameters,layout=layout,check=check,''')
exec(compile(source,str(path)+' [v45 diagnostic instrumentation]','exec'),globals())
