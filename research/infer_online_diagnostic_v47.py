"""Freeze matched generated candidates and stable outputs before replay labels."""
import argparse,hashlib,json,time
import numpy as np,torch
from hand3d_v8_common import V7,save
from online_parameter_model_v47 import RUN,OnlineVisual,load_head
from parameter_codec_v31 import observation_batch
from parameter_candidates_v43 import select_sequence
from parameter_trajectory_model_v33 import ParameterTrajectoryDecoder
from complete_hand_tracks_acceleration_v46 import solve_profile,profile_config
from stability_trajectory_v43 import saved_check
from evaluate_motion_thresholds_v45 import serialize

OUT=RUN/'diagnostic_v46'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['initial','frozen','joint','frozen_last','joint_last'],required=True);a=ap.parse_args()
    torch.set_num_threads(4);device='cuda:3';start=time.time();assert json.loads((OUT/'ready.json').read_text())['complete']
    for mode in ['joint','frozen']:assert json.loads((RUN/f'dit_{mode}/done.json').read_text())['complete']
    dest=OUT/a.mode;dest.mkdir(exist_ok=True)
    if (dest/'sealed.json').exists():return
    data=torch.load(OUT/'inputs.pt',weights_only=False,mmap=True);data={k:v.to(device) if torch.is_tensor(v) else v for k,v in data.items()}
    coarse=torch.load(OUT/'coarse.pt',weights_only=False).to(device);right=torch.load(OUT/'right.pt',weights_only=False).to(device)
    cache={k:v.to(device) for k,v in torch.load(OUT/'cache.pt',weights_only=False).items()};rows=json.loads((OUT/'rows.json').read_text())
    model,initial_path,initial=load_head('dit',device);visual=OnlineVisual(device,False)
    if a.mode!='initial':
        arm=a.mode.split('_')[0];name='last.pt' if a.mode.endswith('_last') else 'best.pt'
        path=RUN/f'dit_{arm}'/name;ck=torch.load(path,weights_only=False,map_location=device)
        model.load_state_dict(ck['model']);visual.load_tail(ck['visual_tail'])
    else:path=initial_path;ck=initial
    model.eval();visual.configure();pixels=np.load(OUT/'pixels.npy',mmap_mode='r')
    candidates=dest/'candidates.pt'
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        bank=torch.zeros(len(right),192,1280,device=device)
        for begin in range(1,len(right),128):
            end=min(begin+128,len(right));bank[begin:end]=visual.encode_pixels([pixels[i] for i in range(begin,end)],device)
        pieces={k:[] for k in ['states','xyz','right']};poison_exact=False
        for begin in range(0,len(rows),8):
            ix=torch.arange(begin,min(begin+8,len(rows)),device=device)
            b=observation_batch(data,ix,cache['risk'],coarse,right,preserve_fitted=True);b['rgb_native']=bank[data['feature_ids'][ix]]
            def generate(bb):
                enc=model.encode(bb);delta=model.sample_trajectory(bb,enc,10,4,202610131+begin)
                state=bb['kinematic_coarse'][None]+delta;side=enc[3]['side_logits'].argmax(-1)
                xyz=torch.stack([model.codec.decode(x,side)[:,8] for x in state])
                return state[:,:,8].float().transpose(0,1),xyz.float().transpose(0,1),side
            state,xyz,side=generate(b)
            if begin==0:
                dirty=generate(dict(b,gt=torch.full((len(ix),17,20,3),float('nan'),device=device),gt_right=1-side,gt_shape=torch.ones(len(ix),10,device=device)*999))
                assert torch.equal(state,dirty[0]) and torch.equal(xyz,dirty[1]);poison_exact=True
            for key,value in [('states',state),('xyz',xyz),('right',side)]:pieces[key].append(value.cpu())
            if begin%800==0:print(json.dumps(dict(mode=a.mode,stage='candidates',done=min(begin+8,len(rows)),total=len(rows),seconds=time.time()-start)),flush=True)
        samples={k:torch.cat(v) for k,v in pieces.items()};torch.save(samples,candidates)
    obs,path_selection=select_sequence(samples,cache,rows)
    other,poison=select_sequence(samples,dict(cache,gt=torch.full_like(cache['base'],float('nan')),valid=torch.zeros(len(rows),20,dtype=torch.bool,device=device)),rows)
    assert poison['selected_index']==path_selection['selected_index'] and torch.equal(obs['parameter_state'],other['parameter_state'])
    save(dest/'selection.json',dict(path_selection,gt_poison_exact=True))
    obs['parameter_side_outlier']=obs['right']!=right[data['feature_ids'][:,8]].cpu()
    result=solve_profile(ParameterTrajectoryDecoder(device),cache,{k:v.to(device) for k,v in obs.items()},rows,profile='acc_x2',
        progress=lambda x:print(json.dumps(dict(mode=a.mode,stage='solver',**x)),flush=True))
    assert result['check']['passed'];torch.save(serialize(result),dest/'result.pt')
    saved=torch.load(dest/'result.pt',weights_only=False,map_location=device)
    world,check=saved_check(ParameterTrajectoryDecoder(device),saved['parameters'],saved['layout'],profile_config('acc_x2'))
    assert check['passed'] and torch.allclose(world,saved['world'],atol=1e-7,rtol=0)
    save(dest/'sealed.json',dict(complete=True,generator_gt_poison_exact=poison_exact,selector_gt_poison_exact=True,parameter_redecode_passed=True,
        check=check,checkpoint=str(path),selected_step=ck['step'],checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        result_sha256=hashlib.sha256((dest/'result.pt').read_bytes()).hexdigest(),observations=len(rows),seconds=time.time()-start,
        scope='Previously evaluated v46 clips. Fixed last-step diagnostics do not override dev-select admission. All variants use same frozen risk/RGB selection evidence and same batched stable solver.',default_changed=False))
    print((dest/'sealed.json').read_text(),flush=True)
if __name__=='__main__':main()
