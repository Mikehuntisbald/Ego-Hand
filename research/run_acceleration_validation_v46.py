"""Fresh fixed-path acceleration comparison, using observation-only cached RGB."""
import json,hashlib,collections,time,argparse
from pathlib import Path
import numpy as np,torch
from hand3d_v8_common import V7,save
import complete_hand_tracks_stable_v42 as engine
import complete_hand_tracks_dit_v43 as generator
import encode_hand3d_dense_v14 as sampling
from complete_hand_tracks_dit_v44 import complete
from complete_hand_tracks_acceleration_v46 import solve_profile,profile_config,bounded_sample
from stability_trajectory_v43 import saved_check
from evaluate_motion_thresholds_v45 import serialize
from compare_detectors import iou
from offline_rgb_encoder import crop_roi

RUN=V7.parent/'acceleration_validation_v46'

def main():
    assert (RUN/'fresh_ready.json').exists()
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);args=ap.parse_args()
    torch.set_num_threads(4);device=f'cuda:{args.shard}'
    rows=json.loads((RUN/'fresh_rows.json').read_text())
    affected_context=set(json.loads((RUN/'context_adapter_audit.json').read_text())['affected_tracks'])
    side_audit=json.loads((RUN/'side_adapter_audit.json').read_text());side_bank=torch.tensor(side_audit['predicted_right'],dtype=torch.bool)
    fields=['xyz','rgb','dense','risk_rgb','positions','roi','rays_camera']
    chunks=[torch.load(p,weights_only=False,map_location='cpu') for p in sorted((RUN/'fresh_observations').glob('*.pt'))]
    bank={k:torch.cat([c[k] for c in chunks]) for k in fields};del chunks
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):groups[(r['sequence'],r['clip'],r['track_id'])].append(i)
    groups={key:ids for j,(key,ids) in enumerate(groups.items()) if j%args.shards==args.shard}
    tracks=[]
    for key,ids in groups.items():
        ids.sort(key=lambda i:rows[i]['timestamp_ns']);tid='_'.join(map(str,key))
        if (RUN/'tracks'/f'{tid}_acc_x2_redecode.json').exists():
            prior=torch.load(RUN/'tracks'/f'{tid}_inputs.pt',weights_only=False,map_location='cpu')
            used=prior['observations']['right'].bool()^prior['observations']['parameter_side_outlier']
            if torch.equal(used,side_bank[torch.tensor(prior['indices'])]) and (tid not in affected_context or prior.get('context_version')=='bounded_v46'):continue
        frames=[]
        for i in ids:
            r=rows[i];right=int(side_bank[i])
            frames.append(dict(image=r['image'],camera=r['camera'],timestamp_s=r['timestamp_ns']*1e-9,
                box_xyxy=r['box'],box_confidence=r['score'],clip=r['clip'],observation_index=i,
                xyz_camera_m=bank['xyz'][i].tolist(),predicted_right=right,confirmed_3d=[False]*20))
        tracks.append(dict(id=tid,frames=frames))
    save(RUN/f'inference_input_{args.shard}.json',dict(tracks=tracks))
    (RUN/'tracks').mkdir(exist_ok=True)
    index=[0];parity=[False];previous_encode=engine.encode;previous_solver=generator.fit_stable;previous_selector=generator.select_sequence;previous_sample=sampling.sample
    def cached_encode(track,encoder,probe,projection,where):
        ids=torch.tensor([f['observation_index'] for f in track['frames']]);rr=[rows[i] for i in ids.tolist()]
        R=[];T=[];xy=[];obs=[]
        from hand3d_data_v7 import camera_pose
        import spatial_rgb_common as s
        for r,x in zip(rr,bank['xyz'][ids]):
            a,t=camera_pose(r['camera']);R.append(a);T.append(t)
            uv=s.common.from_json(r['camera']).eye_to_window(x.numpy())/1408
            xy.append(np.nan_to_num(uv));obs.append(np.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<1).all(-1))
        z=dict(times=np.array([f['timestamp_s'] for f in track['frames']]),xy=np.asarray(xy,np.float32),available=np.asarray(obs),
            positions=bank['positions'][ids].numpy(),roi=bank['roi'][ids].numpy(),scores=np.array([r['score'] for r in rr],np.float32))
        sampled,chosen=sampling.sample(z,where);slots=sampled['rgb_valid'];ix=sampled['ids']
        xyz=bank['xyz'][ids].to(where);R=torch.tensor(np.asarray(R),device=where);T=torch.tensor(np.asarray(T),device=where)
        world=torch.einsum('njc,nkc->njk',xyz,R)+T[:,None]
        aligned=torch.einsum('ntjc,nck->ntjk',world[ix]-T[:,None,None],R)*slots[:,:,None,None]
        rw=torch.einsum('nsc,nkc->nsk',bank['rays_camera'][ids].to(where),R)
        b=dict(xyz=aligned,available=slots[:,:,None].expand(-1,-1,20),base=xyz,dt=sampled['dt'],xy=sampled['xy'],observed_2d=sampled['observed_2d'],
            positions=sampled['positions'],roi=sampled['roi'],scores=sampled['scores'],
            rays=torch.einsum('ntsc,nck->ntsk',rw[ix],R)*slots[:,:,None,None],
            camera_origin=torch.einsum('ntc,nck->ntk',T[ix]-T[:,None],R)*slots[:,:,None],rgb_valid=slots,
            confirmed=torch.zeros(len(ids),20,dtype=torch.bool,device=where))
        for name,key in [('rgb','rgb'),('rgb_native','dense'),('risk_rgb','risk_rgb')]:b[name]=bank[key][ids][ix.cpu()].to(where)*slots[:,:,None,None]
        if not parity[0]:
            parity[0]=True
            small=dict(track,frames=track['frames'][:16])
            cached,cc=cached_encode(small,encoder,probe,projection,where)
            online,oc=previous_encode(small,encoder,probe,projection,where)
            assert cc==oc
            differences={k:float((cached[k].float()-online[k].float()).abs().max()) for k in cached}
            assert max(differences.values())<.002,differences
            save(RUN/f'cached_adapter_parity_{args.shard}.json',dict(passed=True,frames=len(small['frames']),max_abs=differences))
        return b,chosen
    def solve(decoder,cache,obs,solver_rows,config):
        track=tracks[index[0]];tid=track['id'];ix=[f['observation_index'] for f in track['frames']]
        torch.save(dict(cache={k:v.cpu() for k,v in cache.items()},observations={k:v.cpu() for k,v in obs.items()},
            rows=[{k:v for k,v in rows[i].items() if k not in ['gt','projection_valid','visible_fraction','matched']} for i in ix],indices=ix,initial_right=side_bank[torch.tensor(ix)],context_version='bounded_v46',solver_rows=solver_rows),RUN/'tracks'/f'{tid}_inputs.pt')
        result=None
        for profile in ['strict','acc_x2']:
            result=solve_profile(decoder,cache,obs,solver_rows,config,profile)
            assert result['check']['passed'];dest=RUN/'tracks'/f'{tid}_{profile}.pt';torch.save(serialize(result),dest)
            saved=torch.load(dest,weights_only=False,map_location=device)
            redecoded,check=saved_check(decoder,saved['parameters'],saved['layout'],profile_config(profile,config))
            assert check['passed'] and torch.allclose(redecoded,saved['world'],atol=1e-7,rtol=0)
            source=torch.tensor(saved['layout']['source_map'],device=device)
            camera=torch.einsum('njc,nck->njk',redecoded[source]-cache['translation'][:,None],cache['rotation'])
            assert torch.allclose(camera,saved['prediction'],atol=1e-7,rtol=0)
            save(RUN/'tracks'/f'{tid}_{profile}_redecode.json',dict(passed=True,check=check))
        print(json.dumps(dict(stage='paired_track',track=tid,frames=len(ix),done=index[0]+1,total=len(tracks))),flush=True)
        index[0]+=1;return result
    def checked_select(samples,cache,rr):
        obs,path=previous_selector(samples,cache,rr)
        poisoned=dict(cache,gt=torch.full_like(cache['base'],float('nan')),valid=torch.zeros_like(cache['base'][...,0],dtype=torch.bool))
        other,check=previous_selector(samples,poisoned,rr)
        assert path['selected_index']==check['selected_index'] and torch.equal(obs['parameter_state'],other['parameter_state'])
        tid=tracks[index[0]]['id'];save(RUN/'tracks'/f'{tid}_selection.json',dict(path,gt_poison_exact=True))
        return obs,path
    try:
        generator.select_sequence=checked_select;sampling.sample=lambda z,d:bounded_sample(z,d,previous_sample)
        engine.encode=cached_encode;generator.fit_stable=solve
        if tracks:
            result=complete(dict(tracks=tracks),device,True,1.6)
            result.update(config=profile_config('acc_x2'),mode='experimental_acceleration_v46',motion_profile='acc_x2',acceleration_multiplier=2,speed_multiplier=1,default_replaced=False)
            save(RUN/f'runtime_outputs_{args.shard}.json',result)
    finally:engine.encode=previous_encode;generator.fit_stable=previous_solver;generator.select_sequence=previous_selector;sampling.sample=previous_sample
    assert all((RUN/'tracks'/('_'.join(map(str,key))+'_acc_x2_redecode.json')).exists() for key in groups)
    save(RUN/f'outputs_frozen_{args.shard}.json',dict(complete=True,tracks=len(groups),observations=len(rows),gt_used_in_inference=False,
        same_candidates_and_path=True,profiles={p:profile_config(p) for p in ['strict','acc_x2']},
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for key in groups for p in (RUN/'tracks').glob('_'.join(map(str,key))+'_*.pt')}))

if __name__=='__main__':main()
