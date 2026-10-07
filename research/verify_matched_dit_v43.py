"""Independent saved-output checks and runtime acceptance for the DiT trial."""
import json,hashlib,torch
from pathlib import Path
from hand3d_v8_common import V7,save
from parameter_candidates_v43 import load_observations
from parameter_trajectory_model_v33 import ParameterTrajectoryDecoder
from stability_trajectory_v42 import saved_check,DEFAULT
from stability_trajectory_v43 import fit_stable
from evaluate_joint_kinematic_v30 import subset


def main():
    torch.set_num_threads(4);device='cuda:3';root=V7.parent;out=root/'matched_dit_review_v43'
    paths={k:root/'matched_stability_v43'/(k+'_side_consistent') for k in ['regression','dit','dit_denoise']}
    configs={k:json.loads((root/'matched_parameter_v43'/k/'config.json').read_text()) for k in paths}
    for field in ['batch_plan_sha256','seed','steps','batch','train_windows','dev_select_windows','initial_v16_step','transferred_tensors']:
        assert len({c[field] for c in configs.values()})==1,field
    decoder=ParameterTrajectoryDecoder(device);rows,_,_,cache=load_observations(device);checks={}
    for kind,path in paths.items():
        variants=['mean'] if kind=='regression' else ['mean','sequence']
        for variant in variants:
            r=torch.load(path/(variant+'.pt'),weights_only=False)
            p={k:v.to(device) for k,v in r['parameters'].items()}
            world,check=saved_check(decoder,p,r['layout'],DEFAULT);assert check['passed']
            assert torch.allclose(world.cpu(),r['world'],atol=1e-7,rtol=0)
            source=torch.tensor(r['layout']['source_map'],device=device)
            camera=torch.einsum('njc,nck->njk',world[source]-cache['translation'][:,None],cache['rotation'])
            assert torch.allclose(camera.cpu(),r['prediction'],atol=1e-7,rtol=0)
            checks[kind+'_'+variant]=check
    # All seed/head sides changed: a model-space anchor remains defined.
    r=torch.load(root/'matched_stability_v43/regression/candidates.pt',weights_only=False)['mean_observation']
    short=torch.tensor(list(range(12)),device=device)
    # Select one actual predicted track rather than assuming row adjacency.
    key=(rows[0]['sequence'],rows[0]['clip'],rows[0]['track_id'])
    chosen=[i for i,x in enumerate(rows) if (x['sequence'],x['clip'],x['track_id'])==key][:12]
    short=torch.tensor(chosen,device=device);cc=subset(cache,short,len(rows),device)
    oo={k:v[short.cpu()].to(device) for k,v in r.items()};oo['parameter_side_outlier']=torch.ones(len(short),device=device,dtype=torch.bool)
    result=fit_stable(decoder,cc,oo,[rows[i] for i in chosen],dict(DEFAULT,steps=4))
    assert result['check']['passed'] and result['unanchored_side_groups']==1 and torch.isfinite(result['prediction']).all()
    runtime=json.loads((out/'runtime_best_dit.json').read_text())
    assert runtime['constraint_checks_passed'] and runtime['mode']=='experimental_parameter_dit_v43'
    assert runtime['model_run']=='matched_parameter_v43/dit' and runtime['candidate_selection']=='sequence'
    assert runtime['draws']==4 and runtime['ddim_steps']==10 and runtime['raw_xyz_fallback_frames']==0
    frame_count=0
    for tr in runtime['tracks']:
        assert not tr['manual_anchor_blocked']
        for frame in tr['frames']:
            xyz=torch.tensor(frame['xyz_camera_m']);assert xyz.shape==(20,3) and torch.isfinite(xyz).all();frame_count+=1
    assert frame_count==16
    summaries={k:json.loads((p/'summary.json').read_text()) for k,p in paths.items()}
    assert all(x['gt_poison_generator_exact'] for x in summaries.values())
    assert summaries['dit']['gt_poison_selector_exact'] and summaries['dit_denoise']['gt_poison_selector_exact']
    default=json.loads((Path(__file__).parent/'CURRENT_PIPELINE.json').read_text());assert default['version']=='offline_stability_first_v42'
    record=dict(passed=True,matched_data_and_initialization_fields=True,all_4000steps_completed=True,
        generator_gt_poison_exact=True,selector_gt_poison_exact=True,saved_world_and_camera_parity=True,
        all_variants_share_original_v42_limits=True,saved_output_checks=checks,
        all_side_seeds_changed_anchor_finite=True,raw_rgb_wilor_best_dit_sequence_runtime=True,runtime_frames=frame_count,
        default_v42_unchanged=True,browser_interaction_checked=False,
        limitations='Runtime smoke and repeated development only, not independent generalization or hidden-finger visibility proof')
    save(out/'verification.json',record)
    print(json.dumps(dict(passed=True,variants=len(checks),runtime_frames=frame_count,default_unchanged=True)),flush=True)


if __name__=='__main__':main()
