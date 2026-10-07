"""Check the instrumented solver with all modules enabled before interpreting deltas."""
import json,time
import torch
import ablate_modules_v49 as a
from parameter_trajectory_model_v33 import ParameterTrajectoryDecoder
from parameter_candidates_v43 import select_sequence
from stability_trajectory_v43 import fit_stable
from complete_hand_tracks_acceleration_v46 import profile_config

def main():
    torch.set_num_threads(4);cache=torch.load(a.INPUT/'cache.pt',weights_only=False,map_location='cpu');rows=json.loads((a.INPUT/'rows.json').read_text())
    samples=torch.load(a.SOURCE/'protected_best/candidates.pt',weights_only=False,map_location='cpu');obs,selection=select_sequence(samples,cache,rows)
    expected=json.loads((a.SOURCE/'protected_best/selection.json').read_text());assert selection['selected_index']==expected['selected_index']
    data=torch.load(a.INPUT/'inputs.pt',weights_only=False,mmap=True);right=torch.load(a.INPUT/'right.pt',weights_only=False)
    obs['parameter_side_outlier']=obs['right']!=right[data['feature_ids'][:,8]]
    decoder=ParameterTrajectoryDecoder('cpu');start=time.time()
    baseline=fit_stable(decoder,cache,obs,rows,profile_config('acc_x2'))
    derived=a.load_solver().fit_stable(decoder,cache,obs,rows,profile_config('acc_x2'))
    assert torch.equal(baseline['prediction'],derived['prediction'])
    gpu=torch.load(a.RUN/'full/result.pt',weights_only=False,map_location='cpu')['prediction']
    delta=(baseline['prediction']-gpu).norm(dim=-1)*1000
    check=dict(passed=bool(delta.max()<.02),instrumented_original_CPU_exact=True,selector_path_matches_original=True,
        CPU_GPU_mean_delta_mm=float(delta.mean()),CPU_GPU_max_delta_mm=float(delta.max()),tolerance_mm=.02,seconds=time.time()-start)
    (a.RUN/'solver_parity.json').write_text(json.dumps(check,indent=2));assert check['passed'],check
    print(json.dumps(check),flush=True)
if __name__=='__main__':main()
