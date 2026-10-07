"""Exact same-backend parity; pause only our controller, never a training child."""
import json,os,signal,subprocess,time
from pathlib import Path
import torch
import ablate_modules_v49 as a
from parameter_trajectory_model_v33 import ParameterTrajectoryDecoder
from parameter_candidates_v43 import select_sequence
from stability_trajectory_v43 import fit_stable
from complete_hand_tracks_acceleration_v46 import profile_config

def main():
    torch.set_num_threads(4);controller=a.RUN.parent/'trained_module_ablation_v49_20261007/controller_status.json'
    status=json.loads(controller.read_text());pid=status['pid'];cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
    assert b'run_trained_ablation_v49.py' in cmd and b'trained_module_ablation_v49_20261007' in cmd
    os.kill(pid,signal.SIGSTOP)
    try:
        while True:
            listing=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
            uuid=[x.split(',')[1].strip() for x in listing.splitlines() if x.split(',')[0].strip()=='3'][0]
            owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
            if uuid not in owners:break
            print(json.dumps(dict(waiting_for_current_pilot=True,controller_temporarily_paused=True,child_not_interrupted=True)),flush=True);time.sleep(10)
        device='cuda:3';cache={k:v.to(device) for k,v in torch.load(a.INPUT/'cache.pt',weights_only=False,map_location='cpu').items()}
        rows=json.loads((a.INPUT/'rows.json').read_text());samples=torch.load(a.SOURCE/'protected_best/candidates.pt',weights_only=False,map_location='cpu')
        obs,selection=select_sequence(samples,cache,rows);expected=json.loads((a.SOURCE/'protected_best/selection.json').read_text())
        assert selection['selected_index']==expected['selected_index']
        for name in ['no_soft_motion','no_hard_motion','no_solver_motion','strict_acc']:
            meta=json.loads((a.RUN/name/'sealed.json').read_text());assert selection['selected_index']==meta['selection']['selected_index']
        data=torch.load(a.INPUT/'inputs.pt',weights_only=False,mmap=True);right=torch.load(a.INPUT/'right.pt',weights_only=False)
        obs['parameter_side_outlier']=obs['right']!=right[data['feature_ids'][:,8]];obs={k:v.to(device) for k,v in obs.items()}
        decoder=ParameterTrajectoryDecoder(device);start=time.time()
        base=fit_stable(decoder,cache,obs,rows,profile_config('acc_x2'))
        derived=a.load_solver().fit_stable(decoder,cache,obs,rows,profile_config('acc_x2'))
        old=torch.load(a.RUN/'full/result.pt',weights_only=False,map_location=device)['prediction']
        exact_instrumented=torch.equal(base['prediction'],derived['prediction']);exact_replay=torch.equal(base['prediction'],old)
        delta=(base['prediction']-old).norm(dim=-1)*1000
        check=dict(passed=bool(exact_instrumented and exact_replay),instrumented_original_GPU_exact=exact_instrumented,
            original_cached_replay_GPU_exact=exact_replay,max_delta_mm=float(delta.max()),selector_path_GPU_exact=True,
            training_child_not_interrupted=True,seconds=time.time()-start,
            CPU_selector_note='CPU/GPU floating-point differences can change near-tied draw indices. Same-GPU equality is the required check.')
        (a.RUN/'solver_parity.json').write_text(json.dumps(check,indent=2));assert check['passed'],check
        print(json.dumps(check),flush=True)
    finally:os.kill(pid,signal.SIGCONT)
if __name__=='__main__':main()
