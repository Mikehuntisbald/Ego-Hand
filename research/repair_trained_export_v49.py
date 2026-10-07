"""Repair missing export metadata, preserving frozen source and sealed outputs."""
import datetime,hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path('/mnt/why/HOT3D/experiments/trained_module_ablation_v49_20261007')
def main():
    state=json.loads((ROOT/'controller_status.json').read_text());assert state['stage']=='needs_attention' and state['task']=='evaluation'
    error=ROOT/('export_failure_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));error.mkdir()
    for name in ['controller_status.json','evaluation.log','code_hashes.json','freeze.json']:
        if (ROOT/name).exists():shutil.copy2(ROOT/name,error/name)
    for name in ['evaluate_trained_ablation_v49.py','ablate_modules_v49.py']:
        shutil.copy2(ROOT/'code_snapshot'/name,error/name)
    arms=['full','no_teacher','frozen_same_start','no_original_protection','no_protection','no_motion_supervision','no_rgb','center_only','pooled_rgb']
    configs={a:json.loads((ROOT/a/'config.json').read_text()) for a in arms}
    assert len({x['initial_checkpoint_sha256'] for x in configs.values()})==1 and len({x['batch_plan_sha256'] for x in configs.values()})==1
    seals={a:json.loads((ROOT/a/'sealed.json').read_text()) for a in arms}
    for a in arms:
        assert seals[a]['complete'] and hashlib.sha256((ROOT/a/'result.pt').read_bytes()).hexdigest()==seals[a]['result_sha256']
        assert json.loads((ROOT/a/'done.json').read_text())['steps']==1800
    protocol=dict(variants=arms,fixed='Same joint600 warmstart, same batchplan,1800 continuation steps,effectivebatch4; final checkpoints fixed before replay.',
        initial_checkpoint_sha256=configs['full']['initial_checkpoint_sha256'],batch_plan_sha256=configs['full']['batch_plan_sha256'],
        scope='10 previously used clips,5sequences,2known actors; no new generalization claim.',
        reconstructed_export_metadata=True,original_design='DEVELOPMENT_V49.md plus preexisting immutable per-arm config/batchplan files',
        code_unchanged=True,inference_outputs_unchanged=True,default_changed=False,
        failure=str(error),training_ablation='Warmstart adaptation, not from-scratch; equal steps/exposure,not equal GPU compute.')
    assert not (ROOT/'protocol.json').exists();(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with (ROOT/'evaluation_repair.log').open('w') as f:
        subprocess.run([sys.executable,str(ROOT/'code_snapshot/evaluate_trained_ablation_v49.py')],cwd=ROOT/'code_snapshot',stdout=f,stderr=subprocess.STDOUT,check=True)
    for a in arms:assert hashlib.sha256((ROOT/a/'result.pt').read_bytes()).hexdigest()==seals[a]['result_sha256']
    state.update(stage='complete',training_complete=True,evaluation_complete=True,repair_evidence=str(error),repair='Missing export protocol reconstructed from immutable configs; original code/output unchanged',review=str(ROOT/'review/report.html'))
    (ROOT/'controller_status.json').write_text(json.dumps(state,indent=2));print(json.dumps(dict(complete=True,outputs_unchanged=True,evidence=str(error))),flush=True)
if __name__=='__main__':main()
