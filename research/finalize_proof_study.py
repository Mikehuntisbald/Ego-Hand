import hashlib,json
from pathlib import Path
run=Path('/mnt/why/HOT3D/experiments/dit_lowconfidence_v1')
results=json.loads((run/'locked_selective_temporal_test_results.json').read_text())
checkpoints={
    'detector':run/'detector/weights/best.pt','coarse_3d':run/'coarse_fine/best.pt',
    'single_dit':run/'selective_dit/best.pt','single_regression':run/'selective_regression/best.pt',
    'temporal_dit':run/'temporal_selective_dit/best.pt','temporal_regression':run/'temporal_selective_regression/best.pt'}
rows={}
for name,path in checkpoints.items():
    assert path.exists() and path.stat().st_size>100000
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    rows[name]=dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())
(run/'checkpoints.json').write_text(json.dumps(rows,indent=2))
(run/'training_results.json').write_bytes((run/'locked_selective_temporal_test_results.json').read_bytes())
protocol=json.loads((run/'proof_protocol.json').read_text())
protocol.update(revision='selective_camera_space_gates_and_causal_context',
    corrections=['Shared root update replaced with per-point final XYZ gates; precise point preservation loss',
                 'Threshold/strength calibration on P0003 only; identical policy for direct regression',
                 'Added causal contexts from all YOLO proposals, no GT tracks'],
    temporal_context_seconds=[0,-1/6,-1/3,-2/3],temporal_inference='No GT camera-motion compensation',
    blinding='Initial automatic test calculation was archived unread after tune preservation failure; all subsequent choices used P0003 only')
(run/'proof_protocol.json').write_text(json.dumps(protocol,indent=2))
status=dict(stage='initial_study_complete',test_samples=results['samples'],test_subjects=results['test_subjects'],
    positive_evidence=False,verdict='Current implementation does not demonstrate solving low-confidence/occlusion/ray alignment',
    checkpoints=rows,report=str(run/'training_report.html'))
(run/'study_status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2))
