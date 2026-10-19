"""New objective snapshot; preserve original completed risk-penalty experiment."""
import hashlib,json
from pathlib import Path
R=Path('/mnt/why/native_candidate_risk_v11_20261019');REV=R/'error_only_r1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert json.loads((R/'controller_status.json').read_text())['complete'];assert not (R/'evaluation').exists();REV.mkdir(exist_ok=False)
    for n in ['src','receipts','logs','runs']:(REV/n).mkdir()
    source=(R/'src/risk_model.py').read_text();a="score=delta.mean(-1)/5+2*probability.mean(-1)";assert source.count(a)==1;source=source.replace(a,"score=delta.mean(-1)/5");a="gt_score=delta.mean(-1)/5+2*harm.mean(-1)";assert source.count(a)==1;source=source.replace(a,"gt_score=delta.mean(-1)/5");compile(source,'risk_model_r1.py','exec');(REV/'src/risk_model_r1.py').write_text(source)
    source=(R/'src/train_quality.py').read_text();assert source.count("R=Path('/mnt/why/native_candidate_risk_v11_20261019')")==1;source=source.replace("R=Path('/mnt/why/native_candidate_risk_v11_20261019')",f"R=Path('{REV}')").replace('from risk_model import','from risk_model_r1 import');compile(source,'train_quality_r1.py','exec');(REV/'src/train_quality_r1.py').write_text(source)
    protocol=dict(complete=True,GT_scope='Original train and fixed dev-select only; no new six-clip/full/fresh scoring labels opened',reason='Original learned difference-only argmin chose9 alternatives/25 dev frames for RGB, composite heuristic chose1; risk penalty in score was overly conservative',change='Deployment and GT listwise target use predicted mean native relative3D error-difference/5mm only; retain harm/good auxiliary learning and RGB_preserve loss-only expected-harm. All output gates unchanged.',same_architecture_init_data_batches_updates=True,updates_each=600,arms=['pose_base','RGB_base','RGB_preserve'],original_source_sha256={n:sha(R/'src'/n) for n in ['risk_model.py','train_quality.py','training_data.py','input_data.py']},source_input_freeze_sha256=sha(R/'data/inputs_frozen.json'),source_target_sha256=sha(R/'data/quality_targets.pt'),production_admitted=False)
    (REV/'protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(dict(complete=True,new_version=True,updates_each=600,output_gates_unchanged=True)))
if __name__=='__main__':main()
