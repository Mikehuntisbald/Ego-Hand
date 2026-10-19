import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
R=Path('/mnt/why/native_rgb_observation_v9_20261019');sys.path.insert(0,str(R/'src'))
from rgb_head import RGBObservation
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(4);source=json.loads((R/'receipts/source_seal.json').read_text())
    for n,h in source.items():assert sha(R/'src'/n)==h
    resume=json.loads((R/'receipts/actual_resume.json').read_text());assert all(resume['exact_fields'].values())
    done={m:json.loads((R/'runs'/m/'done.json').read_text()) for m in ['spatial','pooled']};assert done['spatial']['batch_plan_sha256']==done['pooled']['batch_plan_sha256']
    for d in done.values():assert all(x>0 for x in d['gradient_sums'].values()) and all(x>0 for x in d['actual_weight_updates'].values())
    init=[]
    for mode in ['spatial','pooled']:torch.manual_seed(190019);init.append(RGBObservation(mode=mode).state_dict())
    assert all(torch.equal(init[0][k],init[1][k]) for k in init[0]);head=json.loads((R/'full_pipeline/outputs/head_clip2102/headpose.json').read_text())['frames'];assert len({h['segment'] for h in head})==1,'Motion audit would need segment masks'
    gates={}
    for mode in ['BA_spatial','BA_pooled','DiT_spatial','DiT_pooled']:
        g=json.loads((R/f'full_pipeline/outputs/ba_{mode}_guarded/optimization.json').read_text());assert g['RMSE_mm']<=5 and g['P95_mm']<=10 and g['max_mm']<=30 and g['reprojection_increase_px']<=.2;gates[mode]={k:g[k] for k in ['RMSE_mm','P95_mm','max_mm','reprojection_increase_px']}
    assert all(v==0 for v in json.loads((R/'full_pipeline/receipts/BA_parity.json').read_text())['difference_max'].values())
    out=dict(complete=True,immutable_training_sources=True,same_model_initialization_recreated=True,same_batch_plan=True,actual_gradients_and_updates=True,actual_resume=True,BA_no_new_observations_exact=True,full_replay_has_single_camera_segment=True,gates=gates,production_admitted=False)
    (R/'receipts/final_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
