"""Verified v16 XYZ/RGB refiner; upstream handedness is prepared separately."""
import argparse,hashlib,json
from pathlib import Path
import torch
from hand3d_v8_common import V7
from hand3d_risk_v7 import Risk3D
from density_model_v13 import DensityTrajectoryHand3D
from spatial_rgb_model import SpatialHead
import spatial_rgb_common as s
import infer_hand3d_v14 as engine
RUN=V7.parent/'side_native_v16';MODEL=RUN/'consensus/uniform_adaptive/best.pt';RISK=V7.parent/'side_data_v16/consensus/risk_dense';CODE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_models(device):
    frozen=json.loads((RUN/'fifth_seal.json').read_text());assert json.loads((RUN/'fifth_results.json').read_text())['passed']
    assert sha(MODEL)==frozen['models']['consensus'] and sha(RISK/'risk_all.pt')==frozen['risk_sha256']['consensus'] and sha(RISK/'calibration.json')==frozen['temperature_sha256']['consensus']
    for name,h in frozen['code_sha256'].items():assert sha(CODE/name)==h,name
    if (RUN/'deployment_seal.json').exists():
        runtime=json.loads((RUN/'deployment_seal.json').read_text())
        for n,h in runtime['code_sha256'].items():assert sha(CODE/n)==h,n
        for p,h in runtime['asset_sha256'].items():assert sha(Path(p))==h,p
    model=DensityTrajectoryHand3D('dit',True).to(device).eval();model.load_state_dict(torch.load(MODEL,weights_only=False,map_location=device)['model'])
    ck=torch.load(RISK/'risk_all.pt',weights_only=False,map_location=device);risk=Risk3D(ck['dim']).to(device).eval();risk.load_state_dict(ck['model']);temp=torch.tensor(json.loads((RISK/'calibration.json').read_text())['temperature'],device=device)
    probe=SpatialHead().to(device).eval();probe.load_state_dict(torch.load(s.RUN/'sealed/rgb_probe.pt',weights_only=False,map_location=device)['model']);projection=torch.load(V7.parent/'natural_reliability_v4/sealed/risk_projection.pt',weights_only=False)
    full,_=s.common.load_model(device);encoder=full.backbone;del full
    return model,risk,temp,probe,projection,encoder,frozen

def infer(source,device='cuda:2'):
    engine.load_models=load_models;result=engine.infer(source,device)
    result.update(mode='offline_native3d_dit_v16',note='Fifth12unusedclips passed frozen recovery/protection gates. InputXYZ uses prediction-only temporal handedness consensus. Reusedsubjects/sourcesequences; notfullOOF. Existingboxes/tracks required. Everyunconfirmed point needs annotationreview; rawdraw spread is not calibrated correctness; severeocclusion remains imperfect.')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:2');a=p.parse_args();source=Path(a.input).resolve();dest=Path(a.output).resolve()
    result=infer(json.loads(source.read_text()),a.device);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2));print(json.dumps(dict(output=str(dest),confirmed=result['confirmed_checked'])),flush=True)

if __name__=='__main__':main()
