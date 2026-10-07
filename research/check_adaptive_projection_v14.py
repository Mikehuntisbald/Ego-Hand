import json,torch
from hand3d_v8_common import V7,save
from adaptive_projection_v14 import apply
RUN=V7.parent/'adaptive_projection_v14/dit_dense'

def main():
    torch.manual_seed(202610116);B=256;base=torch.randn(B,20,3)*.5;raw=base+torch.randn_like(base)*20;pc=torch.rand(B,20);pr=torch.rand(B,20);confirmed=torch.rand(B,20)<.1;b=dict(base=base,risk_camera=pc,risk_relative=pr,confirmed=confirmed);policy=json.loads((RUN/'selection.json').read_text())['selected']['policy'];pred=apply(raw,b,policy);assert torch.equal(pred[confirmed],base[confirmed]);dc=(pred-base).norm(dim=-1);dr=((pred-pred[:,5:6])-(base-base[:,5:6])).norm(dim=-1);rc=torch.where(pc>=policy['camera_risk_threshold'],policy['large_cap_m'],.0095);rr=torch.where(pr>=policy['relative_risk_threshold'],policy['large_cap_m'],.0095);rc[confirmed]=0;rr[confirmed]=.0095
    assert (dc<=rc+1e-6).all() and (dr<=rr+1e-6).all();poison=apply(raw,{**b,'gt':torch.ones_like(base)*999,'gt_side':'unknown','visibility':torch.zeros(B,20)},policy);assert torch.equal(poison,pred)
    save(RUN/'projection_checks.json',dict(passed=True,extreme_cases=B,confirmed_exact=True,GT_fields_invariant=True,per_point_camera_and_relative_radii_satisfied=True,maximum_camera_displacement_mm=float(dc.max()*1000),maximum_relative_displacement_mm=float(dr.max()*1000),scope='Geometric/routing check; confidence errors are evaluated separately on frozen data'))
    print((RUN/'projection_checks.json').read_text())

if __name__=='__main__':main()
