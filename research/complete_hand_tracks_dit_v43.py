"""Experimental DiT branch, reusing the tested RGB/WiLoR v42 adapter.

Common RGB localization evidence is from the frozen v16 reference, as in
the development comparison. The default production entry stays v42.
"""
import argparse,json
from pathlib import Path
import torch
from hand3d_v8_common import V7
from density_model_v13 import DensityTrajectoryHand3D
from semantic_parameter_model_v36 import SemanticParameterHand
from parameter_candidates_v43 import select_sequence
from stability_trajectory_v43 import fit_stable
import complete_hand_tracks_stable_v42 as engine


class CapturedGenerator:
    def __init__(self,model,reference):self.model=model;self.reference=reference;self.batches=[]

    def generate(self,b,seed):
        encoded=self.model.encode(b);delta=self.model.sample_trajectory(b,encoded,10,4,seed)
        state=b['kinematic_coarse'][None]+delta;side=encoded[3]['side_logits'].argmax(-1)
        draws=torch.stack([self.model.codec.decode(x,side)[:,8] for x in state]);mean=state.mean(0)
        xyz=self.model.codec.decode(mean,side)
        common=self.reference.encode(b)[3]['xy'][:,8]
        self.batches.append(dict(states=state[:,:,8].float().transpose(0,1).cpu(),xyz=draws.float().transpose(0,1).cpu(),
            right=side.cpu(),initial_right=b['predicted_right'].cpu(),heat=common.float().cpu()))
        return dict(state=mean,xyz=xyz,right=side,encoded=encoded)

    def consume(self):
        batches=self.batches;self.batches=[]
        return {k:torch.cat([x[k] for x in batches]) for k in batches[0]}


def complete(source,device='cuda:0',prepared=False,model_run='matched_parameter_v43/dit',selection='sequence'):
    original_loader,original_solver=engine.load_models,engine.fit_stable
    captured=[]
    def load_models(where):
        old,risk,temp,probe,projection,encoder=original_loader(where);del old
        ck=torch.load(V7.parent/model_run/'best.pt',weights_only=False,map_location=where)
        assert ck['kind']=='dit' and 'coarse_seed' in ck['config'] and 'query_semantics' in ck['config']
        model=SemanticParameterHand('dit',where).to(where).eval();model.load_state_dict(ck['model'])
        reference=DensityTrajectoryHand3D('dit',True).to(where).eval()
        rc=torch.load(V7.parent/'side_native_v16/consensus/uniform_adaptive/best.pt',weights_only=False,map_location=where)
        reference.load_state_dict(rc['model']);wrapper=CapturedGenerator(model,reference);captured.append(wrapper)
        return wrapper,risk,temp,probe,projection,encoder
    def solve(decoder,cache,mean,rows,config):
        samples=captured[0].consume();cache=dict(cache,heat_xy=samples['heat'].to(device))
        if selection=='sequence':obs,path=select_sequence(samples,cache,rows)
        else:obs={k:v.detach().cpu() for k,v in mean.items()};path=dict(method='Parameter mean',uses_gt=False)
        obs['parameter_side_outlier']=samples['right']!=samples['initial_right']
        result=fit_stable(decoder,cache,{k:v.to(device) for k,v in obs.items()},rows,config)
        result['candidate_selection']=path
        return result
    try:
        engine.load_models=load_models;engine.fit_stable=solve
        result=engine.complete(source,device,prepared)
    finally:engine.load_models=original_loader;engine.fit_stable=original_solver
    result.update(mode='experimental_parameter_dit_v43',model_run=model_run,candidate_selection=selection,
        draws=4,ddim_steps=10,default_replaced=False,accuracy_certified=False)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--prepared',action='store_true')
    ap.add_argument('--model-run',default='matched_parameter_v43/dit');ap.add_argument('--selection',choices=['sequence','mean'],default='sequence')
    a=ap.parse_args();result=complete(json.loads(Path(a.input).read_text()),a.device,a.prepared,a.model_run,a.selection)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(output=str(out),model=a.model_run,constraints_passed=result['constraint_checks_passed'])),flush=True)


if __name__=='__main__':main()
