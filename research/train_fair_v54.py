"""Same-start, same-plan full RGB/DiT/FK adaptation, preserving frozen history."""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
from prepare_fair_v54 import ROOT,OLD,INITIAL,save

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['yolo_condition_control','rf_condition_adapt'],required=True);parser.add_argument('--pilot',action='store_true');args,remaining=parser.parse_known_args()
    folder=ROOT/'pilot'/args.arm if args.pilot else ROOT/args.arm
    if args.pilot:
        from prepare_fair_v54 import link
        folder.mkdir(parents=True,exist_ok=True)
        for name in ['native','surgical_pose','domain_masks.pt','domain_pixels.npy','mask_conditions.pt']:link(ROOT/args.arm/name,folder/name)
    assert json.loads((ROOT/'cache_done.json').read_text())['complete']
    # Modify an isolated, recorded trainer revision; never overwrite the v51
    # frozen trainer or experiments. Everything else retains the original loss.
    original=(ROOT/'code/train_full_instance_v51_r1.py').read_text()
    replacements={
      "plain=SemanticParameterHand('dit',a.device)":"plain=InstanceParameterHand('dit',a.device)",
      "initial_development_admitted=True":"initial_development_admitted=True, frontend_conditioning_admitted=False",
      "zero_branch_max_abs_m=delta":"initial_reload_max_abs_m=delta",
      "version=51":"version=54",
      "model=InstanceParameterHand('dit',a.device).to(a.device);missing":"model=InstanceParameterHand('dit',a.device).to(a.device);missing",
    }
    changed=original
    for old,new in replacements.items():
        assert changed.count(old)==1,(old,changed.count(old));changed=changed.replace(old,new)
    generated=folder/'trainer_revision_v54.py';generated.write_text(changed)
    import hashlib
    save(folder/'trainer_revision_provenance.json',dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),revision_sha256=hashlib.sha256(changed.encode()).hexdigest(),replacements=replacements,frozen_v51_source_unchanged=True))
    import types
    trainer=types.ModuleType('isolated_v54_trainer');trainer.__file__=str(generated);sys.modules[trainer.__name__]=trainer
    exec(compile(changed,str(generated),'exec'),trainer.__dict__)
    import train_full_surgical_v51_r2 as surgery
    surgery.RUN=folder;surgery.trainer=trainer
    class RFVisual(surgery.SurgicalVisual):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.overlay=None;self.overlay_index={}
            if (folder/'native/overlay_pixels.npy').exists():
                self.overlay=np.load(folder/'native/overlay_pixels.npy',mmap_mode='r');self.overlay_index={int(fid):i for i,fid in enumerate(json.loads((folder/'native/overlay_fids.json').read_text()))}
        def replace(self,b,feature_ids,pixels):
            outer=self
            class Pixels:
                def __getitem__(self,fid):
                    fid=int(fid)
                    return outer.overlay[outer.overlay_index[fid]] if fid in outer.overlay_index else pixels[fid]
            return super().replace(b,feature_ids,Pixels())
    trainer.RUN=folder;trainer.SOURCE=folder/'native';trainer.INITIAL=INITIAL
    trainer.InstanceParameterHand=surgery.SurgicalFullHand;trainer.OnlineVisual=RFVisual
    save(folder/'adaptation_protocol.json',dict(arm=args.arm,full_RGB_DiT_FK_training=True,common_original_WiLoR_seeds=True,RF_conditioning=args.arm=='rf_condition_adapt',inference_GT_free=True,unchanged_losses=True))
    sys.argv=[sys.argv[0],*remaining];trainer.main()
    if (folder/'core_r1/dit_joint/done.json').exists():
        done=json.loads((folder/'core_r1/dit_joint/done.json').read_text())
        assert (folder/'core_r1/dit_joint/verification.json').exists()
        print(json.dumps(dict(arm=args.arm,done=done)),flush=True)
if __name__=='__main__':main()
