"""New metadata-complete artifact; original checkpoint is never edited."""
import json,hashlib,copy
from pathlib import Path
import torch
from rfdetr import RFDETR
A=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53_20261007')
def main():
    torch.set_num_threads(4)
    ref=torch.load('/mnt/why/HOT3D/experiments/rfdetr_hand_instance_v52_20261007/full/checkpoint_best_total.pth',map_location='cpu',weights_only=False)
    folder=A/'checkpoint_metadata_fix';folder.mkdir(exist_ok=True);results=[]
    for filename in ['best_admitted_ema.pth','best_trained_ema.pth']:
        original=A/'full'/filename;target=folder/filename;state=torch.load(original,map_location='cpu',weights_only=False)
        for key in ['model_name','model_config','rfdetr_version']:state[key]=copy.deepcopy(ref[key])
        torch.save(state,target)
        reloaded=RFDETR.from_checkpoint(str(target),device='cpu',trust_checkpoint=True)
        actual=reloaded.model.model.state_dict();assert set(actual)==set(state['model'])
        assert all(torch.equal(actual[k],v) for k,v in state['model'].items()), 'Checkpoint reload changed tensor values'
        results.append(dict(original=str(original),original_sha256=hashlib.file_digest(original.open('rb'),'sha256').hexdigest(),repaired=str(target),repaired_sha256=hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),all_tensors_equal_after_reload=True,model_name=state['model_name'],resolution=state['model_config']['resolution']))
        del reloaded
    (folder/'verification.json').write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)
if __name__=='__main__':main()
