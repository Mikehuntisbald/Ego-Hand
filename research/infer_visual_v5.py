"""Raw RGB inference for the visual-tail fine-tuning trial; original risk stays frozen."""
import argparse,json,copy
from pathlib import Path
import numpy as np,torch
from infer_natural_reliability import encode_track,windows,predict_windows,load_models
from natural_corrector import NaturalCorrector
from finetune_visual_v5 import Tail,V4,RUN
import spatial_rgb_common as s

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--kind',choices=['regression','dit'],default='regression');p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4)
    source=json.loads(Path(a.input).read_text());assert source['image_size']==[1408,1408]
    old,risk,temp,probe,policy,projection=load_models(V4,a.device);del old
    ck=torch.load(RUN/f'{a.kind}_ft/best.pt',weights_only=False,map_location=a.device)
    model=NaturalCorrector(a.kind).to(a.device).eval();model.load_state_dict(ck['model'])
    full,_=s.common.load_model(a.device);encoder=full.backbone;del full
    # Deep copy only the tail modules: changing correction RGB must not alter
    # the pretrained features used by the frozen error-risk model.
    tail=copy.deepcopy(Tail(encoder,probe)).to(a.device).eval();tail.load_state_dict(ck['tail'])
    tracks=[];locked_count=0
    for track in source['tracks']:
        prefix=[]
        hook=encoder.blocks[28].register_forward_pre_hook(lambda module,args:prefix.append(args[0].detach().clone()))
        z=encode_track(track,encoder,probe,projection,a.device);hook.remove()
        tokens=torch.cat(prefix)[:len(track['frames'])];spatial=[]
        for start in range(0,len(tokens),16):
            with torch.autocast('cuda',dtype=torch.bfloat16):spatial.append(tail(tokens[start:start+16]).float().cpu())
        z['spatial']=torch.cat(spatial);w=windows(z,a.device);chunks=[]
        for start in range(0,len(track['frames']),8):
            chunks.append(predict_windows({k:v[start:start+8] for k,v in w.items()},{'regression':model},risk,temp,probe,policy))
        out={k:torch.cat([c[k] for c in chunks]) for k in chunks[0]};frames=[]
        for i,f in enumerate(track['frames']):
            pts=[]
            for j in range(20):
                locked=bool(z['confirmed'][i,j]);available=bool(z['available'][i,j]);selected=bool(out['selected'][i,j]);xy=(out['xy'][i,j]*1408).tolist()
                if locked or (available and not selected):xy=f['xy_px'][j]
                assert np.isfinite(xy).all()
                if locked:assert xy==f['xy_px'][j];locked_count+=1
                pts.append(dict(xy_px=xy,confirmed=locked,input_available=available,correction_applied=selected,review_required=not locked,input_error_risk=float(out['p_bad'][i,j]) if available else None))
            frames.append(dict(image=f['image'],timestamp_s=f['timestamp_s'],points=pts))
        tracks.append(dict(id=track.get('id'),frames=frames))
    result=dict(kind=a.kind,checkpoint_step=ck['step'],policy=policy,tracks=tracks,confirmed_points_checked=locked_count,note='Experimental joint fine-tuning; original risk encoder is unchanged; automatic outputs still require review.')
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2));print(json.dumps(dict(output=str(dest),frames=sum(len(t['frames']) for t in tracks),confirmed_checked=locked_count)))

if __name__=='__main__':main()
