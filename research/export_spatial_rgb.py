"""Keep all 192 spatial cells; only reduce channels with the validated RGB head."""
import argparse,json,hashlib,time
import spatial_rgb_common as s
import torch
from spatial_rgb_model import SpatialHead

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:3');a=p.parse_args()
    torch.set_num_threads(4)
    rgb=json.loads((s.RUN/'probe_rgb/done.json').read_text())
    geom=json.loads((s.RUN/'probe_geometry/done.json').read_text())
    r=rgb['development'];g=geom['development']
    gates=dict(clean_beats_geometry_20pct=r['clean']['all_px']<.8*g['clean']['all_px'],
        partial_beats_geometry_10pct=r['partial']['all_px']<.9*g['partial']['all_px'],
        rgb_content_matters=r['shuffle_rgb']['all_px']>1.05*r['partial']['all_px'],
        spatial_order_matters=r['shuffle_space']['all_px']>1.05*r['partial']['all_px'])
    s.save(s.RUN/'rgb_localization_gate.json',dict(passed=all(gates.values()),gates=gates,rgb=r,geometry=g,test_used=False))
    assert all(gates.values()),gates
    ck=torch.load(s.RUN/'probe_rgb/best.pt',weights_only=False,map_location=a.device)
    model=SpatialHead().to(a.device).eval();model.load_state_dict(ck['model'])
    folder=s.RUN/'spatial_chunks';folder.mkdir(exist_ok=True);started=time.time()
    with torch.inference_mode():
        for path in sorted((s.RUN/'dense_chunks').glob('*.pt')):
            dest=folder/path.name
            if dest.exists():continue
            c=torch.load(path,weights_only=False,mmap=True);raw=c['features'].flatten(0,1);out=[]
            for start in range(0,len(raw),96):
                part=raw[start:start+96].to(a.device);n=len(part)
                if n<96:part=torch.cat([part,part[-1:].expand(96-n,-1,-1)])
                with torch.autocast('cuda',dtype=torch.bfloat16):x=model.features(part)
                out.append(x[:n].flatten(2).transpose(1,2).half().cpu())
            result={k:v for k,v in c.items() if k!='features'}
            result['features']=torch.cat(out).reshape(c['end']-c['start'],6,192,128)
            tmp=dest.with_suffix('.partial');torch.save(result,tmp);tmp.replace(dest)
    s.save(s.RUN/'spatial_done.json',dict(complete=True,grid=[16,12],channels=128,spatial_pooling=False,
        projection_sha256=hashlib.sha256((s.RUN/'probe_rgb/best.pt').read_bytes()).hexdigest(),seconds=time.time()-started))

if __name__=='__main__':main()
