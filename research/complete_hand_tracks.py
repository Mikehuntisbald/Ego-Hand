"""Hand annotation entry: stable estimates by default, v16 accuracy optional."""
import argparse,json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--mode',choices=['stable','accuracy'],default='stable');ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--prepared',action='store_true');a=ap.parse_args();source=json.loads(Path(a.input).read_text())
    if a.mode=='stable':
        from complete_hand_tracks_stable_v42 import complete
        result=complete(source,a.device,a.prepared)
    elif a.prepared:
        from infer_hand3d_v16 import infer
        result=infer(source,a.device)
    else:
        from complete_hand_tracks_v16 import complete
        _,result=complete(source,a.device)
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(output=str(dest),mode=a.mode)),flush=True)


if __name__=='__main__':main()
