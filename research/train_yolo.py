"""Run only when requested: baseline detector or 20-point hand pose FT."""
import argparse
from pathlib import Path
from ultralytics import YOLO
ROOT=Path('/mnt/why/HOT3D')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['detect','pose'],default='detect')
    ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--imgsz',type=int,default=960)
    ap.add_argument('--batch',type=int,default=32);ap.add_argument('--device',default='0')
    ap.add_argument('--stride',type=int,default=5);ap.add_argument('--name',default='yolo26s_hand_baseline')
    a=ap.parse_args();folder=ROOT/'export'/a.task
    import json,yaml
    status=json.loads((ROOT/'export_status.json').read_text())
    if status['stage']!='complete':raise RuntimeError('Download and label export must finish before baseline training')
    data=yaml.safe_load((folder/'data.yaml').read_text())
    # Preserve all native RGB/GT while reducing temporal duplication in FT.
    # Full-rate lists remain available for final evaluation.
    for split in ['train','val']:
        src=(folder/f'{split}.txt').read_text().splitlines()
        selected=[s for s in src if int(Path(s).stem)%a.stride==0]
        name=f'{split}_stride{a.stride}.txt';(folder/name).write_text('\n'.join(selected)+'\n');data[split]=name
    path=folder/f'data_stride{a.stride}.yaml';path.write_text(yaml.safe_dump(data,sort_keys=False))
    weight=ROOT/'weights'/('yolo26s-pose.pt' if a.task=='pose' else 'yolo26s.pt')
    YOLO(str(weight)).train(data=str(path),epochs=a.epochs,imgsz=a.imgsz,batch=a.batch,
        device=a.device,workers=8,seed=20261002,project=str(ROOT/'runs'),name=a.name,
        cache=False,degrees=0,fliplr=.5,flipud=0,mosaic=0,mixup=0,copy_paste=0,
        patience=15,close_mosaic=0,plots=True)

if __name__=='__main__':main()
