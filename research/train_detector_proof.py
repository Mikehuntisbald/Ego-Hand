"""YOLO26s hand detector. P0003 tunes; P0010/P0015 never select checkpoints."""
import json,random
from pathlib import Path
import yaml
from ultralytics import YOLO
ROOT=Path('/mnt/why/HOT3D');RUN=ROOT/'experiments/dit_lowconfidence_v1'

def main():
    RUN.mkdir(parents=True,exist_ok=True)
    folder=ROOT/'export/detect';data=yaml.safe_load((folder/'data.yaml').read_text())
    lines=(folder/'train.txt').read_text().splitlines();train=[p for p in lines if int(Path(p).stem)%5==0]
    val=[p for p in (folder/'val.txt').read_text().splitlines() if '/P0003_' in p and int(Path(p).stem)%10==0]
    assert len(train)==11400 and val
    (RUN/'detector_train.txt').write_text('\n'.join(train)+'\n');(RUN/'detector_tune.txt').write_text('\n'.join(val)+'\n')
    data.update(train=str(RUN/'detector_train.txt'),val=str(RUN/'detector_tune.txt'))
    path=RUN/'detector_data.yaml';path.write_text(yaml.safe_dump(data,sort_keys=False))
    YOLO(str(ROOT/'weights/yolo26s.pt')).train(data=str(path),epochs=40,imgsz=960,batch=64,
        device='0',workers=8,seed=20261002,project=str(RUN),name='detector',exist_ok=True,
        optimizer='AdamW',lr0=.0007,lrf=.1,weight_decay=.0005,warmup_epochs=2,cache='ram',
        degrees=0,fliplr=.5,flipud=0,mosaic=0,mixup=0,copy_paste=0,patience=10,
        close_mosaic=0,plots=True,save_period=5,amp=True)
    (RUN/'detector_done.json').write_text(json.dumps(dict(completed=True,checkpoint=str(RUN/'detector/weights/best.pt'))))

if __name__=='__main__':main()
