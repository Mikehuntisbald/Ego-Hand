"""Calibrated image -> YOLO hand boxes -> WiLoR conditions -> gated DiT XYZ.

No annotation file is read. Camera JSON is required, in official HOT3D format.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True)
    parser.add_argument('--camera',required=True,help='Official camera JSON object; no hand labels required')
    parser.add_argument('--run',default='/mnt/why/HOT3D/experiments/dit_wilor_v4')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--method',choices=['dit','regression'],default='dit')
    args=parser.parse_args()
    image_path=Path(args.image).resolve();camera_path=Path(args.camera).resolve()
    output=Path(args.output).resolve();run=Path(args.run).resolve()
    sys.path.insert(0,str(run/'sealed'))
    import wilor_eval_common as common
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from dit_v3_inference import ObservationEncoder,Refiner,prepare_observation
    torch.set_num_threads(4);cv2.setNumThreads(0)
    selection=json.loads((run/'sealed/selection.json').read_text())
    for name,digest in selection['code'].items():
        assert hashlib.sha256((run/'sealed'/name).read_bytes()).hexdigest()==digest,name
    checkpoint=run/'sealed'/f'{args.method}.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==selection['models'][args.method]['sha256']
    image=cv2.imread(str(image_path));assert image is not None,str(image_path)
    camera=json.loads(camera_path.read_text())
    detector=YOLO(str(common.ROOT/'experiments/dit_lowconfidence_v1/detector/weights/best.pt'))
    side_detector=YOLO(str(common.ROOT/'experiments/detector_compare_wilor_20261003/wilor_detector.pt'))
    started=time.perf_counter()
    p=detector.predict(image,imgsz=960,device=args.device,conf=.01,max_det=10,verbose=False)[0]
    s=side_detector.predict(image,imgsz=960,device=args.device,conf=.001,max_det=100,verbose=False)[0]
    boxes=p.boxes.xyxy.cpu().tolist();scores=p.boxes.conf.cpu().tolist()
    sides=dict(boxes=s.boxes.xyxy.cpu().tolist(),classes=s.boxes.cls.cpu().tolist(),scores=s.boxes.conf.cpu().tolist())
    del detector,side_detector;torch.cuda.empty_cache()
    records=[]
    if boxes:
        prepared=[prepare_observation(image,box,camera,sides) for box in boxes]
        cwd=os.getcwd()
        try:encoder=ObservationEncoder(args.device)
        finally:os.chdir(cwd)
        b=encoder.features(*[[p[k] for p in prepared] for k in ['wilor_crop','rotation','focal','right','side_conf','side_iou','reprojection_error','crop_fallback']])
        b.update(encoder.coarse_predictions([p['coarse_crop'] for p in prepared],[p['coarse_geometry'] for p in prepared]))
        encoder.handle.remove();del encoder;torch.cuda.empty_cache()
        refiner=Refiner(checkpoint,args.device)
        out=refiner(b)
        for i,box in enumerate(boxes):
            records.append(dict(box_xyxy=box,detector_score=scores[i],right_predicted=prepared[i]['right'],
                side_confidence=prepared[i]['side_conf'],xyz_camera_m=out['xyz'][i].cpu().tolist(),
                coarse_xyz_camera_m=out['coarse'][i].cpu().tolist(),wilor_xyz_camera_m=b['wilor'][i].cpu().tolist(),
                gate=out['gates'][i].cpu().tolist(),pose_confidence=b['confidence'][i].cpu().tolist()))
    result=dict(image=str(image_path),camera=str(camera_path),method=args.method,run=str(run),
                coordinates='canonical20 camera XYZ, meters; wrist index 5',
                detector_confidence_threshold=.01,checkpoint_sha256=selection['models'][args.method]['sha256'],
                sampling=selection['models'][args.method]['sampling'],hands=records,
                seconds_including_model_loading=time.perf_counter()-started,
                note='All detector proposals are returned, including possible false positives. No GT inputs.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(output=str(output),hands=len(records),seconds=result['seconds_including_model_loading'])))


if __name__=='__main__':main()
