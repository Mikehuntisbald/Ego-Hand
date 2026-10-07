"""Paired prediction-only front ends, sealed arrays before held-out labels."""
import json, hashlib, collections, argparse
from pathlib import Path
import numpy as np, cv2, torch
from scipy.optimize import linear_sum_assignment
ROOT=Path('/mnt/why/HOT3D/experiments/rfdetr_hand_instance_v52_20261007')
PAIRED=Path('/mnt/why/HOT3D/experiments/full_model_gloves_multihand_v51_20261007/paired_protocol')

def box_iou(a,b):
    a=np.asarray(a).reshape(-1,4);b=np.asarray(b).reshape(-1,4)
    inter=np.maximum(np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]),0).prod(-1)
    return inter/np.maximum(np.maximum(a[:,2:]-a[:,:2],0).prod(-1)[:,None]+np.maximum(b[:,2:]-b[:,:2],0).prod(-1)[None]-inter,1)

def assignments(iou):
    if 0 in iou.shape:return [],[]
    x,y=linear_sum_assignment(-((iou>=.5).astype(float)+iou*1e-3));return x,y

def select():
    pred=json.loads((ROOT/'dev_predictions/predictions.json').read_text());rows={r['id']:r for r in map(json.loads,(PAIRED/'domain_records.jsonl').read_text().splitlines())};scores=[]
    for threshold in [.05,.1,.15,.2,.25,.3,.4,.5,.6,.7,.8]:
        tp=npred=ngt=0
        for f in pred['frames']:
            d=np.load(f['output']);boxes=d['boxes'][d['scores']>=threshold];gt=[h['box_xyxy'] for h in rows[f['id']]['hands']];ov=box_iou(gt,boxes);x,y=assignments(ov);tp+=sum(ov[i,j]>=.5 for i,j in zip(x,y));npred+=len(boxes);ngt+=len(gt)
        scores.append(dict(threshold=threshold,matched=int(tp),predicted=npred,GT_hands=ngt,recall=tp/ngt,precision=tp/max(npred,1),F1=2*tp/max(npred+ngt,1)))
    chosen=max(scores,key=lambda x:(x['F1'],x['recall']));(ROOT/'threshold_selection.json').write_text(json.dumps(dict(selected=chosen,curve=scores,rule='Max development box F1, recall tie break; fixed before test scoring'),indent=2));print(json.dumps(chosen),flush=True)

def baseline():
    from ultralytics import YOLO
    from instance_masks_v51_r1 import HandInstanceSegmenter
    torch.set_num_threads(4);cv2.setNumThreads(0)
    data=json.loads((ROOT/'test_predictions/predictions.json').read_text());groups=collections.defaultdict(list)
    for f in data['frames']:groups[f['group']].append(f)
    selected=[]
    for g,rows in sorted(groups.items()):selected+=sorted(rows,key=lambda f:hashlib.sha256(('mask52:'+f['id']).encode()).hexdigest())[:20]
    threshold=json.loads((ROOT/'threshold_selection.json').read_text())['selected']['threshold'];folder=ROOT/'mask_baseline';folder.mkdir(exist_ok=True)
    model=YOLO(json.loads((PAIRED/'detector/selection.json').read_text())['selected'])
    out=model.predict([f['image'] for f in selected],imgsz=960,conf=.05,iou=.7,max_det=20,batch=8,device='0',verbose=False)
    boxes=[p.boxes.xyxy.cpu().numpy() for p in out];conf=[p.boxes.conf.cpu().numpy() for p in out];del model;torch.cuda.empty_cache();sam=HandInstanceSegmenter('cuda:0');records=[]
    for i,(f,b,s) in enumerate(zip(selected,boxes,conf)):
        image=cv2.imread(f['image']);d=np.load(f['output']);rb=d['boxes'][d['scores']>=threshold]
        outputs=sam(image,b);rm=sam(image,rb)
        h,w=image.shape[:2]
        raw=np.stack([p['mask'] for p in outputs]) if outputs else np.zeros((0,h,w),bool)
        same=np.stack([p['mask'] for p in rm]) if rm else np.zeros((0,h,w),bool)
        mapped=np.zeros((len(b),h,w),bool);ov=box_iou(b,d['boxes'])
        if ov.size:
            x,y=linear_sum_assignment(-ov)
            for bi,ri in zip(x,y):
                if ov[bi,ri]>=.25:mapped[bi]=d['masks'][ri]
        mapped &= ~(mapped.sum(0)>1)[None]
        path=folder/f'{i:04d}.npz';np.savez_compressed(path,boxes=b,scores=s,masks=raw,rf_boxes_sam_masks=same,yolo_rf_masks=mapped)
        records.append(dict(f,baseline=str(path)))
        if i%20==0:print('MASK_BASELINE',i,flush=True)
    (folder/'predictions.json').write_text(json.dumps(dict(frames=records,RF_threshold=threshold,GT_free=True,mask_overlap_policy='shared pixels unknown',SAM_box_prompts='predicted boxes only')))
    (folder/'freeze.json').write_text(json.dumps(dict(complete=True,sha256={Path(f['baseline']).name:hashlib.file_digest(open(f['baseline'],'rb'),'sha256').hexdigest() for f in records})))

def score_masks(gt,pred):
    inter=np.array([[np.logical_and(a,b).sum() for b in pred] for a in gt],float).reshape(len(gt),len(pred));ga=gt.sum((1,2)) if len(gt) else np.array([]);pa=pred.sum((1,2)) if len(pred) else np.array([])
    iou=inter/np.maximum(ga[:,None]+pa[None]-inter,1);x,y=assignments(iou);matched=[float(iou[a,b]) for a,b in zip(x,y) if iou[a,b]>=.5];success=np.zeros(len(gt),bool);value=np.zeros(len(gt))
    for a,b in zip(x,y):success[a]=iou[a,b]>=.5;value[a]=iou[a,b]
    merge=int(((inter/np.maximum(ga[:,None],1)>=.5).sum(0)>=2).sum())
    return dict(hands=len(gt),predictions=len(pred),matched=len(matched),sum_iou=float(value.sum()),merged_predictions=merge,success=success.tolist(),assigned_iou=value.tolist())

def score():
    baseline=json.loads((ROOT/'mask_baseline/predictions.json').read_text());assert (ROOT/'mask_baseline/freeze.json').exists() and (ROOT/'test_predictions/freeze.json').exists()
    rows={r['id']:r for r in map(json.loads,(PAIRED/'domain_records.jsonl').read_text().splitlines())};results=[];threshold=baseline['RF_threshold']
    for f in baseline['frames']:
        gt=np.load(rows[f['id']]['mask_gt'])['masks'].astype(bool);d=np.load(f['output']);dmask=d['masks'][d['scores']>=threshold];ref=np.load(f['baseline']);crowded=bool((box_iou([h['box_xyxy'] for h in rows[f['id']]['hands']],[h['box_xyxy'] for h in rows[f['id']]['hands']])-np.eye(len(gt))>.1).any())
        exclusive=dmask&~(dmask.sum(0)>1)[None] if len(dmask) else dmask
        raw=score_masks(gt,exclusive);same=score_masks(gt,ref['rf_boxes_sam_masks']);old=score_masks(gt,ref['masks'])
        results.append(dict(id=f['id'],group=f['group'],crowded_bbox_proxy=crowded,RFDETR=raw,RFDETR_raw=score_masks(gt,dmask),SAM_same_RF_boxes=same,YOLO_SAM=old,YOLO_RF=score_masks(gt,ref['yolo_rf_masks'])))
    totals={}
    for method in ['RFDETR','RFDETR_raw','SAM_same_RF_boxes','YOLO_SAM','YOLO_RF']:
        sums={k:sum(r[method][k] for r in results) for k in ['hands','predictions','matched','sum_iou','merged_predictions']};sums.update(mask_recall_IoU50=sums['matched']/sums['hands'],mask_precision_IoU50=sums['matched']/max(sums['predictions'],1),mean_assigned_mask_IoU=sums['sum_iou']/sums['hands']);totals[method]=sums
    intervals={};rng=np.random.default_rng(52);groups=sorted(set(r['group'] for r in results))
    for ref in ['SAM_same_RF_boxes','YOLO_SAM']:
        delta=[sum(r['RFDETR']['matched']-r[ref]['matched'] for r in results if r['group']==g)/sum(r['RFDETR']['hands'] for r in results if r['group']==g) for g in groups]
        before=np.array([v for r in results for v in r[ref]['success']]);after=np.array([v for r in results for v in r['RFDETR']['success']]);intervals[ref]=dict(groups=len(groups),recall_delta_CI95=np.quantile(rng.choice(delta,(5000,len(delta))).mean(1),[.025,.975]).tolist(),previous_correct=int(before.sum()),lost_correct=int((before&~after).sum()),previous_failures=int((~before).sum()),recovered=int((~before&after).sum()))
    delta=[sum(r['YOLO_RF']['matched']-r['YOLO_SAM']['matched'] for r in results if r['group']==g)/sum(r['YOLO_RF']['hands'] for r in results if r['group']==g) for g in groups]
    before=np.array([v for r in results for v in r['YOLO_SAM']['success']]);after=np.array([v for r in results for v in r['YOLO_RF']['success']]);intervals['same_YOLO_boxes_RF_vs_SAM']=dict(groups=len(groups),recall_delta_CI95=np.quantile(rng.choice(delta,(5000,len(delta))).mean(1),[.025,.975]).tolist(),previous_correct=int(before.sum()),lost_correct=int((before&~after).sum()),previous_failures=int((~before).sum()),recovered=int((~before&after).sum()))
    # Whole 1200-image RF box coverage, fixed development threshold.
    test=json.loads((ROOT/'test_predictions/predictions.json').read_text());tp=npred=ngt=0
    for f in test['frames']:
        d=np.load(f['output']);gt=[h['box_xyxy'] for h in rows[f['id']]['hands']];b=d['boxes'][d['scores']>=threshold];ov=box_iou(gt,b);x,y=assignments(ov);tp+=sum(ov[i,j]>=.5 for i,j in zip(x,y));npred+=len(b);ngt+=len(gt)
    result=dict(mask_statistics=totals,paired_comparisons=intervals,images=len(results),box_test=dict(images=len(test['frames']),hands=ngt,predictions=npred,matched=int(tp),recall=float(tp/ngt),precision=float(tp/max(npred,1))),threshold=threshold,GT_3D=False,test_is_reused_v51_diagnostic=True,no_default_replacement=True)
    (ROOT/'mask_evaluation.json').write_text(json.dumps(result,indent=2));(ROOT/'mask_per_image.json').write_text(json.dumps(results));print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['select','baseline','score']);a=p.parse_args();globals()[a.action]()
