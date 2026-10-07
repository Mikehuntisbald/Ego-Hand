"""Open fresh test XML only after all actual frontend predictions are sealed."""
import json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from benchmark_rfdetr_v52 import box_iou,assignments

B=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53B_20261007')

def summary(rows,pred):
    by={};details={}
    for r,p in zip(rows,pred):
        assert r['id']==p['id'];gt=[v['box_xyxy'] for v in r['hands']];mat=box_iou(gt,p['boxes']);i,j=assignments(mat);ok=np.zeros(len(gt),bool)
        for x,y in zip(i,j):ok[x]=mat[x,y]>=.5
        s=by.setdefault(r['dataset'],dict(images=0,hands=0,predictions=0,matched=0));s['images']+=1;s['hands']+=len(gt);s['predictions']+=len(p['boxes']);s['matched']+=int(ok.sum());details[r['id']]=dict(success=ok.tolist(),count=len(p['boxes']))
    for domain,s in by.items():
        s.update(recall=s['matched']/max(s['hands'],1),precision=s['matched']/max(s['predictions'],1),F1=2*s['matched']/max(s['hands']+s['predictions'],1))
        s['precision_covers_all_hands']=domain in ['egohands','100doh','hot3d_preservation']
    return by,details

def paired(rows,a,b):
    groups={};recover=lost=correct=failed=0
    for r in rows:
        x=np.asarray(a[r['id']]['success'],bool);y=np.asarray(b[r['id']]['success'],bool)
        recover+=int((~x&y).sum());lost+=int((x&~y).sum());correct+=int(x.sum());failed+=int((~x).sum())
        v=groups.setdefault(r['group'],np.zeros(5));v+=np.array([len(x),x.sum(),y.sum(),a[r['id']]['count'],b[r['id']]['count']])
    data=np.array(list(groups.values()));rng=np.random.default_rng(5308);samples=data[rng.integers(len(data),size=(10000,len(data)))].sum(1)
    recall=(samples[:,2]-samples[:,1])/np.maximum(samples[:,0],1);precision=samples[:,2]/np.maximum(samples[:,4],1)-samples[:,1]/np.maximum(samples[:,3],1)
    return dict(groups=len(data),hands=correct+failed,baseline_correct=correct,baseline_failed=failed,recovered=recover,correct_lost=lost,recall_delta_CI95=np.quantile(recall,[.025,.975]).tolist(),precision_delta_CI95=np.quantile(precision,[.025,.975]).tolist(),bootstrap='10000 source-group paired resamples')

def main():
    root=B/'replacement_eval';models=['yolo','stageA','stageB'];pred={};freezes={}
    for name in models:
        folder=root/name;freeze=json.loads((folder/'freeze.json').read_text());assert freeze['complete'] and freeze['GT_free'] and not freeze['test_XML_read']
        raw=(folder/'predictions.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==freeze['prediction_sha256'];pred[name]=json.loads(raw);freezes[name]=freeze
    rows=[r for r in map(json.loads,(B/'domain_records.jsonl').read_text().splitlines()) if r['split']=='test']
    for r in map(json.loads,(B/'fresh_test_rgb_records.jsonl').read_text().splitlines()):
        xml=ET.parse(r['annotation_xml']).getroot();w,h=r['image_size'];hands=[]
        for i,o in enumerate(xml.findall('object')):
            if o.findtext('name')!='hand':continue
            b=o.find('bndbox');x,y,z,t=[float(b.findtext(k)) for k in ['xmin','ymin','xmax','ymax']];hands.append(dict(instance_id=str(i),box_xyxy=[max(x-1,0),max(y-1,0),min(z,w),min(t,h)],difficult=int(o.findtext('difficult') or 0)))
        rows.append(dict(r,hands=hands))
    metadata=json.loads((root/'test_rgb.json').read_text())['frames'];lookup={r['id']:r for r in rows};rows=[lookup[v['id']] for v in metadata]
    stats={};details={}
    for name in models:stats[name],details[name]=summary(rows,pred[name])
    comparisons={}
    for domain in stats['yolo']:
        subset=[r for r in rows if r['dataset']==domain];comparisons[domain]=paired(subset,details['yolo'],details['stageB'])
    overlap=[]
    for r in rows:
        boxes=[v['box_xyxy'] for v in r['hands']];iou=box_iou(boxes,boxes);np.fill_diagonal(iou,0)
        if len(boxes)>1 and (iou>.1).any():overlap.append(r)
    overlap_result=paired(overlap,details['yolo'],details['stageB']) if overlap else None
    gallery=[];folder=root/'cases';folder.mkdir(exist_ok=True)
    for kind in ['recovered','harmed']:
        pool=[]
        for r in rows:
            a=np.asarray(details['yolo'][r['id']]['success'],bool);b=np.asarray(details['stageB'][r['id']]['success'],bool);n=int(((~a&b) if kind=='recovered' else (a&~b)).sum())
            if n:pool.append((n,r))
        pool.sort(key=lambda x:(-x[0],x[1]['id']))
        for k,(n,r) in enumerate(pool[:12]):
            image=Image.open(r['image']).convert('RGB');d=ImageDraw.Draw(image)
            for v in r['hands']:d.rectangle(v['box_xyxy'],outline='#00aa00',width=3)
            for name,color in [('yolo','#00baff'),('stageB','#ff00dd')]:
                p=pred[name][metadata.index(next(v for v in metadata if v['id']==r['id']))]
                for b,s in zip(p['boxes'],p['scores']):d.rectangle(b,outline=color,width=2);d.text((b[0],max(0,b[1]-12)),f'{name} {s:.2f}',fill=color)
            image.thumbnail((1280,900));path=folder/f'{kind}_{k:02d}.jpg';image.save(path,quality=93);gallery.append(dict(kind=kind,id=r['id'],dataset=r['dataset'],points=n,image='cases/'+path.name))
    out=dict(statistics=stats,paired_vs_yolo=comparisons,overlap_proxy=dict(images=len(overlap),paired=overlap_result,definition='at least two labelled hand boxes with IoU>0.1; not finger occlusion GT'),freezes=freezes,gallery=gallery,fresh_domain='100doh',fresh_groups=400,other_test_domains='previous v51/v52 test replay; diagnostic',selection=json.loads((root/'selection.json').read_text()),test_used_for_training=False,default_replaced=False,GT_3D_absent=True)
    (root/'statistics.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(statistics=stats,paired=comparisons,overlap=out['overlap_proxy'])),flush=True)

if __name__=='__main__':main()
