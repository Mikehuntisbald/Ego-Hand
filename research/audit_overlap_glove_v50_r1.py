"""Post-hoc real-scene overlap audit; cached inference only, no glove assumption."""
import collections,hashlib,html,json
from pathlib import Path
import cv2,numpy as np,torch
from hand3d_v8_common import V7,metrics,save
from calibrate_hand3d_v8 import paired_ci
from evaluate_joint_kinematic_v30 import identities,EDGES

ROOT=V7.parent;RUN=ROOT/'overlap_glove_audit_v50_20261007';REPLAY=ROOT/'acceleration_validation_v46'

def bbox(hand,size):
    if hand['box_amodal_xyxy'] is None or sum(hand['keypoint_projection_valid'])<2:return None
    b=np.asarray(hand['box_amodal_xyxy'],dtype=float).copy();w,h=size;b[[0,2]]=np.clip(b[[0,2]],0,w);b[[1,3]]=np.clip(b[[1,3]],0,h)
    if np.prod(np.maximum(0,b[2:]-b[:2]))<256:return None
    return b
def overlap(a,b):
    inter=float(np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))))
    aa=float(np.prod(a[2:]-a[:2]));bb=float(np.prod(b[2:]-b[:2]));return inter/max(min(aa,bb),1e-9),inter/max(aa+bb-inter,1e-9)
def main():
    torch.set_num_threads(4);RUN.mkdir(parents=True,exist_ok=True);out=RUN/'review';out.mkdir(exist_ok=True)
    src=ROOT/'module_ablation_v49_20261007';predictions={};seals={}
    for name in ['full','wilor']:
        meta=json.loads((src/name/'sealed.json').read_text());assert meta['complete'];assert hashlib.sha256((src/name/'result.pt').read_bytes()).hexdigest()==meta['result_sha256']
        predictions[name]=torch.load(src/name/'result.pt',weights_only=False,map_location='cpu')['prediction'];seals[name]=meta['result_sha256']
    save(RUN/'inference_provenance.json',dict(existing_sealed_predictions_only=True,sha256=seals,no_new_inference=True,no_default_change=True))
    rows=json.loads((REPLAY/'fresh_rows.json').read_text());cache=torch.load(ROOT/'online_rgb_3d_v47/diagnostic_v46/cache.pt',weights_only=False,map_location='cpu')
    windows=torch.load(REPLAY/'fresh_data.pt',weights_only=False,mmap=True)['rows'];centers={r['row_index'] for r in windows}
    gt=torch.zeros(len(rows),20,3);valid=torch.zeros(len(rows),20,dtype=torch.bool)
    for i,r in enumerate(rows):
        if r['matched']:gt[i]=torch.tensor(r['gt']);valid[i]=torch.isfinite(gt[i]).all(-1)
    sides=identities(rows,dict(gt=gt,valid=valid));files={};frames={};byframe=collections.defaultdict(list)
    for i,r in enumerate(rows):byframe[r['image']].append(i)
    # Include images with zero detections, so detection coverage isn't conditional.
    manifest=json.loads((REPLAY/'fresh_manifest.json').read_text())
    for seq in manifest['sequences']:
        sp=seq['split'];sn=seq['sequence']
        for clip in seq['clips']:
            path=ROOT.parent/'export/annotations'/sp/sn/(f'clip-{clip:06d}.jsonl')
            records=[json.loads(x) for x in path.read_text().splitlines()]
            for rec in records:
                image=str(ROOT.parent/'export'/rec['image']) if not Path(rec['image']).is_absolute() else rec['image'];boxes=[(h['side'],bbox(h,rec['image_size'])) for h in rec['hands']];boxes=[(s,b) for s,b in boxes if b is not None]
                ioa=iou=0.
                for a in range(len(boxes)):
                    for b in range(a+1,len(boxes)):
                        q,t=overlap(boxes[a][1],boxes[b][1]);ioa=max(ioa,q);iou=max(iou,t)
                category='single_projected_hand' if len(boxes)<2 else 'two_separate' if ioa<.1 else 'two_partial_overlap' if ioa<.5 else 'two_strong_overlap'
                frames[image]=dict(category=category,ioa=ioa,iou=iou,boxes=boxes,rec=rec)
    for r in rows:assert r['image'] in frames
    stats={};cm=torch.tensor([i in centers for i in range(len(rows))]);frame_counts=collections.Counter(f['category'] for f in frames.values())
    for category in ['single_projected_hand','two_separate','two_partial_overlap','two_strong_overlap']:
        images=[p for p,f in frames.items() if f['category']==category];indices=[i for i,r in enumerate(rows) if frames[r['image']]['category']==category]
        center_ids=torch.tensor([i for i in indices if cm[i] and valid[i].all()],dtype=torch.long)
        expected=sum(len(frames[p]['boxes']) for p in images);matched=0;false_dets=0
        for p in images:
            present={sides[i] for i in byframe[p] if sides[i] is not None};matched+=len(present & {s for s,b in frames[p]['boxes']})
            false_dets+=sum(sides[i] is None for i in byframe[p])
        m={n:metrics(p[center_ids],predictions['wilor'][center_ids],gt[center_ids],valid[center_ids]) for n,p in predictions.items()} if len(center_ids) else {}
        seqs=sorted({rows[i]['sequence'] for i in center_ids.tolist()})
        ci=paired_ci(predictions['full'][center_ids],predictions['wilor'][center_ids],gt[center_ids],valid[center_ids],[rows[i] for i in center_ids.tolist()]) if len(seqs)>=2 else None
        stats[category]=dict(frames=len(images),detector_observations=len(indices),GT_projected_hand_instances=expected,matched_hand_instances=matched,
            projected_hand_match_coverage=matched/expected if expected else None,unmatched_detector_observations=false_dets,
            matched_centers=len(center_ids),sequences=seqs,metrics=m,paired_CI_full_vs_wilor=ci,
            denominator_note='GT boxes clipped toimage,atleast2 projected keypoints andarea256px;includes occluded projected hands,not pixel-visible-hand recall.')
    # ID audit separates track changes from previously published within-track switches.
    tracks=collections.defaultdict(list);physical=collections.defaultdict(list)
    for i,r in enumerate(rows):
        if sides[i] is not None:
            tracks[(r['sequence'],r['clip'],r['track_id'])].append(i);physical[(r['sequence'],r['clip'],sides[i])].append(i)
    switches=[];fragments=[]
    for key,ix in tracks.items():
        ix.sort(key=lambda i:rows[i]['frame'])
        for a,b in zip(ix,ix[1:]):
            if rows[b]['frame']==rows[a]['frame']+1 and sides[a]!=sides[b]:switches.append(dict(track=list(key),a=a,b=b))
    for key,ix in physical.items():
        ix.sort(key=lambda i:rows[i]['frame'])
        for a,b in zip(ix,ix[1:]):
            if rows[b]['frame']==rows[a]['frame']+1 and rows[a]['track_id']!=rows[b]['track_id']:
                fragments.append(dict(GT_hand=list(key),a=a,b=b,category=frames[rows[b]['image']]['category']))
    examples=[]
    candidates=sorted([p for p,f in frames.items() if f['category']=='two_strong_overlap'],key=lambda p:frames[p]['ioa'],reverse=True)
    used=set()
    for image in candidates:
        f=frames[image];seq=f['rec']['sequence']
        if seq in used:continue
        used.add(seq);img=cv2.imread(image);h,w=img.shape[:2]
        boxes=np.stack([b for s,b in f['boxes']]);lo=boxes[:,:2].min(0);hi=boxes[:,2:].max(0);center=(lo+hi)/2;size=min(max(float(max(hi-lo)*1.5),220.),min(w,h));x=int(np.clip(center[0]-size/2,0,w-size));y=int(np.clip(center[1]-size/2,0,h-size));size=int(size)
        panels=[]
        for mode,label in [('wilor','WiLoR'),('full','full'),('GT','GT')]:
            panel=img.copy()
            if mode=='GT':values=[(hand['side'],np.asarray(hand['uv_pixels']),None) for hand in f['rec']['hands'] if bbox(hand,f['rec']['image_size']) is not None]
            else:
                from spatial_rgb_common import common
                values=[(sides[i] or 'unmatched',common.from_json(rows[i]['camera']).eye_to_window(predictions[mode][i].numpy()),rows[i]['track_id']) for i in byframe[image]]
            for side,uv,tid in values:
                color=(30,220,70) if side=='left' else (0,160,255) if side=='right' else (160,160,160)
                for a,b in EDGES:
                    if np.isfinite(uv[[a,b]]).all():cv2.line(panel,tuple(np.clip(uv[a],-4000,4000).astype(int)),tuple(np.clip(uv[b],-4000,4000).astype(int)),color,3)
                if tid is not None and np.isfinite(uv[5]).all():cv2.putText(panel,str(tid),tuple(uv[5].astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
            panel=cv2.resize(panel[y:y+size,x:x+size],(480,480));cv2.rectangle(panel,(0,0),(480,35),(20,20,20),-1);cv2.putText(panel,label,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2);panels.append(panel)
        file=f'overlap_{len(examples)}.jpg';cv2.imwrite(str(out/file),np.concatenate(panels,1));examples.append(dict(sequence=seq,frame=f['rec']['frame'],image=file,ioa=f['ioa'],iou=f['iou'],detected=len(byframe[image]),GT_hands=len(f['boxes'])))
        if len(examples)>=5:break
    # A sampled visual inventory is evidence for inspection, not a glove label.
    clips=collections.defaultdict(list)
    for p,f in frames.items():clips[(f['rec']['sequence'],f['rec']['clip'])].append(p)
    sheet=[];sampled=[]
    for key,paths in sorted(clips.items()):
        paths.sort(key=lambda p:frames[p]['rec']['frame']);cells=[]
        for j in sorted(set(np.linspace(0,len(paths)-1,4).round().astype(int))):
            p=paths[j];f=frames[p];img=cv2.imread(p);h,w=img.shape[:2]
            if f['boxes']:
                bb=np.stack([b for s,b in f['boxes']]);lo=bb[:,:2].min(0);hi=bb[:,2:].max(0);center=(lo+hi)/2;size=min(max(float(max(hi-lo)*1.4),200.),min(w,h));x=int(np.clip(center[0]-size/2,0,w-size));y=int(np.clip(center[1]-size/2,0,h-size));size=int(size);img=img[y:y+size,x:x+size]
            img=cv2.resize(img,(320,320));cv2.rectangle(img,(0,0),(320,25),(0,0,0),-1);cv2.putText(img,f'{key[0]} {key[1]} f{f["rec"]["frame"]}',(4,18),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1);cells.append(img);sampled.append(p)
        while len(cells)<4:cells.append(np.zeros_like(cells[0]))
        sheet.append(np.concatenate(cells,1))
    for k in range(0,len(sheet),5):cv2.imwrite(str(out/f'coverage_{k//5}.jpg'),np.concatenate(sheet[k:k+5],0))
    summary=dict(complete=True,strata=stats,overlap_definition='Intersection divided by smaller GT amodal projected boundingbox; separate<0.1,partial0.1-0.5,strong>=0.5. Box proxy,not pixel occlusion truth.',
        frames=len(frames),max_GT_annotated_hands=max(len(f['rec']['hands']) for f in frames.values()),
        matched_ID_switches=switches,consecutive_GThand_predictedID_changes=fragments,
        glove_annotation_available=False,glove_performance_validated=False,visual_inventory_samples=len(sampled),sample_images=sampled,examples=examples,
        scope='Already evaluated10clips/5sequences/2actors;GT contains wearer left/right hands only,not crowded multiperson identity annotations.',
        metrics_conditional_on_matched_centers=True,no_glove_claim_from_absence_of_metadata=True,inference_inputs_unchanged=True)
    save(RUN/'summary.json',summary);save(out/'summary.json',summary)
    table=''
    labels=dict(single_projected_hand='单手有效投影',two_separate='双手框分开',two_partial_overlap='双手框部分重叠',two_strong_overlap='双手框强重叠')
    for n,x in stats.items():
        ms=x['metrics'];mm=lambda mode,key:f"{ms[mode][key]:.2f}" if ms else '—';ci=x['paired_CI_full_vs_wilor'];c=ci['relative']['ci95_delta_mm'] if ci else None
        table+=f"<tr><td>{labels[n]}</td><td>{x['frames']}</td><td>{x['matched_centers']}</td><td>{x['matched_hand_instances']}/{x['GT_projected_hand_instances']}</td><td>{mm('wilor','relative_mm')}</td><td>{mm('full','relative_mm')}</td><td>{html.escape(str(c))}</td></tr>"
    panels=''.join(f"<p>{html.escape(str(x))}</p><img src='{x['image']}' style='max-width:100%'>" for x in examples)
    page=f"""<!doctype html><html lang='zh'><meta charset='utf-8'><title>手套与双手重叠覆盖审计</title><style>body{{font:16px sans-serif;margin:28px}}td,th{{border:1px solid #ccc;padding:8px}}table{{border-collapse:collapse}}</style><h1>手套与多手重叠：当前证据范围</h1><p>手套没有分类标注，也没有经确认的专门测试，不能给出手套性能。以下统计仅为同一穿戴者左右两手，不能推到多人三只以上手交叉。</p><p>重叠按GT投影框相交面积/较小框面积分组：小于0.1、0.1至0.5、大于等于0.5。框重叠是代理，不是手指像素遮挡真值。包含没有检测结果的图像，匹配覆盖的分母是有效GT投影手实例，并非像素可见手的召回率。</p><table><tr><th>分组</th><th>图像帧</th><th>匹配中心</th><th>匹配覆盖</th><th>WiLoR相对mm</th><th>完整模型相对mm</th><th>完整减WiLoR的序列配对95%CI</th></tr>{table}</table><p>身份统计：同一预测ID内已匹配的左右手切换{len(switches)}次；同一GT手相邻帧预测ID改变{len(fragments)}次。未匹配和断档不包含在切换准确性中。只约束单条预测轨迹，无手间碰撞约束，稳定不等于身份正确。</p><h2>真实强重叠案例</h2><p>三列WiLoR/完整模型/GT；绿为左手、橙为右手、灰为未匹配检测；数字是预测轨迹ID。</p>{panels}<h2>40张抽样手部画面，用于人工核对手套覆盖</h2><img src='coverage_0.jpg' style='max-width:100%'><img src='coverage_1.jpg' style='max-width:100%'><p>10片段5已用序列2已见受试者，诊断回放，非独立泛化。<a href='summary.json'>统计与来源</a></p></html>"""
    (out/'report.html').write_text(page,encoding='utf-8');print(json.dumps(dict(complete=True,strata=stats,identity_switches=len(switches),predictedID_changes=len(fragments),max_annotated_hands=summary['max_GT_annotated_hands'])),flush=True)
if __name__=='__main__':main()
