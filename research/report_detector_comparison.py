"""Paired sequence uncertainty and visual audit of frozen detector outputs."""
import html
import json
from pathlib import Path
import cv2
import numpy as np
from compare_detectors import matches, operating, save

out=Path('/mnt/why/HOT3D/experiments/detector_compare_wilor_20261003')
result=json.loads((out/'comparison.json').read_text())
frames=json.loads((out/'frames.json').read_text())
names=['yolo26_hot3d','wilor_released']
predictions={n:json.loads((out/f'{n}_predictions.json').read_text())['predictions'] for n in names}
test_ids=[i for i,f in enumerate(frames) if f['split']=='test']
paired={}
for point in ['tune_best_f1','tune_fp_budget_0.05']:
    thresholds={n:result['results'][n]['metrics']['operating_points'][point]['test']['threshold'] for n in names}
    sequences=sorted({frames[i]['sequence'] for i in test_ids})
    stats=[]
    for seq in sequences:
        ids=[i for i in test_ids if frames[i]['sequence']==seq]
        stats.append({n:operating([frames[i] for i in ids],[predictions[n][i] for i in ids],thresholds[n]) for n in names})
    rng=np.random.default_rng(20261003)
    resamples=rng.integers(len(stats),size=(2000,len(stats)))
    groups={}
    for group in stats[0][names[0]]['groups']:
        counts=np.array([[s[names[0]]['groups'][group]['tp']-s[names[1]]['groups'][group]['tp'],s[names[0]]['groups'][group]['total']] for s in stats])
        totals=counts[resamples].sum(1)
        valid=totals[:,1]>0
        deltas=totals[valid,0]/totals[valid,1]*100
        groups[group]=dict(recall_difference_pp=float(counts[:,0].sum()/counts[:,1].sum()*100) if counts[:,1].sum() else None,
                           ci95_pp=np.quantile(deltas,[.025,.975]).tolist() if len(deltas) else None)
    paired[point]=dict(sequences=sequences,groups=groups,per_sequence=stats,
                       note='YOLO minus WiLoR; paired bootstrap of six source sequences, two reused test subjects')
save(out/'paired_analysis.json',paired)

point='tune_best_f1'
thresholds={n:result['results'][n]['metrics']['operating_points'][point]['test']['threshold'] for n in names}
candidates=[]
for i in test_ids:
    f=frames[i]
    used={n:matches(f,predictions[n][i],thresholds[n])[2] for n in names}
    yw=int((used[names[0]] & ~used[names[1]]).sum())
    ww=int((used[names[1]] & ~used[names[0]]).sum())
    candidates.append((yw,ww,i))
chosen=[]
clips=set()
for direction in [0,1]:
    for row in sorted(candidates,key=lambda r:r[direction],reverse=True):
        i=row[2]
        if row[direction]<=0 or frames[i]['clip'] in clips:
            continue
        chosen.append(i);clips.add(frames[i]['clip'])
        if sum((c[0] if direction==0 else c[1])>0 and c[2] in chosen for c in candidates)>=3:
            break
for i in test_ids:
    if len(chosen)>=6:
        break
    if frames[i]['clip'] not in clips and any(h['visibility']<.25 for h in frames[i]['hands']):
        chosen.append(i);clips.add(frames[i]['clip'])
panels=[]
for i in chosen[:6]:
    f=frames[i];image=cv2.imread(f['image']);h,w=image.shape[:2]
    columns=[]
    for n in names:
        im=cv2.resize(image,(512,512));scale=np.array([512/w,512/h,512/w,512/h])
        for hand in f['hands']:
            b=(np.array(hand['box'])*scale).astype(int)
            cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(0,240,0),2)
        p=predictions[n][i]
        for b,score in zip(p['boxes'],p['scores']):
            if score<thresholds[n]:continue
            b=(np.array(b)*scale).astype(int)
            cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(0,140,255),2)
            cv2.putText(im,f'{score:.2f}',(int(b[0]),max(15,int(b[1])-3)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,140,255),1)
        cv2.rectangle(im,(0,0),(512,42),(24,24,24),-1)
        cv2.putText(im,f'{n} | {f["clip"]}:{f["frame"]}',(8,26),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        columns.append(im)
    panels.append(np.concatenate(columns,axis=1))
if panels:
    cv2.imwrite(str(out/'comparison_examples.jpg'),np.concatenate(panels,axis=0))
save(out/'example_frames.json',[frames[i] for i in chosen[:6]])

rows=[]
for n in names:
    r=result['results'][n];m=r['metrics'];o=m['operating_points'][point]['test']
    rows.append(f'<tr><td>{n}</td><td>{m["ap50"]*100:.2f}</td><td>{m["ap75"]*100:.2f}</td><td>{m["map50_95"]*100:.2f}</td><td>{o["precision"]*100:.2f}</td><td>{o["recall"]*100:.2f}</td><td>{o["fp"]}</td><td>{o["groups"]["occluded_lt_0.5"]["recall"]*100:.2f}</td><td>{o["groups"]["severe_lt_0.25"]["recall"]*100:.2f}</td><td>{r["timing"]["median_ms"]:.2f}</td></tr>')
markup='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>HOT3D detector comparison</title>
<style>body{font-family:system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#172334}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccd}img{max-width:100%}pre{white-space:pre-wrap;background:#f3f5f7;padding:18px}</style>
<h1>HOT3D: YOLO26 vs WiLoR detector</h1>
<p>现有 YOLO26s HOT3D 微调权重，对比 WiLoR 官方预训练检测器。900 张 P0003 开发帧选择阈值；720 张测试帧，1329 个有效手框。统一 960 输入、FP32、H20，左右手类别合并评估。</p>
<p>这是现有模型的部署对照；训练数据和监督不同，不能据此证明架构普遍优劣。测试序列此前已用于本工程评估。AP 为单类 101 点插值指标；非完整 COCO 分面积套件。</p>
<table><tr><th>模型</th><th>AP50%</th><th>AP75%</th><th>mAP50:95%</th><th>精确率%</th><th>召回率%</th><th>误检数</th><th>遮挡召回%</th><th>严重遮挡召回%</th><th>中位延迟ms</th></tr>'''+''.join(rows)+'''</table>
<p>表中工作点由 P0003 最佳 F1 选择，并冻结用于测试。遮挡：整手可见比例 &lt;0.5，108 个手；严重遮挡：&lt;0.25，42 个手。未额外限定全部关节在视场内，与先前 DiT 分组不同。延迟包括预处理/网络/后处理，不含图片解码。</p>
<p>绿色为 HOT3D 完整手框标注；橙色为各检测器输出。示例按两模型召回差异和困难帧挑选，非随机总体展示。</p><img src="comparison_examples.jpg" alt="Ground truth and detector boxes">
<p><a href="comparison.json">完整结果</a> · <a href="paired_analysis.json">按源序列配对区间</a> · <a href="protocol.json">协议</a></p>
<pre>'''+html.escape(json.dumps(paired[point]['groups'],indent=2))+'''</pre></html>'''
(out/'report.html').write_text(markup)
print(json.dumps({n:dict(ap50=result['results'][n]['metrics']['ap50'],map=result['results'][n]['metrics']['map50_95'],operating=result['results'][n]['metrics']['operating_points']['tune_best_f1']['test'],timing=result['results'][n]['timing']) for n in names},indent=2))
