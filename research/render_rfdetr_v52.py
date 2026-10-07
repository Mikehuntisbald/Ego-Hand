"""Human-readable quantitative and natural-RGB failure comparison."""
import json,html
from pathlib import Path
import cv2,numpy as np
from benchmark_rfdetr_v52 import ROOT,PAIRED
COLORS=[(20,210,250),(240,160,20),(70,220,70),(220,60,220),(80,100,250)]

def overlay(image,masks,boxes=None,scores=None,label=''):
    out=image.copy()
    for i,m in enumerate(masks):
        color=np.array(COLORS[i%len(COLORS)]);out[m]=(out[m]*.6+color*.4).astype(np.uint8)
        contour,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(out,contour,-1,tuple(map(int,color)),2)
        if boxes is not None:
            x,y,z,t=np.rint(boxes[i]).astype(int);cv2.rectangle(out,(x,y),(z,t),tuple(map(int,color)),2)
            text=str(i)+(f' {scores[i]:.2f}' if scores is not None else '');cv2.putText(out,text,(max(x,0),max(y,20)),cv2.FONT_HERSHEY_SIMPLEX,.7,tuple(map(int,color)),2)
    cv2.rectangle(out,(0,0),(out.shape[1],42),(25,25,25),-1);cv2.putText(out,label,(12,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2);return out

def main():
    out=ROOT/'review';out.mkdir(exist_ok=True);metrics=json.loads((ROOT/'mask_evaluation.json').read_text());full=json.loads((ROOT/'full_glove_evaluation.json').read_text());mask_only=json.loads((ROOT/'full_glove_mask_only_evaluation.json').read_text());threshold=metrics['threshold']
    results=json.loads((ROOT/'mask_per_image.json').read_text());pred={r['id']:r for r in json.loads((ROOT/'mask_baseline/predictions.json').read_text())['frames']};gt={r['id']:r for r in map(json.loads,(PAIRED/'domain_records.jsonl').read_text().splitlines())};panels=[]
    # Select clear wins and failures after the quantitative arrays have been frozen.
    key=lambda r:r['YOLO_RF']['sum_iou']-r['YOLO_SAM']['sum_iou']
    selected=sorted(results,key=key)[:4]+sorted(results,key=key,reverse=True)[:4]
    for i,r in enumerate(selected):
        f=pred[r['id']];image=cv2.imread(f['image']);d=np.load(f['output']);keep=d['scores']>=threshold;b=np.load(f['baseline']);m=np.load(gt[r['id']]['mask_gt'])['masks'].astype(bool);raw=d['masks'][keep];exclusive=raw&~(raw.sum(0)>1)[None] if len(raw) else raw
        tiles=[overlay(image,m,label='Human masks'),overlay(image,b['masks'],b['boxes'],b['scores'],'YOLO + SAM2'),overlay(image,b['yolo_rf_masks'],b['boxes'],b['scores'],'YOLO + RF masks'),overlay(image,exclusive,d['boxes'][keep],d['scores'][keep],'RF boxes + masks')]
        tiles=[cv2.resize(v,(480,270)) for v in tiles];path=out/f'mask_case_{i:02d}.jpg';cv2.imwrite(str(path),np.concatenate(tiles,axis=1));panels.append(f'<figure><img src="{path.name}"><figcaption>{html.escape(r["id"])}：只换RF mask成功 {r["YOLO_RF"]["matched"]}/{r["YOLO_RF"]["hands"]}，YOLO+SAM2 {r["YOLO_SAM"]["matched"]}/{r["YOLO_SAM"]["hands"]}；帧内颜色只表示预测实例。</figcaption></figure>')
    natural=json.loads((ROOT/'nail_predictions/predictions.json').read_text());counts=[];writer=cv2.VideoWriter(str(out/'nail_rfdetr_masks.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),29.97002997,(704,704))
    for i,f in enumerate(natural['frames']):
        d=np.load(f['output']);keep=d['scores']>=threshold;counts.append(int(keep.sum()));tile=overlay(cv2.imread(f['image']),d['masks'][keep],d['boxes'][keep],d['scores'][keep],f'RF-DETR frame {i}: {keep.sum()} instances; query colors != IDs');writer.write(cv2.resize(tile,(704,704)))
        if i in [0,24,48,69,95,119]:
            path=out/f'nail_case_{i:03d}.jpg';cv2.imwrite(str(path),tile);panels.append(f'<figure><img src="{path.name}"><figcaption>真实护理 RGB：手套与患者手重叠。没有 mask/身份/3D 真值；该素材此前用于场景复核器弱标签，本轮 RF 未用它训练。</figcaption></figure>')
    writer.release();(out/'natural_counts.json').write_text(json.dumps(dict(frames=len(counts),detected_instances_per_frame=counts,median=float(np.median(counts)),GT_free=True,glove_accuracy_certified=False),indent=2))
    rows=''
    for name,s in metrics['mask_statistics'].items():rows+=f'<tr><td>{name}</td><td>{s["matched"]}/{s["hands"]}</td><td>{s["mask_recall_IoU50"]:.1%}</td><td>{s["mask_precision_IoU50"]:.1%}</td><td>{s["mean_assigned_mask_IoU"]:.3f}</td><td>{s["merged_predictions"]}</td></tr>'
    paired=''.join(f'<p>对照 {k}：按交互对 bootstrap 的召回差95% CI {v["recall_delta_CI95"]}，{v["groups"]} 组；恢复{v["recovered"]}/{v["previous_failures"]}个旧失败，损失{v["lost_correct"]}/{v["previous_correct"]}个原成功实例。</p>' for k,v in metrics['paired_comparisons'].items())
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>RF-DETR 手实例分割 v52</title><style>body{{font:16px system-ui;max-width:1280px;margin:32px auto;padding:0 20px;color:#243040}}p{{line-height:1.75}}table{{border-collapse:collapse}}th,td{{border:1px solid #d3dce8;padding:10px}}img,video{{max-width:100%}}figure{{margin:24px 0}}.note{{background:#fff2d5;padding:14px}}</style><h1>v52：训练 RF-DETR 手实例分割替代 SAM2</h1><p>COCO 预训练 SegSmall → EgoHands 原人工逐手轮廓微调 → 预测框+实例mask → 现有 WiLoR / 强空间 RGB / 时序参数 DiT / UmeTrack FK / 整段稳定化。576 输入，6轮；无人工遮挡。推理中没有 GT 框、mask、关键点或教师输出。</p><p class="note">裸手分割、手套检出和最终3D分别验证。手套只有2D标签，没有 mask/3D GT。默认未替换。测试复用了v51留出组，属于受控诊断对照，不能作为新一轮盲测泛化。</p><h2>真人多人多手实例 mask 对照</h2><p>120图、6个同步交互对；同一像素若归属多个预测实例，双方都标为未知。RFDETR_raw 是未消歧的原mask，单列参考。IoU≥0.5，一对一实例匹配。RF阈值{threshold:.2f}仅按200验证图F1选择；YOLO沿用已冻结的0.05阈值。SAM_same_RF_boxes 隔离分割器差异，YOLO_SAM比较完整前端。</p><table><tr><th>方法</th><th>成功实例</th><th>召回</th><th>精确率</th><th>含漏手的平均匹配IoU</th><th>覆盖两个GT手的预测mask</th></tr>{rows}</table>{paired}<p>全1200图框检出：{metrics['box_test']['matched']}/{metrics['box_test']['hands']}，召回{metrics['box_test']['recall']:.1%}，精确率{metrics['box_test']['precision']:.1%}。</p><h2>实际预测实例 → 完整3D：手套留出</h2><p>{full['images']}图 / {full['groups']}源视频 / {full['hands']}手 / {full['points']}标注2D点，漏手计失败。相同3D核心：YOLO+SAM2 PCK10 {full['SAM_PCK10']:.1%} → RF-DETR {full['RF_PCK10']:.1%}，组配对差CI {full['paired_source_PCK_delta_CI95']}。恢复{full['previous_failures_recovered']}个旧失败点，损失{full['previous_correct_lost']}/{full['SAM_correct']}个原成功点。该结果是3D输出的2D投影精度，不能证明手套的物理深度或遮挡3D准确性。</p><h2>自然失败与恢复可视化</h2><p>前三栏分别为人工轮廓、YOLO+SAM2、RF-DETR。后续为真实戴手套护理回放。逐帧query颜色不代表持续物理ID；稳定化约束通过也不能保证身份正确。</p><video controls src="nail_rfdetr_masks.mp4"></video>{''.join(panels)}<p><a href="mask_evaluation.json">实例完整统计</a> · <a href="full_glove_evaluation.json">完整3D投影与正确点保护</a> · <a href="training_verification.json">梯度、权重、checkpoint验证</a> · <a href="../../DEVELOPMENT_V52.md">开发说明</a></p><p>来源：<a href="https://github.com/roboflow/rf-detr">RF-DETR 官方代码</a>、<a href="https://vision.soic.indiana.edu/projects/egohands/">EgoHands</a>、<a href="https://github.com/MichiganCOG/Surgical_Hands_RELEASE">Surgical Hands</a>、<a href="https://commons.wikimedia.org/wiki/File:Nail_Care.webm">Nail Care 真实视频</a>。</p></html>'''
    point=f'<h2>主对照：保留YOLO与同一WiLoR观测，仅换SAM2 mask</h2><p>{mask_only["images"]}图 / {mask_only["points"]}点：PCK10 {mask_only["SAM_PCK10"]:.1%} → {mask_only["RF_PCK10"]:.1%}，配对源视频差CI {mask_only["paired_source_PCK_delta_CI95"]}；恢复{mask_only["previous_failures_recovered"]}点，损失{mask_only["previous_correct_lost"]}/{mask_only["SAM_correct"]}个旧成功点。RF无框提示接口，按预测框IoU≥0.25一对一关联RF查询，未匹配实例mask未知；从低分RF候选中关联，不使用GT。前面的RF框+mask结果另含检测器变化。</p>'
    page=page.replace('<h2>自然失败与恢复可视化</h2>',point+'<h2>自然失败与恢复可视化</h2>').replace('前三栏分别为人工轮廓、YOLO+SAM2、RF-DETR。','四栏分别为人工轮廓、YOLO+SAM2、保留YOLO仅换RF mask、RF框+mask。')
    (out/'report.html').write_text(page,encoding='utf-8')
    for name in ['mask_evaluation.json','full_glove_evaluation.json','full_glove_mask_only_evaluation.json','training_verification.json']:(out/name).write_bytes((ROOT/name).read_bytes())
    print(json.dumps(dict(report=str(out/'report.html'))),flush=True)

if __name__=='__main__':main()
