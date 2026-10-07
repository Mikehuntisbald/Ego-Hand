"""Post-training development review against fixed-square YOLO; never read test targets."""
import json,copy,hashlib
from pathlib import Path
import numpy as np,torch
from rfdetr_dev_selection_v53 import score
from rfdetr_dev_selection_stageB_v53 import nms_predictions
B=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53B_20261007')
A=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53_20261007')
def main():
    torch.set_num_threads(4);out=B/'strict_review_r1';out.mkdir(exist_ok=True)
    rows=[r for r in map(json.loads,(B/'domain_records.jsonl').read_text().splitlines()) if r['split']=='dev'];baseline=json.loads((B/'yolo_development_baseline_fixed_square.json').read_text());old=baseline['statistics'];mask_baseline=json.loads((A/'full/done.json').read_text())['metrics']['val/ema_segm_mAP_50_95'];curve=[]
    for epoch in range(8):
        predictions=json.loads((B/f'full/dev_epoch_{epoch:02d}_sealed.json').read_text());mask_ap=json.loads((B/f'full/dev_epoch_{epoch:02d}.json').read_text())['candidate']['mask_AP']
        for policy,pp in [('raw',predictions),('box_nms_0.7',nms_predictions(predictions))]:
            for threshold in [.05,.1,.15,.2,.25,.3,.4,.5,.6,.7,.8,.9]:
                stats,ok=score(rows,pp,threshold);lost=sum(int((np.asarray(baseline['success'][r['id']],bool)&~np.asarray(ok[r['id']],bool)).sum()) for r in rows if r['dataset']=='hot3d_preservation')
                gates=dict(native_old_correct_preserved=lost==0,native_precision_not_lower=stats['hot3d_preservation']['precision']>=old['hot3d_preservation']['precision'],Ego_recall_not_lower=stats['egohands']['recall']>=old['egohands']['recall'],Ego_precision_not_lower=stats['egohands']['precision']>=old['egohands']['precision'],Surgical_coverage_not_lower=stats['surgical_hands']['recall']>=old['surgical_hands']['recall'],CPPE_glove_coverage_not_lower=stats['cppe5']['recall']>=old['cppe5']['recall'],CPPE_candidate_count_not_higher=stats['cppe5']['predictions']<=old['cppe5']['predictions'],Surgical_candidate_count_not_higher=stats['surgical_hands']['predictions']<=old['surgical_hands']['predictions'],reviewed_background_FP_not_higher=stats['clinical_background_review']['predictions']<=old['clinical_background_review']['predictions'],DOH_F1_not_lower=stats['100doh']['F1']>=old['100doh']['F1'],human_mask_AP_not_lower_than_stageA=mask_ap>=mask_baseline)
                objective=.25*stats['egohands']['F1']+.25*stats['100doh']['F1']+.2*stats['surgical_hands']['recall']+.15*stats['cppe5']['recall']+.15*stats['hot3d_preservation']['F1']
                curve.append(dict(epoch=epoch,threshold=threshold,policy=policy,statistics=stats,native_correct_lost=lost,mask_AP=mask_ap,gates=gates,development_qualified=all(gates.values()),objective=objective))
    qualified=[v for v in curve if v['development_qualified']];selected=max(qualified or curve,key=lambda v:v['objective']);checkpoint=B/f'full/checkpoint_{selected["epoch"]}.ckpt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    ema=state['callbacks']['RFDETREMACallback']['average_model_state_dict'];model={k[len('module.model.'):]:v for k,v in ema.items() if k.startswith('module.model.')};assert model
    expected={k[len('model.'):]:v for k,v in state['state_dict'].items() if k.startswith('model.')};assert set(model)==set(expected)
    payload=dict(model=model,args=state['args'],epoch=selected['epoch'],class_names=['hand'],**{k:state[k] for k in ['model_name','model_config','rfdetr_version']},notes=dict(EMA_extracted_from_full_checkpoint=str(checkpoint),selection='strict fixed-square development only',test_not_read=True))
    path=out/('best_development_ema.pth' if qualified else 'best_trained_diagnostic_ema.pth');torch.save(payload,path)
    # Confirm extraction is the same EMA used for the sealed development output.
    from rfdetr import RFDETR
    loaded=RFDETR.from_checkpoint(str(path),device='cpu',trust_checkpoint=True);actual=loaded.model.model.state_dict();assert set(actual)==set(model) and all(torch.equal(actual[k],v) for k,v in model.items())
    selection=dict(selected,checkpoint=str(path),checkpoint_sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),source_checkpoint=str(checkpoint),diagnostic_only=not bool(qualified),default_replaced=False,baseline_preprocessing='YOLO imgsz960 rect=False; identical square letterboxing independent of image batching',test_labels_used=False,EMA_extraction_reload_exact=True)
    (out/'selection.json').write_text(json.dumps(selection,indent=2));(out/'review.json').write_text(json.dumps(dict(selection=selection,all_candidates=curve,baseline=baseline,unchanged_training_source=True,posthoc_development_review_only=True,test_labels_used=False),indent=2));print(json.dumps(selection),flush=True)
if __name__=='__main__':main()
