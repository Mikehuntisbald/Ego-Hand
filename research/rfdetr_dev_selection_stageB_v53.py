"""Development-only selection with native precision and candidate-cost guards."""
import json,copy
from pathlib import Path
import numpy as np,torch
from pytorch_lightning import Callback
from rfdetr_dev_selection_v53 import BoxesOnly,score

def nms_predictions(predictions,iou=.7):
    out=[]
    for p in predictions:
        boxes=np.asarray(p['boxes'],float).reshape(-1,4);scores=np.asarray(p['scores']);order=np.argsort(-scores,kind='stable');keep=[]
        while len(order):
            j=order[0];keep.append(int(j));tail=order[1:]
            if not len(tail):break
            tl=np.maximum(boxes[j,:2],boxes[tail,:2]);br=np.minimum(boxes[j,2:],boxes[tail,2:]);inter=np.maximum(br-tl,0).prod(1);area=np.maximum(boxes[:,2:]-boxes[:,:2],0).prod(1)
            overlap=inter/np.maximum(area[j]+area[tail]-inter,1e-8);order=tail[overlap<=iou]
        out.append(dict(p,boxes=boxes[keep].tolist(),scores=scores[keep].tolist(),keep_indices=keep))
    return out

class DevelopmentSelection(Callback):
    def __init__(self,wrapper,root):
        self.wrapper=wrapper;self.root=Path(root);self.rows=[r for r in map(json.loads,(self.root/'domain_records.jsonl').read_text().splitlines()) if r['split']=='dev'];self.baseline=json.loads((self.root/'yolo_development_baseline.json').read_text());self.best=-1.;self.best_trained=-1.
        initial=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53_20261007')
        self.args=copy.deepcopy(torch.load(initial/'full/best_admitted_ema.pth',map_location='cpu',weights_only=False)['args'])
        reference=torch.load('/mnt/why/HOT3D/experiments/rfdetr_hand_instance_v52_20261007/full/checkpoint_best_total.pth',map_location='cpu',weights_only=False)
        self.architecture={k:copy.deepcopy(reference[k]) for k in ['model_name','model_config','rfdetr_version']}
        self.mask_baseline=json.loads((initial/'full/done.json').read_text())['metrics']['val/ema_segm_mAP_50_95']
    def on_validation_end(self,tr,module):
        if tr.sanity_checking:return
        model=module._resolve_eval_model();old_model=self.wrapper.model.model;old_post=self.wrapper.model.postprocess;was_training=model.training;model.eval();self.wrapper.model.model=model;self.wrapper.model.postprocess=BoxesOnly(old_post);predictions=[]
        try:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for begin in range(0,len(self.rows),8):
                    batch=self.rows[begin:begin+8];out=self.wrapper.predict([r['image'] for r in batch],threshold=.001,include_source_image=False)
                    predictions.extend(dict(id=r['id'],boxes=d.xyxy.tolist(),scores=d.confidence.tolist()) for r,d in zip(batch,out))
        finally:self.wrapper.model.model=old_model;self.wrapper.model.postprocess=old_post;model.train(was_training)
        epoch=tr.current_epoch;folder=self.root/'full';folder.mkdir(exist_ok=True);(folder/f'dev_epoch_{epoch:02d}_sealed.json').write_text(json.dumps(predictions))
        old=self.baseline['statistics'];curve=[];mask_ap=float(tr.callback_metrics.get('val/ema_segm_mAP_50_95',-1))
        for policy,pp in [('raw',predictions),('box_nms_0.7',nms_predictions(predictions))]:
            for threshold in [.05,.1,.15,.2,.25,.3,.4,.5,.6,.7,.8,.9]:
                stats,ok=score(self.rows,pp,threshold)
                lost=sum(int((np.asarray(self.baseline['success'][r['id']],bool)&~np.asarray(ok[r['id']],bool)).sum()) for r in self.rows if r['dataset']=='hot3d_preservation')
                gates=dict(native_old_correct_preserved=lost==0,native_precision_not_lower=stats['hot3d_preservation']['precision']>=old['hot3d_preservation']['precision'],Ego_recall_not_lower=stats['egohands']['recall']>=old['egohands']['recall'],Ego_precision_not_lower=stats['egohands']['precision']>=old['egohands']['precision'],Surgical_coverage_not_lower=stats['surgical_hands']['recall']>=old['surgical_hands']['recall'],CPPE_glove_coverage_not_lower=stats['cppe5']['recall']>=old['cppe5']['recall'],CPPE_candidate_count_not_higher=stats['cppe5']['predictions']<=old['cppe5']['predictions'],Surgical_candidate_count_not_higher=stats['surgical_hands']['predictions']<=old['surgical_hands']['predictions'],reviewed_background_FP_not_higher=stats['clinical_background_review']['predictions']<=old['clinical_background_review']['predictions'],DOH_F1_not_lower=stats['100doh']['F1']>=old['100doh']['F1'],human_mask_AP_not_lower_than_stageA=mask_ap>=self.mask_baseline)
                objective=.25*stats['egohands']['F1']+.25*stats['100doh']['F1']+.2*stats['surgical_hands']['recall']+.15*stats['cppe5']['recall']+.15*stats['hot3d_preservation']['F1']
                curve.append(dict(threshold=threshold,policy=policy,statistics=stats,native_correct_lost=lost,mask_AP=mask_ap,gates=gates,development_admitted=all(gates.values()),objective=objective))
        candidate=max(curve,key=lambda r:r['objective']);qualified=[r for r in curve if r['development_admitted']];selected=max(qualified,key=lambda r:r['objective']) if qualified else None
        event=dict(epoch=epoch,step=tr.global_step,EMA=True,candidate=candidate,selected=selected,curve=curve,GT_free_inference=True,test_labels_opened=False,development_gates_only=True,default_replaced=False)
        (folder/f'dev_epoch_{epoch:02d}.json').write_text(json.dumps(event,indent=2));(folder/'development_status.json').write_text(json.dumps(event,indent=2))
        def payload():return dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},args=self.args,epoch=epoch,class_names=['hand'],**self.architecture,notes=dict(experiment='v53B five-source',test_not_used_for_selection=True))
        if candidate['objective']>self.best_trained:
            self.best_trained=candidate['objective'];torch.save(payload(),folder/'best_trained_ema.pth');(folder/'best_trained_selection.json').write_text(json.dumps(dict(epoch=epoch,step=tr.global_step,**candidate),indent=2))
        if selected and selected['objective']>self.best:
            self.best=selected['objective'];torch.save(payload(),folder/'best_development_ema.pth');(folder/'development_selection.json').write_text(json.dumps(dict(epoch=epoch,step=tr.global_step,**selected),indent=2))
        print('STAGEB_DEV',json.dumps(dict(epoch=epoch,candidate=candidate,selected=selected)),flush=True)
    def state_dict(self):return dict(best=self.best,best_trained=self.best_trained)
    def load_state_dict(self,state):self.best=state['best'];self.best_trained=state['best_trained']
