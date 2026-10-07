"""EMA development inference + YOLO preservation gates; test is never opened."""
import json,copy,hashlib
from pathlib import Path
import numpy as np,torch
from torch import nn
from pytorch_lightning import Callback
from benchmark_rfdetr_v52 import box_iou,assignments

class BoxesOnly(nn.Module):
    def __init__(self,base):super().__init__();self.base=base
    def forward(self,outputs,*args,**kwargs):return self.base({k:v for k,v in outputs.items() if k!='pred_masks'},*args,**kwargs)

def score(rows,predictions,threshold):
    by_domain={};success={}
    for r,p in zip(rows,predictions):
        boxes=np.asarray(p['boxes']).reshape(-1,4);scores=np.asarray(p['scores']);boxes=boxes[scores>=threshold];gt=[v['box_xyxy'] for v in r['hands']];iou=box_iou(gt,boxes);x,y=assignments(iou);ok=np.zeros(len(gt),bool)
        for i,j in zip(x,y):ok[i]=iou[i,j]>=.5
        s=by_domain.setdefault(r['dataset'],dict(hands=0,matched=0,predictions=0));s['hands']+=len(gt);s['matched']+=int(ok.sum());s['predictions']+=len(boxes);success[r['id']]=ok.tolist()
    for s in by_domain.values():s.update(recall=s['matched']/max(s['hands'],1),precision=s['matched']/max(s['predictions'],1),F1=2*s['matched']/max(s['hands']+s['predictions'],1))
    return by_domain,success

class DevelopmentSelection(Callback):
    def __init__(self,wrapper,root):
        self.wrapper=wrapper;self.root=Path(root);self.rows=[r for r in map(json.loads,(self.root/'domain_records.jsonl').read_text().splitlines()) if r['split']=='dev'];self.baseline=json.loads((self.root/'yolo_development_baseline.json').read_text());self.best=-1.;self.best_trained=-1.
        self.args=copy.deepcopy(torch.load('/mnt/why/HOT3D/experiments/rfdetr_hand_instance_v52_20261007/full/checkpoint_best_total.pth',map_location='cpu',weights_only=False)['args'])
    def on_validation_end(self,trainer,module):
        if trainer.sanity_checking:return
        model=module._resolve_eval_model();old_model=self.wrapper.model.model;old_post=self.wrapper.model.postprocess;was_training=model.training;model.eval();self.wrapper.model.model=model;self.wrapper.model.postprocess=BoxesOnly(old_post);predictions=[]
        try:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for begin in range(0,len(self.rows),8):
                    batch=self.rows[begin:begin+8];out=self.wrapper.predict([r['image'] for r in batch],threshold=.001,include_source_image=False)
                    predictions.extend(dict(id=r['id'],boxes=d.xyxy.tolist(),scores=d.confidence.tolist()) for r,d in zip(batch,out))
        finally:self.wrapper.model.model=old_model;self.wrapper.model.postprocess=old_post;model.train(was_training)
        epoch=trainer.current_epoch;folder=self.root/'full';folder.mkdir(exist_ok=True);(folder/f'dev_epoch_{epoch:02d}_sealed.json').write_text(json.dumps(predictions))
        curve=[]
        for threshold in [.05,.1,.15,.2,.25,.3,.4,.5,.6,.7,.8,.9]:
            stats,ok=score(self.rows,predictions,threshold);lost=sum(int((np.asarray(self.baseline['success'][r['id']],bool)&~np.asarray(ok[r['id']],bool)).sum()) for r in self.rows if r['dataset']=='hot3d_preservation');old=self.baseline['statistics']
            gates=dict(native_old_correct_preserved=lost==0,Ego_recall_not_lower=stats['egohands']['recall']>=old['egohands']['recall'],Ego_precision_not_lower=stats['egohands']['precision']>=old['egohands']['precision'],Surgical_coverage_not_lower=stats['surgical_hands']['recall']>=old['surgical_hands']['recall'],CPPE_glove_coverage_not_lower=stats['cppe5']['recall']>=old['cppe5']['recall'])
            objective=.35*stats['egohands']['F1']+.3*stats['surgical_hands']['recall']+.2*stats['cppe5']['recall']+.15*stats['hot3d_preservation']['recall']
            curve.append(dict(threshold=threshold,statistics=stats,native_correct_lost=lost,gates=gates,admitted=all(gates.values()),objective=objective))
        candidate=max(curve,key=lambda r:r['objective']);admitted=[r for r in curve if r['admitted']];selected=max(admitted,key=lambda r:r['objective']) if admitted else None
        event=dict(epoch=epoch,step=trainer.global_step,EMA=True,candidate=candidate,selected=selected,curve=curve,GT_free_inference=True,test_labels_opened=False)
        (folder/f'dev_epoch_{epoch:02d}.json').write_text(json.dumps(event,indent=2));(folder/'development_status.json').write_text(json.dumps(event,indent=2))
        payload=lambda:dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},args=self.args,epoch=epoch,class_names=['hand'],notes=dict(experiment='v53 multidata',partial_mask_supervision=True,test_not_used_for_selection=True))
        if candidate['objective']>self.best_trained:
            self.best_trained=candidate['objective'];torch.save(payload(),folder/'best_trained_ema.pth');(folder/'best_trained_selection.json').write_text(json.dumps(candidate,indent=2))
        if selected and selected['objective']>self.best:
            self.best=selected['objective'];torch.save(payload(),folder/'best_admitted_ema.pth');(folder/'admitted_selection.json').write_text(json.dumps(dict(epoch=epoch,step=trainer.global_step,**selected),indent=2))
        print('MULTIDATA_DEV',json.dumps(dict(epoch=epoch,step=trainer.global_step,candidate=candidate,admitted=selected)),flush=True)
    def state_dict(self):return dict(best=self.best,best_trained=self.best_trained)
    def load_state_dict(self,state):self.best=state['best'];self.best_trained=state['best_trained']
