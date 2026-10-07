"""True mask loss only; box-only hands never become empty-mask negatives."""
import torch
import torch.nn.functional as F

def install():
    from rfdetr.models.criterion import SetCriterion
    from rfdetr.models.matcher import HungarianMatcher
    from rfdetr.utilities import box_ops
    old_masks=SetCriterion.loss_masks;old_cost=HungarianMatcher._compute_mask_costs;old_labels=SetCriterion.loss_labels

    def mask_cost(self,outputs,targets):
        ce,dice=old_cost(self,outputs,targets)
        valid=torch.cat([t['masks'].flatten(1).any(1) for t in targets]).to(ce.device)
        return ce*valid[None],dice*valid[None]

    def mask_loss(self,outputs,targets,indices,num_boxes):
        valid=[t['masks'].flatten(1).any(1) for t in targets]
        filtered=[(i[v[j]],j[v[j]]) for v,(i,j) in zip(valid,indices)]
        count=sum(v.sum() for v in valid)
        total=sum(len(t['labels']) for t in targets)
        norm=(num_boxes*count/max(total,1)).clamp_min(1)
        return old_masks(self,outputs,targets,filtered,norm)

    def labels(self,outputs,targets,indices,num_boxes,log=True,matched_targets=None):
        # CPPE labels gloves only: do not mark its unannotated bare hands as background.
        partial=[3_000_000<=int(t['image_id'].reshape(-1)[0])<4_000_000 for t in targets]
        if not any(partial):return old_labels(self,outputs,targets,indices,num_boxes,log,matched_targets)
        assert self.ia_bce_loss and self._matched_valid_mask(targets,indices) is None
        logits=outputs['pred_logits'].float();idx=self._get_src_permutation_idx(indices)
        classes=torch.cat([t['labels'][j] for t,(_,j) in zip(targets,indices)])
        boxes=torch.cat([t['boxes'][j] for t,(_,j) in zip(targets,indices)])
        iou,_=box_ops.elementwise_box_iou(box_ops.box_cxcywh_to_xyxy(outputs['pred_boxes'][idx].detach()),box_ops.box_cxcywh_to_xyxy(boxes))
        prob=logits.sigmoid();pos=torch.zeros_like(logits);neg=prob**2;loc=(*idx,classes)
        weight=(prob[loc]**self.focal_alpha*iou**(1-self.focal_alpha)).clamp_min(.01).detach()
        pos[loc]=weight;neg[loc]=1-weight
        use=torch.ones(logits.shape[:2],device=logits.device,dtype=torch.bool)
        for i,flag in enumerate(partial):
            if flag:use[i]=False;use[i,indices[i][0]]=True
        loss=(neg*logits-F.logsigmoid(logits)*(pos+neg))*use[:,:,None]
        result={'loss_ce':loss.sum()/num_boxes}
        if log:result['class_error']=100-100*(logits[idx].argmax(-1)==classes).float().mean()
        return result

    SetCriterion.loss_masks=mask_loss;SetCriterion.loss_labels=labels;HungarianMatcher._compute_mask_costs=mask_cost
    # Optimized multi-layer paths must not bypass the experiment's partial-label rules.
    SetCriterion._can_batch_detection_losses=lambda *args,**kwargs:False
    HungarianMatcher._match_many=lambda *args,**kwargs:None
    return old_masks,old_cost,old_labels
