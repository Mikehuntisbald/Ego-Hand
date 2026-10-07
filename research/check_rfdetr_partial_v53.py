"""Loss isolation checks; does not claim to verify the live visual encoder."""
import json,torch
from pathlib import Path
from rfdetr.models.criterion import SetCriterion
from rfdetr.models.matcher import HungarianMatcher
from rfdetr_partial_supervision_v53 import install
ROOT=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53_20261007')

def main():
    torch.set_num_threads(4);torch.manual_seed(53)
    old_mask,old_cost,old_labels=install();matcher=HungarianMatcher();c=SetCriterion(1,matcher,{},.25,['labels','boxes','masks'],ia_bce_loss=True)
    def target(image_id,mask):return dict(labels=torch.tensor([0]),boxes=torch.tensor([[.5,.5,.4,.4]]),masks=mask,image_id=torch.tensor([image_id]))
    strong=target(1,torch.ones(1,16,16,dtype=torch.bool));weak=target(3_000_001,torch.zeros(1,16,16,dtype=torch.bool));indices=[(torch.tensor([0]),torch.tensor([0]))]
    masks=torch.randn(1,4,16,16,requires_grad=True);outputs=dict(pred_logits=torch.zeros(1,4,1,requires_grad=True),pred_boxes=torch.tensor([[[.5,.5,.4,.4]]*4],requires_grad=True),pred_masks=masks)
    original=old_labels(c,outputs,[strong],indices,torch.tensor(1.))['loss_ce'];current=c.loss_labels(outputs,[strong],indices,torch.tensor(1.))['loss_ce'];assert torch.equal(original,current)
    loss=c.loss_masks(outputs,[weak],indices,torch.tensor(1.));total=sum(loss.values());total.backward(retain_graph=True);assert total.item()==0 and masks.grad.abs().max().item()==0
    out=c.loss_labels(outputs,[weak],indices,torch.tensor(1.));out['loss_ce'].backward(retain_graph=True);g=outputs['pred_logits'].grad;assert g[0,0].abs().sum()>0 and g[0,1:].abs().sum()==0
    ce,dice=matcher._compute_mask_costs(outputs,[weak]);assert ce.abs().max()==0 and dice.abs().max()==0
    masks.grad.zero_();state=torch.get_rng_state();reference=old_mask(c,outputs,[strong],indices,torch.tensor(1.));torch.set_rng_state(state);actual=c.loss_masks(outputs,[strong],indices,torch.tensor(1.));assert all(torch.equal(reference[k],actual[k]) for k in reference)
    sum(actual.values()).backward();assert masks.grad.abs().sum()>0
    result=dict(passed=True,strong_mask_loss_exact_parity=True,strong_class_loss_exact_parity=True,box_only_mask_loss_and_gradient_zero=True,box_only_mask_matching_cost_zero=True,CPPE_unmatched_query_classification_gradient_zero=True,matched_CPPE_query_gradient_nonzero=True,CPU_loss_only_not_live_visual_preflight=True)
    (ROOT/'partial_loss_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
