"""Live natural RGB -> trainable final ViT blocks -> parameters -> FK."""
import numpy as np,torch
from torch import nn
from hand3d_v8_common import V7
from semantic_parameter_model_v36 import SemanticParameterHand
import spatial_rgb_common as s

RUN=V7.parent/'online_rgb_3d_v47'

class OnlineVisual(nn.Module):
    def __init__(self,device='cuda:3',joint=False):
        super().__init__();full,_=s.common.load_model(device);self.backbone=full.backbone;del full
        assert len(self.backbone.blocks)==32 and not self.backbone.skip_blocks
        self.joint=joint;self.configure()
    def configure(self):
        self.backbone.eval();self.backbone.requires_grad_(False)
        if self.joint:
            for block in self.backbone.blocks[28:]:block.requires_grad_(True)
            self.backbone.last_norm.requires_grad_(True)
    def tail_state(self):
        return {k:v.detach().cpu() for k,v in self.backbone.state_dict().items() if k.startswith(tuple(f'blocks.{i}.' for i in range(28,32))+('last_norm.',))}
    def load_tail(self,state):
        if state is not None:
            missing,unexpected=self.backbone.load_state_dict(state,strict=False)
            assert not unexpected and all(not k.startswith(tuple(f'blocks.{i}.' for i in range(28,32))+('last_norm.',)) for k in missing)
    def encode_pixels(self,images,device):
        out=[]
        # Match original BF16 and padded16-frame shape. No .detach() or feature
        # cache exists between the trainable visual layers and the 3D loss.
        for begin in range(0,len(images),16):
            part=list(images[begin:begin+16]);n=len(part);part+=part[-1:]*(16-n)
            x=s.common.input_tensor(part,[1]*16,rotation=1).to(device)
            with torch.autocast('cuda',dtype=torch.bfloat16):feature=self.backbone(x[:,:,:,32:-32])[-1].flatten(2).transpose(1,2)
            # Keep live gradients out of FP16's small-value range. Forward
            # values still originate from the matched BF16 encoder.
            out.append(feature[:n].float())
        return torch.cat(out)
    def replace(self,b,feature_ids,pixels):
        unique,inverse=torch.unique(feature_ids,return_inverse=True);keep=unique>0
        ids=unique[keep].cpu().tolist();live=self.encode_pixels([pixels[i] for i in ids],feature_ids.device)
        bank=torch.zeros(len(unique),192,1280,device=feature_ids.device,dtype=live.dtype)
        bank=bank.index_copy(0,torch.where(keep)[0],live)
        b=dict(b);b['rgb_native']=bank[inverse.flatten()].reshape(len(feature_ids),17,192,1280)
        return b

def load_head(kind,device):
    path=V7.parent/('matched_parameter_v43/dit' if kind=='dit' else 'fitted_parameter_v39/regression')/'best.pt'
    ck=torch.load(path,weights_only=False,map_location=device)
    model=SemanticParameterHand(kind,device).to(device);model.load_state_dict(ck['model'])
    return model,path,ck
