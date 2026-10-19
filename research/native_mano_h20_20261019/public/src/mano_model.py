"""WiLoR-compatible MANO21 with explicit chirality and wrist anchoring.

Local rotations use the right-canonical WiLoR convention. Left-hand geometry
is explicitly reflected before the physical root rotation, matching WiLoR.
The 51-state position is the wrist, not MANO transl. Betas are separate.
"""
import torch
from torch import nn
from smplx.lbs import batch_rodrigues


def load_decoder(device='cuda:0'):
    # Same MANO constructor and defaults as the pinned public WiLoR model.
    import wilor_eval_common as common
    from wilor.models.mano_wrapper import MANO
    assets = common.RUN/'assets'
    layer = MANO(model_path=str(assets/'MANO_RIGHT.pkl'), data_dir=str(assets),
                 gender='neutral', num_hand_joints=15,
                 mean_params=str(assets/'mano_mean_params.npz'), create_body_pose=False)
    return WristAnchoredMANO(layer).to(device).eval()


class WristAnchoredMANO(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.layer.requires_grad_(False)

    def forward(self, state, betas, right, vertices=False):
        size = state.shape[0]
        eye = torch.eye(3, device=state.device, dtype=state.dtype)[None, None].expand(size, 1, 3, 3)
        local = batch_rodrigues(state[:, 6:].reshape(-1, 3)).reshape(size, 15, 3, 3)
        output = self.layer(global_orient=eye, hand_pose=local, betas=betas, pose2rot=False)
        reflection = torch.ones(size, 3, device=state.device, dtype=state.dtype)
        reflection[:, 0] = right.to(state.dtype)*2-1
        root_rotation = batch_rodrigues(state[:, 3:6])
        relative = (output.joints-output.joints[:, :1])*reflection[:, None]
        joints = torch.einsum('bij,bkj->bki', root_rotation, relative)+state[:, None, :3]
        if not vertices:
            return joints
        relative_vertices = (output.vertices-output.joints[:, :1])*reflection[:, None]
        mesh = torch.einsum('bij,bkj->bki', root_rotation, relative_vertices)+state[:, None, :3]
        return joints, mesh
