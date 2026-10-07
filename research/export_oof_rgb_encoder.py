"""Save the exact seeded frozen common RGB adapter for reproducible deployment."""
import torch
from coarse_pose3d import CoarsePose3D
from oof_common import RUN,save,sha
def main():
    torch.set_num_threads(4);torch.manual_seed(20261003)
    model=CoarsePose3D().eval()
    path=RUN/'rgb_encoder.pt'
    torch.save(dict(model=model.state_dict(),initialization='COCO YOLO26s plus frozen seeded pose adapter',seed=20261003,trained_on_hand_GT=False),path)
    save(RUN/'rgb_encoder_receipt.json',dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,seed=20261003,trained_on_hand_GT=False))
    print((RUN/'rgb_encoder_receipt.json').read_text(),flush=True)
if __name__=='__main__':main()

