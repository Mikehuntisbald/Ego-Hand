"""Pure-observation kinematic seeds, no GT pose/shape/identity reads."""
import torch,json,time
from hand3d_v8_common import V7,save
from parameter_codec_v31 import ParameterCodec,OUT

def main():
    torch.set_num_threads(4);d=torch.load(V7.parent/'side_data_v16/consensus/dense_data.pt',weights_only=False,mmap=True)
    right=torch.load(OUT/'predicted_right_bank.pt',weights_only=False);codec=ParameterCodec('cuda:0');pieces=[];start=time.time()
    for i in range(0,len(right),1024):
        with torch.no_grad():x=codec.coarse_states(d['xyz_camera_bank'][i:i+1024].to('cuda:0'),right[i:i+1024].to('cuda:0'))
        pieces.append(x.cpu())
    bank=torch.cat(pieces);assert torch.isfinite(bank).all();torch.save(bank,OUT/'coarse_state_bank.pt')
    save(OUT/'coarse_ready.json',dict(complete=True,observations=len(bank)-1,shape=list(bank.shape),seconds=time.time()-start,
        inputs='OriginalfrozenWiLoRcameraXYZ andpredictiononlyROIhandedness; no GT/subjectcalibration',
        seed='Neutraltrainingangleprior plus observedfingerbends, properpalmKabsch, originalwrist cameraXYZ, zeroPCAshape',
        role='Initialization/conditioning only; not a claimed reconstruction result'))
    print((OUT/'coarse_ready.json').read_text(),flush=True)

if __name__=='__main__':main()
