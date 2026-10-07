import torch,json
from hand3d_data_v7 import RUN,load,batch,risk_features,save
from hand3d_temporal_v7 import TemporalHand3D,pack,unpack,WRIST
from train_hand3d_v7 import initialize,metrics

def main():
    torch.set_num_threads(4);torch.manual_seed(202610074);data=load('cuda:1');ids=torch.arange(12,device='cuda:1');b=batch(data,ids)
    assert (b['xyz'][:,8]-b['base']).abs().max()<2e-6
    assert (unpack(pack(b['base']))-b['base']).abs().max()<2e-6
    altered=dict(data);altered['gt']=data['gt']+123;altered['gt_uv']=data['gt_uv']+123;altered['valid']=~data['valid'];altered['uv_valid']=~data['uv_valid'];b2=batch(altered,ids)
    assert set(b)==set(b2) and all(torch.equal(b[k],b2[k]) for k in b)
    assert torch.equal(risk_features(b),risk_features(b2))
    result={}
    for kind in ['regression','dit']:
        model=TemporalHand3D(kind,True).to('cuda:1');initialize(model,'cuda:1');model.train()
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(b,data['gt'][ids],data['valid'][ids],data['gt_uv'][ids],data['uv_valid'][ids])
        loss.backward();assert torch.isfinite(loss)
        # Meaningful 3D output check, not a 2D prediction with fabricated depth.
        grad=float(model.head.weight.grad.float().norm());assert grad>0
        model.eval();locked=torch.zeros(12,20,dtype=torch.bool,device='cuda:1');locked[:,[0,5,8]]=True;b['confirmed']=locked
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):out=model.predict(b)
        assert out['xyz_camera_m'].shape==(12,20,3) and torch.isfinite(out['xyz_camera_m']).all()
        assert torch.equal(out['xyz_camera_m'][locked],b['base'][locked])
        result[kind]=dict(loss=float(loss.detach()),head_gradient_norm=grad,output_shape=list(out['xyz_camera_m'].shape),locked_xyz_exact=True)
        b['confirmed'].zero_()
    center=data['feature_ids'][:,8];base=data['xyz_camera_bank'][center]
    save(RUN/'preflight.json',dict(passed=True,camera_alignment_center_error_m=float((b['xyz'][:,8]-b['base']).abs().max()),pack_roundtrip=True,gt_invariance_passed=True,models=result,baseline=metrics(base,base,data['gt'],data['valid'])))
    print((RUN/'preflight.json').read_text())

if __name__=='__main__':main()
