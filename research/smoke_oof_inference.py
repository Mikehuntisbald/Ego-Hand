"""Verify exported RGB basis and public inference using real observed crops."""
import cv2,numpy as np,torch
from infer_oof_pose import SubjectOOFPoseRefiner
from oof_common import RUN,save

def main():
    torch.set_num_threads(4);raw=torch.load(RUN/'oof_cache.pt',map_location='cpu',weights_only=False)
    ix=[i for i,r in enumerate(raw['rows']) if r['role']=='denoise'][:8]
    rows=[raw['rows'][i] for i in ix]
    images=torch.stack([torch.from_numpy(cv2.cvtColor(cv2.imread(r['crop']),cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()).float()/255 for r in rows])
    geometry=torch.tensor([r['geometry'] for r in rows]);results=[]
    for method in ['oof_dit','oof_regression','in_subject_dit']:
        api=SubjectOOFPoseRefiner(method,device='cuda:0');out=api(images,geometry)
        assert all(torch.isfinite(v).all() for v in out.values())
        assert out['refined_xyz_m'].shape==(8,20,3)
        assert ((out['gates']>=0)&(out['gates']<=1)).all()
        padded_image=torch.cat([images,torch.zeros(248,3,256,256)]).to('cuda:0')
        padded_geometry=torch.cat([geometry,torch.zeros(248,21)]).to('cuda:0')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):rgb=api.rgb_encoder(padded_image,padded_geometry)['rgb_tokens'][:8]
        difference=float((rgb.half().float().cpu()-raw['rgb'][ix].float()).abs().max())
        # Exported and cached adapters must represent the same feature basis;
        # allow BF16 convolution rounding across different batch shapes.
        assert difference<.02,difference
        coarse_difference=float((out['coarse_xyz_m'].cpu()-raw['in_subject_coarse'][ix]).abs().max())
        assert coarse_difference<.001,coarse_difference
        results.append(dict(method=method,exported_rgb_cache_max_difference=difference,coarse_cache_max_difference_m=coarse_difference,finite=True))
        del api;torch.cuda.empty_cache()
    save(RUN/'inference_smoke.json',dict(passed=True,GT_supplied_to_inference=False,checks=results))
    print((RUN/'inference_smoke.json').read_text(),flush=True)
if __name__=='__main__':main()
