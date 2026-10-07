import json,copy
import torch
from hand3d_data_v7 import RUN as BASE
from evaluate_hand3d_bounded_v7 import RUN
import spatial_rgb_common as s

def main():
    data=torch.load(BASE/'data.pt',weights_only=False,mmap=True);records,_=s.records_and_index();rows=data['rows']
    hard=json.loads((s.common.ROOT/'experiments/natural_reliability_v4/delivery/hardcase_review_queue.json').read_text());q=next(q for q in hard if q['id']==12);row=rows[q['window_index']]
    key=lambda r:(r['source'],r['sequence'],r['clip'],r['track_id']);fids=set();uv={}
    for j,r in enumerate(rows):
        if key(r)!=key(row):continue
        for k,fid in enumerate(data['feature_ids'][j].tolist()):
            if fid:fids.add(fid);uv[fid]=(data['xy'][j,k]*1408).tolist()
    chosen=sorted(fids,key=lambda fid:records[fid-1]['frame'])[:17];assert len(chosen)==17
    frames=[]
    for fid in chosen:
        r=records[fid-1];f=dict(image=r['image'],camera=r['camera'],timestamp_s=r['frame']/30.,clip=r['clip'],box_xyxy=r['box'],box_confidence=r['score'],
            xyz_camera_m=data['xyz_camera_bank'][fid].tolist(),available_3d=data['available_bank'][fid].tolist(),xy_px=uv[fid],confirmed_3d=[j in [0,5] for j in range(20)])
        frames.append(f)
    obj=dict(image_size=[1408,1408],fixture='Confirmation locks test only; locked values are predictions, not human-validated annotation',tracks=[dict(id='raw_3d_smoke',frames=frames)])
    (RUN/'example_input.json').write_text(json.dumps(obj,indent=2));junk=copy.deepcopy(obj)
    for f in junk['tracks'][0]['frames']:f.update(gt_xyz_camera_m=[[999,999,999]]*20,gt_side='wrong',visibility_label=-999)
    (RUN/'gt_poisoned_input.json').write_text(json.dumps(junk,indent=2))
    print(json.dumps(dict(frames=17,confirmed_fixture_points=34)))

if __name__=='__main__':main()
