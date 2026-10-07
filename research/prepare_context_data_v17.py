"""Same-center contexts; labels follow one hand across time, never select inputs."""
import json,time
from pathlib import Path
import numpy as np,torch
from hand3d_v8_common import V7,save
from hand3d_rollout_v8 import project_fisheye624
import spatial_rgb_common as s
RUN=V7.parent/'context_data_v17';SOURCE=V7.parent/'side_data_v16/consensus';AUDIT=V7.parent/'context_merge_v17'

def main():
    torch.set_num_threads(4);RUN.mkdir(exist_ok=True);started=time.time()
    data=torch.load(SOURCE/'dense_data.pt',weights_only=False,mmap=True);records=json.loads((V7.parent/'dense_sampling_v13/fresh_rows.json').read_text());old_records,_=s.records_and_index();old=torch.load(V7/'data.pt',weights_only=False,mmap=True)
    sides={r['window_index']:r['GT_side_diagnostic'] for r in json.loads((V7.parent/'side_consensus_v16/audit_results.json').read_text())['sets']['train_development']['rows']}
    proposals=torch.load(AUDIT/'train_development_proposed_slots.pt',weights_only=False);assert torch.equal(proposals['source_window_indices'],data['source_window_indices'])
    uv_bank=project_fisheye624(data['xyz_camera_bank'],torch.load(V7.parent/'aligned_density_v13/camera_params.pt',weights_only=False,mmap=True))/1408
    observed=torch.isfinite(uv_bank).all(-1)&(uv_bank>=0).all(-1)&(uv_bank<1).all(-1)&data['available_bank'];annotation_cache={}
    def frame_labels(rec):
        sp,seq,cid=Path(rec['image']).relative_to(s.common.ROOT/'export/images').parts[:3];path=s.common.ROOT/'export/annotations'/sp/seq/(cid+'.jsonl')
        if path not in annotation_cache:annotation_cache[path]=[json.loads(x) for x in path.read_text().splitlines()]
        return annotation_cache[path][rec['frame']]['hands']
    for variant,fids,dt in [('control',data['feature_ids'],data['dt']),('bridge',proposals['feature_ids'],proposals['dt'])]:
        assert torch.equal(fids[:,8],data['feature_ids'][:,8]);dest=RUN/variant;dest.mkdir(exist_ok=True);new={**data,'feature_ids':fids,'dt':dt,'xy':torch.nan_to_num(uv_bank)[fids],'observed_2d':observed[fids]}
        labels=torch.zeros(len(fids),17,20,3);valid=torch.zeros(len(fids),17,20,dtype=torch.bool);uv=torch.zeros(len(fids),17,20,2);uv_valid=torch.zeros_like(valid)
        for k,source_index in enumerate(data['source_window_indices']):
            current=fids[k,8];current_R=data['rotation'][current];current_t=data['translation'][current];side=sides[int(source_index)]
            assert side in ['left','right']
            for slot,fid in enumerate(fids[k]):
                if not fid:continue
                if slot==8:rec=old_records[int(old['feature_ids'][source_index,8])-1]
                else:rec=records[int(fid)-1]
                hands=[h for h in frame_labels(rec) if h['side']==side]
                if not hands:continue
                assert len(hands)==1;hand=hands[0];xyz=torch.tensor(hand['xyz_camera_m']);mask=torch.isfinite(xyz).all(-1);xyz=torch.nan_to_num(xyz)
                world=xyz@data['rotation'][fid].T+data['translation'][fid]
                labels[k,slot]=(world-current_t)@current_R;valid[k,slot]=mask
                image_uv=s.common.from_json(rec['camera']).eye_to_window(xyz.numpy())/1408;uv[k,slot]=torch.tensor(np.nan_to_num(image_uv))
                # This auxiliary image loss only uses points geometrically in
                # the selected crop.3D labels remain valid even when invisible.
                roi=data['roi'][fid].numpy();inside=(image_uv>=roi[:2]).all(-1)&(image_uv<roi[2:]).all(-1)
                uv_valid[k,slot]=torch.tensor(hand['keypoint_projection_valid'])&torch.tensor(np.isfinite(image_uv).all(-1)&inside)&mask
            assert (labels[k,8]-data['gt'][k]).abs().max()<2e-6 and torch.equal(valid[k,8],data['valid'][k])
        torch.save(new,dest/'dense_data.pt');torch.save(dict(gt=labels,valid=valid,gt_uv=uv,uv_valid=uv_valid),dest/'trajectory_labels.pt')
        for name in ['native_bank.pt']:
            link=dest/name
            if not link.exists():link.symlink_to(SOURCE/name)
        save(dest/'ready.json',dict(complete=True,windows=len(fids),variant=variant,centers_gt_rgb_unchanged=True,targets='Onecenterhandidentity across all selectedtimestamps; GTidentity used ONLY for supervised labels, never frame selection or inference',scope='Existingtrain/devonly. Predictedselector may contain wrong/unmatched contexts; retainedratherthandroppedusingGT. Auxiliary2Dlabels limited to suppliedROI,3Dvisibilitynotused.',seconds=time.time()-started))
    save(RUN/'ready.json',dict(complete=True,windows=len(data['roles']),roles={r:data['roles'].count(r) for r in set(data['roles'])},scope='Matchedsame-centers/RGB/XYZ; only inputcontextselection changes. Botharms have consistent samehand3Dlabels. No newtest/fifth/retainedfailures used totrain.',seconds=time.time()-started))
    print((RUN/'ready.json').read_text(),flush=True)

if __name__=='__main__':main()
