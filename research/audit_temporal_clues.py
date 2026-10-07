"""Original-frame visibility audit. GT locates the same hand for review only."""
import argparse,json
from pathlib import Path
import spatial_rgb_common as s
import numpy as np,cv2,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from offline_rgb_encoder import crop_roi,crop_image
R=s.common.ROOT/'experiments/natural_reliability_v4';O=R/'temporal_clue_audit';O.mkdir(exist_ok=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dense',type=int,nargs='*');a=ap.parse_args()
    queue=json.loads((R/'delivery/hardcase_review_queue.json').read_text());cases=[q for q in queue if q['after_px']>40]
    records,index=s.records_and_index();metadata=[]
    for q in cases:
        if a.dense is not None and q['id'] not in a.dense:continue
        rec=records[int(index['feature_ids'][q['window_index'],8])-1]
        annotation=s.common.ROOT/'export/annotations/dit_v3_locked'/q['sequence']/f"clip-{q['clip']:06d}.jsonl"
        frames=[json.loads(l) for l in annotation.read_text().splitlines()];byid={f['frame']:f for f in frames}
        center=byid[q['frame']];hand=min(center['hands'],key=lambda h:np.linalg.norm(np.asarray(h['xyz_camera_m'])-np.asarray(rec['gt'])))
        side=hand['side'];images=[];details=[]
        for f in frames:
            h=next((h for h in f['hands'] if h['side']==side),None)
            if h is None or h['box_amodal_xyxy'] is None:
                images.append(np.zeros((256,256,3),np.uint8));details.append(dict(frame=f['frame'],hand_annotation=False));continue
            im=cv2.imread(str(s.common.ROOT/'export'/f['image']));assert im is not None
            box=np.asarray(h['box_amodal_xyxy'],dtype=np.float32).copy()
            box[[0,2]]=np.clip(box[[0,2]],0,im.shape[1]);box[[1,3]]=np.clip(box[[1,3]],0,im.shape[0])
            roi=crop_roi(box) if box[2]-box[0]>8 and box[3]-box[1]>8 else crop_roi([0,0,im.shape[1],im.shape[0]])
            images.append(np.rot90(crop_image(im,roi)[:,:,::-1]))
            details.append(dict(frame=f['frame'],timestamp_ns=f['timestamp_ns'],hand_annotation=True,hand_visible_fraction=h['modeled_hand_visible_fraction'],roi=roi.tolist()))
        fps=(len(frames)-1)/((frames[-1]['timestamp_ns']-frames[0]['timestamp_ns'])*1e-9)
        item=dict(**q,side=side,frames=len(frames),fps=float(fps),duration_s=(frames[-1]['timestamp_ns']-frames[0]['timestamp_ns'])*1e-9,
            model_frame_range=[max(0,q['frame']-40),min(len(frames)-1,q['frame']+40)],frames_metadata=details)
        item['actual_model_frames']=[records[int(fid)-1]['frame'] if int(fid)>0 else None for fid in index['feature_ids'][q['window_index']]]
        metadata.append(item)
        if a.dense is None:
            samples=list(range(0,len(frames),5));fig,axes=plt.subplots(5,6,figsize=(14,12),layout='constrained')
            pages=[(0,samples,fig,axes)]
        else:
            pages=[]
            for start in range(0,len(frames),50):
                fig,axes=plt.subplots(5,10,figsize=(18,10),layout='constrained');pages.append((start,list(range(start,min(start+50,len(frames)))),fig,axes))
        for page,samples,fig,axes in pages:
            for ax in axes.flat:ax.axis('off')
            for ax,i in zip(axes.flat,samples):
                ax.imshow(images[i]);offset=(frames[i]['timestamp_ns']-center['timestamp_ns'])*1e-9
                ax.set_title(f"f{i:03d}  {offset:+.2f}s"+(' TARGET' if i==q['frame'] else ''),fontsize=8,color='red' if i==q['frame'] else 'black')
            fig.suptitle(f"ID {q['id']} | {q['sequence']} clip {q['clip']} | target f{q['frame']} | {side} hand\nOriginal pixels; GT hand boxes used only to locate review crop; black can be out-of-image padding",fontsize=11)
            name=f"case_{q['id']}_"+(f'dense_{page:03d}' if a.dense is not None else 'overview')+'.png'
            fig.savefig(O/name,dpi=120);plt.close(fig)
        s.save(O/f"case_{q['id']}_metadata.json",item)
    if a.dense is None:s.save(O/'cases.json',metadata)
    print(json.dumps(dict(cases=len(metadata),dense=a.dense is not None)))

if __name__=='__main__':main()
