"""Prediction-only diagnostics. Real motion is not an error without GT."""
import numpy as np
from .clock import motion_pairs

EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
         (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
         (0,17),(17,18),(18,19),(19,20)]


def summary(delivery, max_gap_ns):
    frames=delivery['frames'];clock=np.array([f['timestamp_ns'] for f in frames],np.int64)
    tracks=sorted({str(h['track_id']) for f in frames for h in f['hands']})
    answer=[]
    for track in tracks:
        rows=[next((h for h in f['hands'] if str(h['track_id'])==track),dict(valid=False)) for f in frames]
        valid=np.array([h['valid'] for h in rows])
        xyz=np.array([h['joints_world_m'] if h['valid'] else np.full((21,3),np.nan) for h in rows])
        segments=[f['camera_segment'] if f['head_valid'] else -1 for f in frames]
        pairs,triples=motion_pairs(clock,[track]*len(clock),segments,valid,max_gap_ns)
        v=np.array([(xyz[b,0]-xyz[a,0])/(60*((int(clock[b])-int(clock[a]))/1e9)) for a,b in pairs])
        a=np.array([((xyz[c,0]-xyz[b,0])/(60*((int(clock[c])-int(clock[b]))/1e9))-
                     (xyz[b,0]-xyz[a,0])/(60*((int(clock[b])-int(clock[a]))/1e9)))/
                    (60*((int(clock[c])-int(clock[a]))/2e9)) for a,b,c in triples])
        lengths=np.stack([np.linalg.norm(xyz[valid,b]-xyz[valid,a],axis=-1) for a,b in EDGES],-1) if valid.any() else np.empty((0,20))
        def stats(values):
            z=np.linalg.norm(values,axis=-1)*1000 if len(values) else np.empty(0)
            return dict(count=len(z),P95=float(np.percentile(z,95)) if len(z) else None,maximum=float(z.max()) if len(z) else None)
        answer.append(dict(track_id=track,source_frames=len(clock),valid_frames=int(valid.sum()),
                    valid_ratio=float(valid.mean()),interpolated_frames=sum(h.get('interpolated',False) and h['valid'] for h in rows),
                    wrist_displacement_mm_at_60Hz=stats(v),wrist_second_difference_mm_at_60Hz2=stats(a),
                    bone_length_relative_std_max=float(np.max(lengths.std(0)/np.maximum(lengths.mean(0),1e-9))) if len(lengths) else None))
    return dict(tracks=answer,GT_used=False,absolute_accuracy_evidence=False,
                motion_semantics='raw predicted motion; includes true fast actions, not GT-subtracted jump error',
                preservation_semantics='parameter version gates are reported separately; no correct/bad-point claims without GT',
                quality_admitted=False)
