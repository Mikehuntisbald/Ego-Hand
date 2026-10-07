"""Explicit +/-1.6-second inference experiment; default v42 remains unchanged."""
import argparse,json
from pathlib import Path
import encode_hand3d_dense_v14 as sampling
from complete_hand_tracks_dit_v43 import complete as complete_v43
from temporal_window_v44 import SHORT_OFFSETS,LONG_OFFSETS


def complete(source,device='cuda:0',prepared=False,context_s=1.6):
    assert context_s in [1.6,4.0]
    previous=sampling.OFFSETS
    try:
        sampling.OFFSETS=SHORT_OFFSETS if context_s==1.6 else LONG_OFFSETS
        result=complete_v43(source,device,prepared,'matched_parameter_v43/dit','sequence')
    finally:sampling.OFFSETS=previous
    for track,input_track in zip(result['tracks'],source['tracks']):
        times=[f['timestamp_s'] for f in input_track['frames']]
        for i,frame in enumerate(track['frames']):
            selected=[j for j in frame['context_frame_indices'] if j is not None]
            assert len(selected)==len(set(selected))
            assert all(abs(times[j]-times[i])<=context_s+2e-6 for j in selected)
    result.update(mode='experimental_short_context_dit_v44',context_s=context_s,
                  context_offsets_frames=SHORT_OFFSETS if context_s==1.6 else LONG_OFFSETS,
                  weights_trained_context_s=4.0,context_retrained=False,default_replaced=False)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--prepared',action='store_true')
    ap.add_argument('--context-s',type=float,choices=[1.6,4.0],default=1.6);a=ap.parse_args()
    result=complete(json.loads(Path(a.input).read_text()),a.device,a.prepared,a.context_s)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(output=str(path),context_s=a.context_s,constraints_passed=result['constraint_checks_passed'])),flush=True)


if __name__=='__main__':main()
