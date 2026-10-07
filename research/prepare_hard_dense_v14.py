"""Revisit all 47 retained natural failures with actual 30 FPS context.

These cases have already been inspected. They are diagnostics, not a new test.
Original center observations/targets are kept exactly; only context is denser.
"""
import collections, hashlib, json, time
from pathlib import Path
import torch
from hand3d_v8_common import V7, load, save
import spatial_rgb_common as s
import prepare_dense_sampling_v13 as worker

RUN=V7.parent/'hard_dense_v14'
OUT=V7.parent/'hard_aligned_v14'

def main():
    torch.set_num_threads(4);RUN.mkdir(exist_ok=True)
    queue=json.loads((V7.parent/'natural_reliability_v4/delivery/hardcase_review_queue.json').read_text())
    old=load('cpu');records,_=s.records_and_index();clips={};grouped=collections.defaultdict(list)
    for q in queue:
        r=records[int(old['feature_ids'][q['window_index'],8])-1]
        split=Path(r['image']).relative_to(s.common.ROOT/'export/images').parts[0]
        key=(r['sequence'],int(r['clip']))
        clips[key]=dict(role='diagnostic',subject=r['subject'],split=split)
    for (seq,cid),r in clips.items():
        path=s.common.ROOT/'export/annotations'/r['split']/seq/f'clip-{cid:06d}.jsonl'
        assert len(path.read_text().splitlines())==150
        grouped[(r['split'],r['subject'],seq)].append(cid)
    source=json.loads((s.common.ROOT/'subset_manifest.json').read_text())
    manifest={k:source[k] for k in ['official_repo','revision','mirror','stream']}
    manifest.update(sequences=[dict(split=sp,subject=sub,sequence=seq,clips=sorted(ids)) for (sp,sub,seq),ids in sorted(grouped.items())],created_unix=time.time(),scope='47 previously inspected natural failures; diagnostic only',frames_stride=1)
    save(RUN/'fresh_manifest.json',manifest)
    save(RUN/'fresh_download_done.json',dict(complete=True,source_exports_reused=True,clips=len(clips),network_download=False))
    save(RUN/'case_queue.json',queue)
    worker.RUN=RUN
    if not (RUN/'dense_ready.json').exists():
        worker.prepare_observations();worker.assemble_roles_and_targets(clips)
    # Use the same audited center-preserving assembler, explicitly replacing
    # only the window selection. Keep an exact source snapshot and its hashes.
    path=Path(__file__).resolve().parent/'build_aligned_density_v13.py'
    original=path.read_text();before="ids=torch.tensor([i for i,r in enumerate(old['roles']) if r in ['train','dev_select','dev_calibrate']])"
    after='ids=torch.tensor('+repr([q['window_index'] for q in queue])+')'
    assert original.count(before)==1
    generated=original.replace(before,after);OUT.mkdir(exist_ok=True)
    snapshot=OUT/'assembler_snapshot.py';snapshot.write_text(generated)
    save(OUT/'assembler_provenance.json',dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),replacement={before:after},diagnostic=True))
    ns=dict(__name__='hard_dense_v14_assembler',__file__=str(snapshot))
    exec(compile(generated,str(snapshot),'exec'),ns);ns['RUN']=RUN;ns['OUT']=OUT
    if not (OUT/'ready.json').exists():ns['main']()
    ready=json.loads((OUT/'ready.json').read_text());ready['scope']='All47 retained failures; already inspected; original centers exact; no independent accuracy claim'
    save(OUT/'ready.json',ready);save(OUT/'case_queue.json',queue)
    print(json.dumps(ready),flush=True)

if __name__=='__main__':main()
