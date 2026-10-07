"""Freeze splits and select new sequences without looking at their labels."""
import json,random
from collections import defaultdict
from oof_common import ROOT,OLD,RUN,SUBJECTS,save,sha

def main():
    RUN.mkdir(parents=True,exist_ok=True)
    dest=RUN/'protocol.json'
    if dest.exists():print(dest.read_text());return
    old=json.loads((ROOT/'subset_manifest.json').read_text())
    rows=[json.loads(l) for l in (OLD/'predicted_manifest.jsonl').read_text().splitlines()]
    by=defaultdict(set)
    for r in rows:
        if r['subject'] in SUBJECTS:by[r['subject']].add(r['sequence'])
    rng=random.Random(20261003);gate_sequences=[]
    for subject in SUBJECTS:
        seqs=sorted(by[subject]);rng.shuffle(seqs);gate_sequences.append(seqs[0])
    tune_seq=sorted({r['sequence'] for r in rows if r['role']=='tune'})
    assert len(tune_seq)==2
    definitions=json.loads((ROOT/'provenance/clip_definitions.response').read_text())
    splits=json.loads((ROOT/'provenance/clip_splits.response').read_text())
    eligible=defaultdict(list)
    old_sequences={s['sequence'] for s in old['sequences']}
    for clip in splits['train']['Aria']:
        seq=definitions[str(clip)]['sequence_id']
        if seq.split('_')[0] in ['P0010','P0015'] and seq not in old_sequences:eligible[seq].append(int(clip))
    fresh=[]
    for subject in ['P0010','P0015']:
        seqs=sorted(q for q,clips in eligible.items() if q.startswith(subject+'_') and len(clips)>=4)
        rng.shuffle(seqs);assert len(seqs)>=3
        for seq in seqs[:3]:
            clips=sorted(eligible[seq]);rng.shuffle(clips)
            fresh.append(dict(split='fresh_eval',subject=subject,sequence=seq,clips=sorted(clips[:4])))
    protocol=dict(study='dit_subject_oof_v2',seed=20261003,subjects=SUBJECTS,
        folds={s:dict(held_subject=s,coarse_training_subjects=sorted(set(SUBJECTS)-{s}),
            coarse_training_roles=['coarse'],initialization='COCO only') for s in SUBJECTS},
        denoiser_sequences=sorted(set().union(*by.values())-set(gate_sequences)),gate_fit_sequences=gate_sequences,
        model_selection_sequence=tune_seq[0],probability_calibration_sequence=tune_seq[1],selection_subject='P0003',
        old_test_policy='P0010/P0015 v1 test is now development-only; no independent claim',
        fresh_sequences=fresh,fresh_policy='New uninspected source sequences; subjects P0010/P0015 are reused, not new subjects',
        input_policy='Common frozen COCO RGB features across all fold models; OOF 3D coarse and raw uncertainty only; GT labels never inference inputs',
        final_coarse=str(OLD/'coarse_fine/best.pt'),final_coarse_sha256=sha(OLD/'coarse_fine/best.pt'),
        detector=str(OLD/'detector/weights/best.pt'),detector_sha256=sha(OLD/'detector/weights/best.pt'),
        methods=['oof_dit','oof_regression','in_subject_dit'],steps=5000,gate_steps=2500,
        matched_control='In-subject DiT uses the identical samples, split, RGB basis, model capacity and calibration policy; only coarse prediction provenance differs',
        confidence_calibration='Fit root and per-joint variance scaling on OOF denoiser sequences only, equal subject weight; separate fit for in-subject control',
        gate_calibration='Train gate only on source sequences never used for denoiser loss; logistic probability calibration and threshold/strength on a separate P0003 sequence',
        preservation='Initially correct <=10mm joints harmed >1mm <=5percent in camera and wrist-relative spaces; fallback zero update if no admissible point',
        success='>=5percent hard-group error improvement with clustered95percent CI below zero and preservation passed; same boxes, no detection recall improvement',
        manifest_sha256=sha(OLD/'predicted_manifest.jsonl'))
    save(dest,protocol);save(RUN/'fresh_manifest.json',{**{k:old[k] for k in ['official_repo','revision','mirror','stream']},'sequences':fresh})
    save(RUN/'status.json',dict(stage='fold_training',complete=False))
    print(json.dumps(protocol,indent=2),flush=True)
if __name__=='__main__':main()

