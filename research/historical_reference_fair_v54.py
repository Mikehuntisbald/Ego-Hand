"""Retain exact published no-mask native outputs, alongside common-runtime tests.

The common tracker/coarse adapter is a controlled comparison, not a byte-exact
reproduction of the historical complete pipeline. Preserve both references.
"""
import argparse,collections,json,shutil,time,os,subprocess
from pathlib import Path
import torch
from prepare_fair_v54 import ROOT,E,SOURCE,save,sha

REFS={'historical_yolo_v43':'initial','historical_yolo_v48':'protected_best'}
def prepare():
    rows=json.loads((SOURCE/'diagnostic_v46/rows.json').read_text());clips=json.loads((ROOT/'replay_clips.json').read_text());groups=collections.defaultdict(list)
    for i,r in enumerate(rows):groups[(r['sequence'],r['clip'],r['track_id'])].append((i,r))
    for name,old_mode in REFS.items():
        folder=E/'online_rgb_iterative_v48/diagnostic_v46'/old_mode
        seal=json.loads((folder/'sealed.json').read_text());assert seal['complete'] and sha(folder/'result.pt')==seal['result_sha256']
        result=torch.load(folder/'result.pt',weights_only=False,map_location='cpu');pred=result['prediction'];assert len(pred)==len(rows) and result['check']['passed']
        for clip in clips:
            dest=ROOT/'historical_references'/name/clip['id'];dest.mkdir(parents=True,exist_ok=True)
            tracks=[]
            for (seq,cid,tid),items in groups.items():
                if seq!=clip['sequence'] or cid!=clip['clip']:continue
                frames=[]
                for i,r in items:
                    frames.append(dict(image=r['image'],camera=r['camera'],timestamp_s=r['timestamp_ns']/1e9,box_xyxy=r['box'],box_confidence=r['score'],candidate_xyz_camera_m=pred[i].tolist()))
                tracks.append(dict(id=f'historical_{tid}',frames=frames))
            output=dict(tracks=tracks,constraint_checks_passed=result['check']['passed'],raw_xyz_fallback_frames=0,GT_free=True,instance_masks_used=False,
                        source_original_seal=seal,scope='Exact previously sealed v43-weight/v46-runtime or v48-protected output; original YOLO .01/top10, original tracking and coarse observations; native only.',default_replaced=False)
            save(dest/'prediction.json',output);save(dest/'freeze.json',dict(complete=True,GT_free=True,prediction_sha256=sha(dest/'prediction.json'),original_result_sha256=seal['result_sha256'],original_checkpoint_sha256=seal['checkpoint_sha256']))
    path=ROOT/'protocol.json';p=json.loads(path.read_text());archive=ROOT/'protocol_before_historical_reference_addendum.json'
    if not archive.exists():shutil.copy2(path,archive)
    p['historical_reference_addendum']=dict(created_unix=time.time(),before_new_test_labels=True,models=list(REFS),native_original_outputs_preserved_exact=True,
      reason='Original no-mask versions also used different detection (.01/top10), association and coarse reconstruction. Include their sealed native outputs so a new common tracker cannot weaken the formal historical reference.',
      common_tracker_rows='Controlled runtime comparisons, separate from actual historical native outputs.',
      clinical_reference='No historical calibrated clinical run exists; original no-mask weights use the common pinhole adapter for clinical/nail comparisons. Historical rows alias those adapter outputs outside native.')
    save(path,p);save(ROOT/'historical_reference_prepared.json',dict(complete=True,native_clips=len(clips),models=list(REFS),GT_free=True,original_numeric_predictions_unchanged=True,protocol_sha256=sha(path)))
    print('EXACT HISTORICAL NO-MASK REFERENCES PREPARED',flush=True)
def final():
    if (ROOT/'historical_reference_finalized.json').exists():return
    start=time.time()
    while not (ROOT/'done.json').exists():
        status=ROOT/'controller_status.json'
        if status.exists() and json.loads(status.read_text()).get('stage')=='needs_attention':raise RuntimeError('Main controller needs attention; evidence preserved')
        if time.time()-start>24*3600:raise TimeoutError('Finite finalizer wait exceeded')
        time.sleep(30)
    for historical,old in [('historical_yolo_v43','yolo_v43'),('historical_yolo_v48','yolo_v48')]:
        for tag in ['glove','nail']:
            src=ROOT/'inference'/old/tag;dst=ROOT/'historical_references'/historical/tag;dst.mkdir(parents=True,exist_ok=True)
            for filename in ['prediction.json','freeze.json']:shutil.copy2(src/filename,dst/filename)
    # Preserve the first common-runtime report/statistics before producing the
    # expanded report. Existing sources, weights and predictions stay frozen.
    archive=ROOT/'review_common_runtime'
    if not archive.exists():shutil.copytree(ROOT/'review',archive)
    if not (ROOT/'summary_common_runtime.json').exists():shutil.copy2(ROOT/'summary.json',ROOT/'summary_common_runtime.json')
    source=(ROOT/'code/score_fair_v54.py').read_text()
    replacements={
      'from infer_fair_v54 import MODES':"from infer_fair_v54 import MODES as BASE_MODES\nMODES=BASE_MODES+['historical_yolo_v43','historical_yolo_v48']",
      "folder=ROOT/'inference'/mode/tag":"folder=ROOT/('historical_references' if mode.startswith('historical_') else 'inference')/mode/tag",
      "for ref in ['yolo_v43','yolo_v48','rf_v53']:":"for ref in ['historical_yolo_v43','historical_yolo_v48','yolo_v43','yolo_v48','rf_v53']:",
      "display=['yolo_v43'":"display=['historical_yolo_v43','historical_yolo_v48','yolo_v43'",
      "c=v['comparisons']['yolo_v48']":"c=v['comparisons']['historical_yolo_v48']",
      '相对v48恢复':'相对历史无mask v48恢复',
      "comp=dict(old_good_points=good":"comp=dict(old_good_points=good",
      "common_matched_error=rel_sum/max(common_n,1)":"common_matched_error=rel_sum/common_n if common_n else None",
      "common_matched_camera_mm=cam_sum/max(common_n,1)":"common_matched_camera_mm=cam_sum/common_n if common_n else None",
    }
    for old,new in replacements.items():
        assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
    target=ROOT/'score_with_historical_reference_r1.py';target.write_text(source)
    ns=dict(__name__='historical_reference_scorer',__file__=str(target));exec(compile(source,str(target),'exec'),ns);ns['main']()
    save(ROOT/'historical_reference_finalized.json',dict(complete=True,exact_native_reference=True,common_runtime_report_preserved=True,new_test_labels_read_only_after_all_predictions_sealed=True,default_changed=False))
    print('HISTORICAL REFERENCES INCLUDED IN FINAL REPORT',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--finalize',action='store_true');p.add_argument('--start_finalize',action='store_true');a=p.parse_args()
    if a.start_finalize:
        from prepare_fair_v54 import sha
        save(ROOT/'historical_finalizer_code.json',dict(script=str(Path(__file__).resolve()),sha256=sha(__file__),isolated_revision=True))
        with (ROOT/'historical_finalizer.log').open('a') as log:
            process=subprocess.Popen([str(E/'yolo26_wilor_3d_20261003/venv/bin/python'),__file__,'--finalize'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONPATH=str(ROOT/'code')))
        save(ROOT/'historical_finalizer_pid.json',dict(pid=process.pid,finite=True));print(process.pid,flush=True)
    elif a.prepare:prepare()
    else:final()
