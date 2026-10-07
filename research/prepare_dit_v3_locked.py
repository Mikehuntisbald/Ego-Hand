"""Download/verify preregistered locked data; never run or score model predictions."""
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed,ProcessPoolExecutor
from download_rgb_subset import download
from export_hand_labels import export_clip

ROOT=Path('/mnt/why/HOT3D');RUN=Path(os.environ.get('HOT3D_DIT_RUN',str(ROOT/'experiments/dit_wilor_v3')))


def save(path,obj):
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)


def main():
    if (RUN/'locked_data_ready.json').exists():return
    manifest=json.loads((RUN/'locked_manifest.json').read_text())
    tree={r['path']:r for r in json.loads((ROOT/'provenance/train_aria_tree.json').read_text())}
    jobs=[(s['split'],s['subject'],s['sequence'],clip) for s in manifest['sequences'] for clip in s['clips']]
    receipts=[]
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures={pool.submit(download,job,manifest,tree):job for job in jobs}
        for future in as_completed(futures):
            receipts.append(future.result())
            status=dict(stage='locked_download',done=len(receipts),total=len(jobs),scored=False)
            save(RUN/'locked_data_status.json',status);print(json.dumps(status),flush=True)
    exports=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for receipt in pool.map(export_clip,jobs):
            exports.append(receipt)
            status=dict(stage='locked_export',done=len(exports),total=len(jobs),scored=False)
            save(RUN/'locked_data_status.json',status);print(json.dumps(status),flush=True)
    save(RUN/'locked_data_receipts.json',dict(downloads=receipts,exports=exports))
    save(RUN/'locked_data_ready.json',dict(complete=True,clips=len(jobs),source_sequences=len(manifest['sequences']),scored=False,
        policy='Separate locked split; never loaded by training/development. No predictions until a sealed model passes development acceptance.'))


if __name__=='__main__':main()
