import json
from oof_common import RUN,ROOT,SUBJECTS
def main():
    out={'status':json.loads((RUN/'status.json').read_text())}
    out['folds']={}
    for s in SUBJECTS:
        out['folds'][s]={}
        for phase in ['warmup','fine']:
            f=RUN/'folds'/s/phase/'history.json'
            if f.exists():out['folds'][s][phase]=json.loads(f.read_text())[-1]
    for filename in ['fresh_download_status.json','fresh_done.json','cache_done.json','cache_audit.json']:
        path=RUN/filename
        if path.exists():out[filename]=json.loads(path.read_text())
    out['fresh_pending']=[]
    for s in json.loads((RUN/'fresh_manifest.json').read_text())['sequences']:
        for c in s['clips']:
            tar=ROOT/'rgb_clips'/s['split']/s['sequence']/f'clip-{c:06d}.tar'
            if not tar.with_suffix('.receipt.json').exists():
                part=tar.with_suffix('.tar.partial')
                out['fresh_pending'].append(dict(clip=c,tar_bytes=tar.stat().st_size if tar.exists() else None,partial_bytes=part.stat().st_size if part.exists() else None))
    for method in ['oof_dit','oof_regression','in_subject_dit']:
        path=RUN/f'train_{method}.log'
        if path.exists():out[method]=path.read_text().splitlines()[-1:]
    print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
