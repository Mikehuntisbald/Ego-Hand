import hashlib,json
from pathlib import Path
RUN=Path('/mnt/why/HOT3D/experiments/dit_wilor_v5')
r=json.loads((RUN/'locked_results.json').read_text());assert r['accepted']
assert all(v['passed'] for v in r['methods']['dit']['evidence'].values())
assert r['methods']['dit']['held_subject_no_regression']
receipt=json.loads((RUN/'delivery_archive.json').read_text())
assert receipt['sha256']=='99ab56a5596c7edd7f840abbd5d7dd87e540b280087095ee4be216d6ea1a0572'
assert hashlib.sha256(Path(receipt['path']).read_bytes()).hexdigest()==receipt['sha256']
checks=json.loads((RUN/'sealed_api_checks.json').read_text())
assert all(all(v.values()) for v in checks.values())
status=dict(stage='delivered',complete=True,acceptance_passed=True,delivery_complete=True,
    local_archive_sha256_verified=receipt['sha256'],local_packaged_files_verified=257,
    visualization_reviewed=True,archive=receipt['path'],report=str(RUN/'delivery/report.html'),
    scope=r['scope'],limitations='DiT beats regression in ray-aligned group, but regression has slightly lower overall wrist-relative error; v3/v4 failed evaluations retained')
(RUN/'status.json').write_text(json.dumps(status,indent=2))
print(json.dumps(status,indent=2))
