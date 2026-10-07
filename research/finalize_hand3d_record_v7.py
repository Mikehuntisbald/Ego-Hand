import json,shutil,hashlib,tarfile
from pathlib import Path
from evaluate_hand3d_bounded_v7 import RUN,BASE,PARENT
O=RUN/'delivery';F=BASE.parent/'offline_hand3d_v7_feasible';D=O/'diagnostics/feasible_projection';D.mkdir(parents=True,exist_ok=True)
for name in ['protocol.json','selection.json','policies.json','test_results.json','evaluation_done.json']:
    shutil.copy2(F/name,D/name)
for p in F.glob('*_calibration.json'):shutil.copy2(p,D/p.name)
for name,source in [('initial_3d',BASE),('protected_3d',PARENT)]:shutil.copy2(source/'test_results.json',O/'training'/name/'test_results.json')
for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,O/'code'/p.name)
notes=O/'DEVELOPMENT_NOTES.txt';notes.write_text(notes.read_text(encoding='utf-8')+'\n8. 额外检查：直接将相机位移和相对位移联合投影到安全集合，也没有通过同一恢复门槛。对应校准/测试结果保留在 diagnostics/feasible_projection。没有据测试指标重新选择检查点或放宽验收阈值。\n',encoding='utf-8')
(O/'manifest.json').write_text(json.dumps({str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in O.rglob('*') if p.is_file() and p.name!='manifest.json'},indent=2))
dest=RUN/'offline_hand3d_v7_delivery.tar.gz'
with tarfile.open(dest,'w:gz',compresslevel=1) as tar:tar.add(O,arcname='offline_hand3d_v7')
(RUN/'delivery_receipt.json').write_text(json.dumps(dict(archive=str(dest),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()),indent=2))
print(json.dumps(dict(files=sum(p.is_file() for p in O.rglob('*')),automatic_approved=False,trained_3d_candidates=True)))
