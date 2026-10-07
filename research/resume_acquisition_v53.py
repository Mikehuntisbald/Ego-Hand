"""Wait for our first download process; preserve failures and reuse CRC-verified files."""
import os,time,json,shutil,subprocess
from pathlib import Path
A=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53_20261007')
CODE=Path('/mnt/why/hot3d_hand_residual')
PY='/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python'
def main():
    while Path('/proc/1490943').exists():time.sleep(30)
    if (A/'100doh_done.json').exists():return
    err=A/'errors/100doh_acquisition';err.mkdir(parents=True,exist_ok=True)
    for source in [A/'100doh_acquire.log',CODE/'acquire_100doh_v53.py',CODE/'remote_range_zip_v53.py']:
        shutil.copy2(source,err/source.name)
    for attempt in range(1,4):
        with (A/f'100doh_resume_{attempt}.log').open('w') as log:
            ret=subprocess.run([PY,'-u',str(CODE/'acquire_100doh_v53.py')],cwd=CODE,stdout=log,stderr=subprocess.STDOUT).returncode
        if ret==0:
            (A/'100doh_acquisition_resume.json').write_text(json.dumps(dict(success=True,attempt=attempt,reused_files_by_exact_CRC=True,original_error_preserved=str(err)),indent=2));return
    raise RuntimeError('Repeated acquisition error; all logs retained; do not begin incomplete training')
if __name__=='__main__':main()
