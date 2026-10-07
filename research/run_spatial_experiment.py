import subprocess,sys
from pathlib import Path
CODE=Path(__file__).resolve().parent
for script in ['check_spatial_pixels.py','run_spatial_stage1.py','run_spatial_stage2.py','check_spatial_models.py','evaluate_spatial_temporal.py']:
    subprocess.run([sys.executable,'-u',script],cwd=CODE,check=True)
