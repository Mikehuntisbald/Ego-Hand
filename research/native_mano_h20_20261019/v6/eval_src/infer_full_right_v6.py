"""Use the same right-only native sampler in the full public BA interface."""
import hashlib,json,sys
from pathlib import Path
from infer_v6 import predict_right,forbid
import infer_native
sys.addaudithook(forbid)
infer_native.predict=predict_right
if __name__=='__main__':
    infer_native.main()
    out=Path(sys.argv[sys.argv.index('--output')+1]);path=out/'seal.json';seal=json.loads(path.read_text());seal.update(right_only=True,left_passthrough=True,supervision='native MANO rotations; loss-only shape-compatible FK; actual native GT evaluation',adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());path.write_text(json.dumps(seal,indent=2))
