"""Capture a predeclared diagnostic grid with the unchanged live LightESS engine."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
R=Path('/mnt/why/native_depth_audit_v8_20261019')
PUBLIC=Path('/mnt/why/public_headhand_h20_20261019')
VIEW=Path('/mnt/why/native_mano_dit_v2_20261019/coherent_frontend')
FRAMES=[0,20,40,60,80,100,120,140,148]
EXPECTED='6b29afdb3411601e93ccbd034d04f9eaa0c63fb5603ef987e58f369e5265209c'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3'
    mem=subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    assert int(mem.strip())==0, f'GPU3 is occupied: {mem}'
    source=PUBLIC/'src/run_depth.py';original=source.read_text();assert sha(source)==EXPECTED
    changes={
      'for frame_index in range(0, len(frames), 2):':f'for frame_index in {FRAMES}:',
      'if frame_index == 0:':'if True:  # save every predeclared diagnostic frame',
      "f'example_{target}.npz'":"f'frame{frame_index:03d}_{target}.npz'"}
    changed=original
    for a,b in changes.items():assert changed.count(a)==1;changed=changed.replace(a,b)
    (R/'snapshots').mkdir(exist_ok=True);snapshot=R/'snapshots/run_depth_capture.py';assert not snapshot.exists();snapshot.write_text(changed)
    protocol=dict(frames=FRAMES,frame_selection='fixed 20-frame spacing plus final stereo frame; historically exposed clip2102 diagnostic',
      changes=changes,run_depth_source_sha256=EXPECTED,capture_source_sha256=sha(snapshot),GT_inputs=False,
      variants=['baseline','erode4','iter5','erode4_iter5'],erosion_native_pixels=4,iterations=5,
      inference_rule='same confidence>=0.2, <=96 uniformly spaced mask samples/view, >=6 valid hits and MAD<0.03m; iteration translates fixed-shape mesh along original predicted wrist ray; invalid candidate falls back to baseline',
      production_admitted=False)
    (R/'protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(protocol),flush=True)
    sys.path.insert(0,str(PUBLIC/'src'))
    sys.argv=[str(snapshot),'--input',str(VIEW/'data/clip-002102'),'--coarse',str(VIEW/'outputs/wilor_clip2102'),'--rf',str(VIEW/'outputs/rf_clip2102'),'--output',str(R/'stereo_grid')]
    exec(compile(changed,str(snapshot),'exec'),dict(__name__='__main__',__file__=str(snapshot)))
    files={str(p.relative_to(R/'stereo_grid')):sha(p) for p in (R/'stereo_grid').iterdir() if p.is_file()}
    (R/'stereo_frozen.json').write_text(json.dumps(dict(complete=True,files_sha256=files,protocol_sha256=sha(R/'protocol.json'),GT_inputs=False),indent=2))
if __name__=='__main__':main()
