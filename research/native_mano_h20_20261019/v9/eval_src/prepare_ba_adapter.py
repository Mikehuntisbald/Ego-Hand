"""Create an independent immutable BA copy; only add the image-observation input."""
import hashlib,json
from pathlib import Path
R=Path('/mnt/why/native_rgb_observation_v9_20261019');BASE=Path('/mnt/why/native_mano_dit_v2_20261019/src/native_ba.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    source=BASE.read_text();changes={
      "ap.add_argument('--suffix',default='');":"ap.add_argument('--image-observations',type=Path);ap.add_argument('--suffix',default='');",
      "    with torch.no_grad():\n        test=":"    if args.image_observations is not None:\n        observed=np.load(args.image_observations)\n        assert np.array_equal(observed['timestamp_ns'],times) and np.array_equal(observed['track_id'],track)\n        assert observed['uv_blended_native'].shape==(N,21,2) and np.isfinite(observed['uv_blended_native']).all()\n        uv=tensor(observed['uv_blended_native'])\n    with torch.no_grad():\n        test=",
      "observation='WiLoR projected 2D, not independent image keypoint labels',":"observation='Independent RGB native-MANO amodal head blended with original WiLoR self-projection' if args.image_observations is not None else 'WiLoR projected 2D, not independent image keypoint labels',",
      "                objective_before=before,":"                image_observations=str(args.image_observations),image_observations_sha256=hashlib.sha256(args.image_observations.read_bytes()).hexdigest() if args.image_observations is not None else None,original_WiLoR_reprojection_mean_px=float(np.linalg.norm(uv_end-np.asarray([r['xy'] for r in rows]),axis=-1).mean()),\n                objective_before=before,",
      "if __name__=='__main__':main()":"def forbid(event,args):\n    if event=='open' and isinstance(args[0],(str,bytes)):\n        name=str(args[0]);assert not any(x in name for x in ['.hands.json','__hand_shapes.json__','native_targets.pt','/targets.pt','sealed_scoring_arrays.npz']),name\nif __name__=='__main__':\n    sys.addaudithook(forbid)\n    main()"}
    for a,b in changes.items():assert source.count(a)==1,a;source=source.replace(a,b)
    target=R/'eval_src/native_ba_rgb.py';assert not target.exists();target.write_text(source);(R/'receipts/BA_adapter.json').write_text(json.dumps(dict(base_sha256=sha(BASE),adapter_sha256=sha(target),changes=changes,GT_inputs=False,objective_weights_unchanged=True,production_admitted=False),indent=2))
if __name__=='__main__':main()
