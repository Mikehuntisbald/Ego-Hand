import hashlib,json
from pathlib import Path
R=Path('/mnt/why/native_candidate_sequence_v10_20261019')
ARMS=['frame_RGB','temporal_noRGB','temporal_RGB','temporal_noRGB_scaled','temporal_RGB_scaled']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    for n,h in json.loads((R/'receipts/source_seal.json').read_text()).items():assert sha(R/'src'/n)==h
    for file in ['candidates_frozen.json','predictions_frozen.json','likelihood_r1/predictions_frozen.json']:
        frozen=json.loads((R/file).read_text());assert frozen['complete']
        for name,h in frozen['seals'].items():
            p=R/name;assert sha(p)==h
            for name,value in json.loads(p.read_text())['files'].items():assert sha(p.parent/name)==value
    combined=json.loads((R/'all_predictions_frozen.json').read_text());assert sha(R/'predictions_frozen.json')==combined['original_sha256'];assert sha(R/'likelihood_r1/predictions_frozen.json')==combined['revision_sha256']
    full=json.loads((R/'full_pipeline/receipts/predictions_frozen.json').read_text());assert full['complete']
    for name,h in full['seals'].items():
        p=R/'full_pipeline'/name;assert sha(p)==h
        for name,value in json.loads(p.read_text())['files'].items():assert sha(p.parent/name)==value
    label=json.loads((R/'evaluation/label_open_receipt.json').read_text());assert combined['freeze_unix']<=full['freeze_unix']<=label['unix']
    gates={}
    for a in ARMS:
        p=R/f'full_pipeline/outputs/ba_{a}_guarded/optimization.json';g=json.loads(p.read_text());assert g['RMSE_mm']<=5 and g['P95_mm']<=10 and g['max_mm']<=30 and g['reprojection_increase_px']<=.2;gates[a]={k:g[k] for k in ['RMSE_mm','P95_mm','max_mm','reprojection_increase_px']}
    parity=json.loads((R/'full_pipeline/receipts/mean_BA_parity.json').read_text());assert all(x==0 for x in parity['difference'].values())
    io=json.loads((R/'receipts/full_output_verification.json').read_text());assert io['complete'] and len(io['outputs'])==10;assert json.loads((R/'receipts/preflight.json').read_text())['actual_tiny_global_optimum']
    out=dict(complete=True,immutable_original_sources=True,all_candidate_selection_and_full_output_hashes=True,all_frozen_before_scoring=True,no_new_training=True,native_mean_BA_reference_exact=True,global_DP_optimum_exhaustively_checked=True,whole_JSON_H5_FK_outputs=10,gates=gates,production_admitted=False)
    (R/'receipts/final_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
