"""Seal the complete multi-GPU comparison before any metrics are read."""
import hashlib,json,collections,torch
from hand3d_v8_common import V7,save
from complete_hand_tracks_acceleration_v46 import profile_config
RUN=V7.parent/'acceleration_validation_v46'
def main():
    rows=json.loads((RUN/'fresh_rows.json').read_text());keys={tuple(r[k] for k in ['sequence','clip','track_id']) for r in rows}
    for shard in range(4):
        frozen=json.loads((RUN/f'outputs_frozen_{shard}.json').read_text());assert frozen['complete']
        for name,h in frozen['hashes'].items():assert hashlib.sha256((RUN/'tracks'/name).read_bytes()).hexdigest()==h,name
        parity=json.loads((RUN/f'cached_adapter_parity_{shard}.json').read_text());assert parity['passed']
    affected_context=set(json.loads((RUN/'context_adapter_audit.json').read_text())['affected_tracks'])
    outputs=[];side_bank=torch.tensor(json.loads((RUN/'side_adapter_audit.json').read_text())['predicted_right'],dtype=torch.bool)
    for key in keys:
        tid='_'.join(map(str,key))
        inp=torch.load(RUN/'tracks'/f'{tid}_inputs.pt',weights_only=False,map_location='cpu')
        if tid in affected_context:assert inp.get('context_version')=='bounded_v46',tid
        used=inp['observations']['right'].bool()^inp['observations']['parameter_side_outlier']
        assert torch.equal(used,side_bank[torch.tensor(inp['indices'])]),tid
        selection=json.loads((RUN/'tracks'/f'{tid}_selection.json').read_text());assert selection['gt_poison_exact']
        for p in ['strict','acc_x2']:
            path=RUN/'tracks'/f'{tid}_{p}.pt';assert path.exists();outputs.append(path)
            check=json.loads((RUN/'tracks'/f'{tid}_{p}_redecode.json').read_text());assert check['passed'] and check['check']['passed']
            assert check['check']['limits']=={k:profile_config(p)[k] for k in check['check']['limits']}
    save(RUN/'outputs_frozen.json',dict(complete=True,tracks=len(keys),observations=len(rows),
        gt_used_in_inference=False,all_outputs_frozen_before_evaluation=True,same_candidates_and_path=True,
        saved_parameter_redecode_passed=True,selector_gt_poison_exact=True,online_cached_adapter_parity=True,
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
        profiles={p:profile_config(p) for p in ['strict','acc_x2']},default_changed=False))
    print(json.dumps(dict(sealed=True,tracks=len(keys),observations=len(rows))))
if __name__=='__main__':main()
