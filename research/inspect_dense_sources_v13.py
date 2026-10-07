import json,collections
from hand3d_v8_common import V7,load,save
import spatial_rgb_common as s

def main():
    data=load('cpu');records,_=s.records_and_index();groups=collections.defaultdict(set)
    for i,role in enumerate(data['roles']):
        r=records[int(data['feature_ids'][i,8])-1];groups[(r['sequence'],int(r['clip']))].add(role)
    manifest=json.loads((s.common.ROOT/'subset_manifest.json').read_text())
    print(json.dumps(dict(record_keys=list(records[0]),example={k:records[0][k] for k in ['source','sequence','clip','image']},manifest_keys=list(manifest),manifest_sequence_example=manifest['sequences'][0],clip_count=len(groups),role_clip_counts=dict(collections.Counter('+'.join(sorted(v)) for v in groups.values()))),indent=2),flush=True)
    save(V7.parent/'dense_sampling_v13_source_roles.json',[dict(sequence=k[0],clip=k[1],roles=sorted(v)) for k,v in groups.items()])

if __name__=='__main__':main()
