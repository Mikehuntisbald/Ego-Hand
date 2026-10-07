import json,torch
from hand3d_v8_common import V7,save,metrics
RUN=V7.parent/'side_data_v16';report={}
for variant in ['control','consensus']:
    data=torch.load(RUN/variant/'dense_data.pt',weights_only=False,mmap=True)
    assert set(data['roles'])=={'train','dev_select','dev_calibrate'}
    for role in ['train','dev_select','dev_calibrate']:
        ids=torch.tensor([i for i,r in enumerate(data['roles']) if r==role]);xyz=data['xyz_camera_bank'][data['feature_ids'][ids,8]]
        report[f'{variant}_{role}']=metrics(xyz,data['original_base_for_evaluation'][ids],data['gt'][ids],data['valid'][ids])
control=report['control_dev_calibrate'];candidate=report['consensus_dev_calibrate']
approved=candidate['relative_mm']<control['relative_mm']-.5 and candidate['camera_mm']<=control['camera_mm']+.1 and candidate['camera_good_harm_rate']<=.01 and candidate['relative_good_harm_rate']<=.01
save(RUN/'upstream_results.json',dict(complete=True,approved_for_3d_training=approved,metrics=report,scope='Existingtrain/development only; predicted votes frozen before XYZ; side_GTneverentersXYZ. Matched control holdsRGB/labels/slots fixed. New independent evaluation required for adoption.'))
print(json.dumps(dict(approved_for_3d_training=approved,metrics=report),indent=2),flush=True)
