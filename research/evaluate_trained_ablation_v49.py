"""Same-warmstart trained ablations, distinct from fixed-weight dependence probes."""
import hashlib,json
from pathlib import Path
import ablate_modules_v49 as evaluate
ROOT=Path('/mnt/why/HOT3D/experiments/trained_module_ablation_v49_20261007')
NAMES=dict(full='完整保护联训（复用严格匹配v48对照）',no_teacher='去教师保护（复用严格匹配v48对照，未准入）',
    frozen_same_start='同起点冻结视觉主干',no_original_protection='去WiLoR好点保护',no_protection='去两类好点保护',
    no_motion_supervision='去训练动作监督',no_rgb='去额外RGB特征后续训',center_only='仅当前帧后续训',pooled_rgb='空间特征取均值后续训')
def main():
    evaluate.RUN=ROOT;evaluate.NAMES=NAMES;evaluate.evaluate()
    p=ROOT/'summary.json';s=json.loads(p.read_text());configs={};training={}
    for arm in NAMES:
        folder=ROOT/arm;configs[arm]=json.loads((folder/'config.json').read_text())
        training[arm]={k:json.loads((folder/(k+'.json')).read_text()) for k in ['done','history','preflight','resume_verification']}
        if (folder/'pilot_verification.json').exists():training[arm]['actual_updates']=json.loads((folder/'pilot_verification.json').read_text())
        assert training[arm]['done']['steps']==1800 and training[arm]['done']['cumulative_steps']==2400
    assert len({x['initial_checkpoint_sha256'] for x in configs.values()})==1
    assert len({x['batch_plan_sha256'] for x in configs.values()})==1
    s.update(training_ablation_complete=True,training=training,configs=configs,
        exact_same_warmstart=True,exact_same_batch_plan=True,equal_update_and_exposure_budget=True,equal_GPU_compute=False,
        experiment_scope='Adaptation ablations from same already-trained v47 joint600 weights,1800 additional updates each. Not from-scratch architectural comparison or fresh generalization.',
        no_default_changed=True,loss_targets_only=True)
    for dst in [p,ROOT/'review/summary.json']:dst.write_text(json.dumps(s,indent=2))
    page=ROOT/'review/report.html';text=page.read_text()
    text=text.replace('HOT3D v49 模块消融','HOT3D v49 同起点重训消融').replace('模块消融：逐项去掉，观察贡献与代价','同起点重训消融：各模块经过适应后的贡献')
    text=text.replace('屏蔽RGB/时间/空间是分布外依赖测试，不能当作对应重训后的收益。','RGB/时间/空间组已经按相同起点、批次与预算续训。每组1800步/有效批量4；同一已训练起点。这是续训适应消融，并非从零训练；GPU耗时不同，不声称等计算预算。')
    page.write_text(text,encoding='utf-8')
    print(json.dumps(dict(complete=True,shared_initial=True,shared_plan=True)),flush=True)
if __name__=='__main__':main()
