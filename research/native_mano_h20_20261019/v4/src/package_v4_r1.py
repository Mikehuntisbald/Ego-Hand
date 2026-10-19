"""Compact training/validation/full-chain evidence without third-party assets."""
import csv,hashlib,html,json,shutil,tarfile
from pathlib import Path
import torch
R=Path('/mnt/why/native_mano_rollout_v4_20261019');VALIDATION=Path('/mnt/why/native_mano_multiseq_v3_20261019');V2=Path('/mnt/why/native_mano_dit_v2_20261019')

def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert load(R/'receipts/full_pipeline_done.json')['complete']
    validation=load(VALIDATION/'evaluation_r3/metrics.json');d=R/'delivery';d.mkdir(exist_ok=False)
    for name in ['src','receipts','models','training','reviews','predictions','outputs','metrics']:(d/name).mkdir()
    base=d/'src/base_v2';base.mkdir()
    for name in ['native_dit.py','native_mano.py','native_data.py','train_native.py','shared_noise.py','infer_native.py','native_ba.py']:shutil.copy2(V2/'src'/name,base/name)
    shutil.copytree(VALIDATION/'src',d/'src/validation',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(R/'src',d/'src/rollout',ignore=shutil.ignore_patterns('__pycache__'))
    for root,label in [(R,'rollout'),(VALIDATION,'validation')]:
        shutil.copytree(root/'receipts',d/'receipts'/label)
        for p in root.glob('*protocol*.json'):shutil.copy2(p,d/'receipts'/f'{label}_{p.name}')
    shutil.copy2(R/'inference_lock.json',d/'receipts/rollout_inference_lock.json')
    shutil.copytree(VALIDATION/'evaluation_r3',d/'reviews/validation')
    protocol=load(VALIDATION/'protocol.json');models={k:Path(v['path']) for k,v in protocol['models'].items()}
    models.update({f'rollout_{a}':R/f'runs/{a}/best_development.pt' for a in ['dit','protected']})
    for arm,path in models.items():
        ck=torch.load(path,weights_only=False,map_location='cpu');keep={k:ck[k] for k in ['model','config','step','selection','batch_plan_sha','arm','GT_inference']}
        keep.update(full_source_checkpoint_sha256=sha(path),actual_rollout_training=ck.get('actual_rollout_training',False))
        torch.save(keep,d/f'models/{arm}_development.pt')
    for arm in ['dit','protected']:
        for name in ['done.json','selection.json','progress.json']:shutil.copy2(R/f'runs/{arm}'/name,d/'training'/f'{arm}_{name}')
        shutil.copy2(R/'logs'/f'{arm}_r1.log',d/'training'/f'{arm}.log')
    for clip in protocol['clips']:
        view=Path(clip['view']);target=d/'predictions'/f'clip-{clip["clip"]:06}';target.mkdir()
        for folder in ['predictions','predictions_rollout']:shutil.copytree(view/folder,target/folder)
    full={}
    for arm in ['dit','protected','guarded']:
        review=R/f'comparisons/full_rollout_{arm}/outputs/review_clip2102';shutil.copytree(review,d/f'reviews/full_{arm}');full[arm]=load(review/'metrics.json')
        shutil.copytree(R/f'comparisons/full_rollout_{arm}/receipts',d/f'receipts/full_{arm}')
    for folder in (R/'full_pipeline/outputs').glob('ba_rollout_*'):shutil.copytree(folder,d/'outputs'/folder.name)
    shutil.copytree(V2/'coherent_frontend/outputs/mano_baseline_clip2102',d/'outputs/matched_full_baseline')
    old=load(V2/'comparisons/coherent_cosine_guarded/outputs/review_clip2102/metrics.json')
    shutil.copy2(V2/'comparisons/coherent_cosine_guarded/outputs/review_clip2102/metrics.json',d/'metrics/v2_full_guarded.json')
    summary=dict(complete=True,validation=validation,full_pipeline=full,old_v2_guarded=old,training={a:load(R/f'runs/{a}/done.json') for a in ['dit','protected']},
        actual_resume=load(R/'receipts/actual_resume_verification_r1.json'),sampler_preflight=load(R/'receipts/preflight.json'),production_admitted=False,
        source='public H20 implementation, not private production/GT9',supervision='same fitted MANO pseudo-angle GT20 targets and FK/velocity/protection weights; actual deployment sampler provides the 3D loss predictions',
        visual_backbone='frozen WiLoR; RGB adapter/spatial/temporal/DiT trainable',training_compute='same 600 updates and batches, additional sampler backprop costs more; not equal-FLOPs architecture comparison',
        validation_scope='provided reference camera and RF/WiLoR inputs, before BA; six recordings/three subjects, only P0010/P0015 fully unseen by native optimizer and selection',
        full_scope='same reused clip2102, predicted cuVSLAM/physical depth/shared-shape BA; diagnostic regression not fresh generalization',
        source_cache_and_full_optimizer_remote=True,third_party_pretrained_assets_included=False)
    (d/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    labels={'WiLoR':'WiLoR','regression':'时序回归','linear_dit':'线性DiT','cosine_dit':'余弦DiT v2','cosine_protected':'余弦DiT保护损失 v2','cosine_shared':'余弦DiT共享噪声','cosine_protected_displacement':'余弦DiT参数保护 v2','cosine_RGB_off':'余弦DiT RGB置零诊断','rollout_dit':'实际采样训练DiT v4','rollout_protected':'实际采样训练＋准确点保护','rollout_dit_displacement':'实际采样训练＋参数保护'}
    rows=[];flat=[]
    for arm,m in validation['metrics'].items():
        protection=validation['preservation'][arm]['relative_without_wrist'];ci=validation['paired_uncertainty'].get(arm,{}).get('relative',{}).get('subject_cluster',{}).get('CI95_mm')
        interval='—' if ci is None else f'[{ci[0]:.2f}, {ci[1]:.2f}]'
        rows.append(f'<tr><td>{labels[arm]}</td><td>{m["camera_mm"]["mean"]:.2f}</td><td>{m["relative_without_wrist_mm"]["mean"]:.2f}</td><td>{interval}</td><td>{protection["recovered_to20"]}/{protection["base_hard20"]}</td><td>{protection["good_harmed_over1mm"]}/{protection["base_good10"]}</td><td>{protection["good_to_over20"]}</td></tr>')
        flat.append(dict(arm=arm,camera_mean_mm=m['camera_mm']['mean'],relative_mean_mm=m['relative_without_wrist_mm']['mean'],relative_P95_mm=m['relative_without_wrist_mm']['P95'],**protection))
    with (d/'metrics/native_validation.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    strict_rows=[]
    for arm,m in validation['strictly_unseen_native_subjects']['metrics'].items():
        if arm not in ['WiLoR','regression','cosine_dit','cosine_protected_displacement','rollout_dit','rollout_protected','rollout_dit_displacement']:continue
        p=m['relative_preservation'];strict_rows.append(f'<tr><td>{labels[arm]}</td><td>{m["relative_without_wrist_mm"]["mean"]:.2f}</td><td>{p["good_harmed_over1mm"]}/{p["base_good10"]}</td><td>{p["recovered_to20"]}/{p["base_hard20"]}</td></tr>')
    full_rows=[]
    for label,m in [('同一完整链路基础BA',full['dit']['metrics']['BA']),('v2余弦DiT＋BA后保护',old['metrics']['DiT_BA']),('v4实际采样DiT＋BA',full['dit']['metrics']['DiT_BA']),('v4准确点保护训练＋BA',full['protected']['metrics']['DiT_BA']),('v4实际采样DiT＋BA后保护',full['guarded']['metrics']['DiT_BA'])]:
        e=m['camera_mapped20_mm'];full_rows.append(f'<tr><td>{label}</td><td>{e["mean"]:.2f}</td><td>{e["P95"]:.2f}</td><td>{e["max"]:.2f}</td><td>{m["relative_mapped20_mm"]["mean"]:.2f}</td></tr>')
    main_candidate='rollout_dit_displacement';base=validation['metrics']['WiLoR']['relative_without_wrist_mm']['mean'];candidate=validation['metrics'][main_candidate]['relative_without_wrist_mm']['mean'];harm=validation['preservation'][main_candidate]['relative_without_wrist']
    figures=[]
    for p in sorted((d/'reviews/validation/images').glob('*.jpg'))[:4]:figures.append(f'<figure><img src="{p.relative_to(d).as_posix()}"><figcaption>自然RGB；左为原图，中为WiLoR，右为实际采样训练DiT＋参数保护，绿色为GT20。封存后按改善／损伤／未解决误差选图。</figcaption></figure>')
    document=f'''<!doctype html><meta charset="utf-8"><title>原生MANO DiT：实际采样训练与跨录制验证</title><style>body{{font:16px system-ui;max-width:1550px;margin:32px auto;padding:0 24px;background:#14191e;color:#eee}}table{{border-collapse:collapse;margin:24px 0}}th,td{{border:1px solid #66737c;padding:10px}}img{{width:100%}}pre{{white-space:pre-wrap}}a{{color:#7cd4ff}}.note{{padding:18px;background:#25333e;border-radius:10px}}</style><h1>原生MANO DiT：实际采样训练与跨录制验证</h1>
<p class="note">六段录制、三人员，补全前腕对齐均值 {base:.2f} mm，实际采样训练＋参数保护 {candidate:.2f} mm。困难点恢复 {harm['recovered_to20']}/{harm['base_hard20']}，原准确点损伤&gt;1mm为 {harm['good_harmed_over1mm']}/{harm['base_good10']}；仍不替换生产默认。判断须结合下方人员级区间和损伤。</p>
<h2>这次改变什么</h2><p>架构仍为17帧非causal RGB空间特征→时序注意力→原生MANO旋转DiT→FK。训练中的3D、参数、真实时间运动和准确点保护损失，改为检查实际4采样／10步去噪后的输出；原v-prediction损失、各权重、304个训练窗口、600更新、初始种子和批次计划保持相同。新增保护损失／无保护两组。不是增加损失权重制造对照，训练计算更多，尚非等算力比较。</p>
<p>视觉主干仍冻结，适配层/空间/时序/DiT均有真实梯度与更新。参数监督仍是GT20拟合伪角度；推理无手部GT、无20点IK。实际采样与推理输出一致、采样梯度能到RGB与时序。混合精度缓存预检错误及非确定性恢复误差均封存；确定性设置和CPU读取checkpoint后，真实3＋3步恢复与连续6步的权重、优化器及所有RNG逐位一致。该小检查不代替长训练质量验证。</p>
<h2>六录制：只看补全模块</h2><p>900帧，{validation['counts']['predictions']}个预测，{validation['counts']['matched_hand_frames']}个匹配手帧；全部输出封存后才读取这轮标签。共用提供的参考相机轨迹、RF框和预测轨迹，尚不包含预测头部、物理深度或BA。腕对齐统计去掉恒为零的腕点，单位mm；区间为相对WiLoR差值的人员级95%bootstrap，负数为改善。只有三个人，属于探索性证据。</p>
<table><tr><th>方法</th><th>相机均值</th><th>腕对齐均值（19点）</th><th>人员级差值95%CI</th><th>困难点恢复</th><th>原准确点损伤&gt;1mm</th><th>原准确点变&gt;20mm</th></tr>{''.join(rows)}</table>
<h2>进一步排除参与过权重选择的人员</h2><p>P0003另一录制曾用于选择权重。因此单列P0010/P0015四录制，两人均未参与原生头训练和权重选择；仍只有两个人，不能将几万关节点当作几万个独立样本。</p><table><tr><th>方法</th><th>腕对齐均值mm</th><th>准确点损伤&gt;1mm</th><th>困难点恢复</th></tr>{''.join(strict_rows)}</table>
<h2>完整链路：保持相同头部、深度和BA</h2><p>另在此前已查看的clip2102（150帧／6000点）重放完整链路：预测cuVSLAM、LightESS/UniDepth和BA均固定，只换补全权重。此表是开发诊断，不与上表的参考相机/BA前条件混比。</p><table><tr><th>方案</th><th>相机均值mm</th><th>P95</th><th>最大</th><th>腕对齐均值（含腕20点）</th></tr>{''.join(full_rows)}</table>
<p><a href="reviews/validation/report.html">六段全部自然案例和详细统计</a> · <a href="reviews/full_dit/report.html">完整链路原始候选</a> · <a href="reviews/full_protected/report.html">准确点保护训练候选</a> · <a href="reviews/full_guarded/report.html">BA后保护候选</a></p>{''.join(figures)}
<h2>边界与交付</h2><p>公共验证录制历史上已用于其他项目实验，不称整个项目的新鲜测试；P0010/P0015是本原生头未见人员。所有参数、完整链路JSON/H5、权重、源码、封存和哈希均附。第三方预训练资产和RGB特征缓存留在H20。未修改私有5090产线，未开启定时任务，未发布。仍需更广人员/场景、独立RGB定位与真实MANO监督等证据。</p>
<details><summary>完整统计、保护与置信区间</summary><pre>{html.escape(json.dumps(summary,ensure_ascii=False,indent=2))}</pre></details>'''
    (d/'report.html').write_text(document)
    if (R/'README.md').exists():shutil.copy2(R/'README.md',d/'README.md')
    files={str(p.relative_to(d)):sha(p) for p in d.rglob('*') if p.is_file()};(d/'delivery_files.json').write_text(json.dumps(dict(complete=True,files=files),indent=2))
    archive=R/'delivery_native_mano_v4.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(d,arcname='delivery')
    receipt=dict(complete=True,bytes=archive.stat().st_size,sha256=sha(archive),archive=str(archive),files=len(files));(R/'delivery_archive.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))

if __name__=='__main__':main()
