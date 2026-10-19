"""Consolidated evidence and compact delivery, no pretrained third-party assets."""
import hashlib,html,json,shutil,tarfile
from pathlib import Path
import torch
R=Path('/mnt/why/native_mano_dit_v2_20261019');V1=Path('/mnt/why/native_mano_dit_20261019')
PUBLIC=Path('/mnt/why/public_headhand_h20_20261019/runs/replay_r3')

def load(p):return json.loads(p.read_text())
def fmt(v):return f'{v:.2f}'
def main():
    assert load(R/'receipts/coherent_done.json')['complete']
    d=R/'delivery';d.mkdir(exist_ok=False)
    for directory in ['src','receipts','models','reviews','metrics','parameters','outputs']:(d/directory).mkdir()
    for p in (R/'src').glob('*.py'):shutil.copy2(p,d/'src'/p.name)
    for p in (R/'receipts').iterdir():
        if p.is_file():shutil.copy2(p,d/'receipts'/p.name)
    for arm in ['dit','protected']:
        source=R/f'runs/cosine_{arm}/best_development.pt';ck=torch.load(source,weights_only=False,map_location='cpu')
        keep={k:ck[k] for k in ['model','config','step','selection','batch_plan_sha','arm','GT_inference']}
        keep['full_source_checkpoint_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
        torch.save(keep,d/f'models/cosine_{arm}_development.pt')
        for name in ['done.json','selection.json','progress.json']:shutil.copy2(R/f'runs/cosine_{arm}'/name,d/'metrics'/f'cosine_{arm}_{name}')
    schedule=load(R/'evaluation_schedule/metrics.json');sampling=load(R/'evaluation_sampling/metrics.json')
    shutil.copy2(R/'evaluation_schedule/metrics.json',d/'metrics/schedule.json');shutil.copy2(R/'evaluation_sampling/metrics.json',d/'metrics/sampling.json')
    shutil.copy2(R/'evaluation_schedule/predictions_frozen.json',d/'receipts/schedule_predictions_frozen.json');shutil.copy2(R/'evaluation_sampling/predictions_frozen.json',d/'receipts/sampling_predictions_frozen.json')
    stages={}
    for arm in ['native_raw','native_protected','coherent_regression','coherent_linear_dit','coherent_linear_protected','coherent_cosine_dit','coherent_cosine_protected','coherent_cosine_shared','coherent_cosine_guarded','frontend_repair']:
        source=R/'comparisons'/arm/'outputs/review_clip2102';m=load(source/'metrics.json');stages[arm]=m
        shutil.copy2(source/'metrics.json',d/'metrics'/f'{arm}.json')
        if arm.startswith('coherent_'):
            shutil.copytree(source,d/'reviews'/arm)
            shutil.copytree(R/'comparisons'/arm/'receipts',d/'receipts'/arm)
    for p in (R/'coherent_frontend/outputs').glob('*/parameters.npz'):shutil.copy2(p,d/'parameters'/f'{p.parent.name}.npz')
    for p in (R/'coherent_frontend/outputs').glob('native_*/native_parameters.npz'):shutil.copy2(p,d/'parameters'/f'{p.parent.name}.npz')
    for folder in (R/'coherent_frontend/outputs').glob('*'):
        if (folder/'parameters.npz').exists():
            target=d/'outputs'/folder.name;target.mkdir()
            for name in ['parameters.npz','joints3d_world.json','bundle.h5','optimization.json','seal.json']:shutil.copy2(folder/name,target/name)
    for name in ['ready.json','selection.json']:shutil.copy2(V1/'data'/name,d/'metrics'/f'training_data_{name}')
    shutil.copy2(R/'coherent_frontend/outputs/wilor_clip2102/track_side_decisions.json',d/'receipts/track_side_decisions.json')
    for name in ['depth_clip2102','depth_fused_clip2102','depth_protected_clip2102']:
        shutil.copy2(R/f'coherent_frontend/outputs/{name}/depth.json',d/'metrics'/f'{name}.json')
    base=stages['coherent_regression']['metrics']['BA'];old=load(PUBLIC/'outputs/review_clip2102/metrics.json');v1=load(V1/'comparisons/raw/outputs/review_clip2102/metrics.json')
    labels={'coherent_regression':'时序回归＋保护＋BA','coherent_linear_dit':'线性日程DiT＋保护＋BA','coherent_linear_protected':'线性日程保护DiT＋保护＋BA','coherent_cosine_dit':'余弦日程DiT＋保护＋BA','coherent_cosine_protected':'余弦日程保护DiT＋保护＋BA','coherent_cosine_shared':'余弦DiT＋共享噪声＋保护＋BA','coherent_cosine_guarded':'余弦DiT＋BA后整轨迹参数保护'}
    rows=[]
    for label,m in [('原基础BA',old['metrics']['BA']),('原生DiT原版',v1['metrics']['DiT_BA']),('仅修复BA标记覆盖：基础BA',stages['native_raw']['metrics']['BA']),('仅修复BA标记覆盖：原生DiT',stages['native_raw']['metrics']['DiT_BA']),('一致前端＋基础BA',base)]:
        e=m['camera_mapped20_mm'];rows.append(f'<tr><td>{label}</td><td>{fmt(e["mean"])}</td><td>{fmt(e["P95"])}</td><td>{fmt(e["max"])}</td><td>{fmt(m["relative_mapped20_mm"]["mean"])}</td></tr>')
    for arm,label in labels.items():
        m=stages[arm]['metrics']['DiT_BA'];e=m['camera_mapped20_mm'];rows.append(f'<tr><td>{label}</td><td>{fmt(e["mean"])}</td><td>{fmt(e["P95"])}</td><td>{fmt(e["max"])}</td><td>{fmt(m["relative_mapped20_mm"]["mean"])}</td></tr>')
    temporal=[]
    for arm,label in labels.items():
        t=stages[arm]['temporal']['DiT_BA'];h=stages[arm]['DiT_vs_BA_protection_camera']
        temporal.append(f'<tr><td>{label}</td><td>{h["recovered_to20"]}/{h["base_hard20"]}</td><td>{h["good_harmed_over1mm"]}/{h["base_good10"]}</td><td>{h["good_to_over20mm"]}</td><td>{fmt(t["error_velocity_mm_at_60Hz"]["P95"])}</td><td>{fmt(t["error_acceleration_mm_at_60Hz2"]["P95"])}</td><td>{t["wrist_error_jumps_over10mm"]}/{t["wrist_pairs"]}</td><td>{t["fast_motion_amplitude_ratio"]:.3f}</td><td>{t["bone_length_CV_mean"]:.4f}</td></tr>')
    sched=[]
    for arm,m in sampling['metrics'].items():
        h=sampling['protection_relative'].get(arm,{});t=sampling['temporal'].get(arm,{})
        sched.append(f'<tr><td>{arm}</td><td>{fmt(m["relative"]["mean_mm"])}</td><td>{fmt(m["relative"]["P95_mm"])}</td><td>{h.get("good_harmed_over1mm","—")}</td><td>{h.get("good_to_over20","—")}</td><td>{fmt(t["relative_velocity_error_at60Hz"]["P95_mm"]) if t else "—"}</td></tr>')
    cards=[]
    for arm in ['coherent_linear_protected','coherent_cosine_dit']:
        for p in sorted((d/'reviews'/arm/'images').glob('zoom_*.jpg'))[:2]:cards.append(f'<figure><img src="{p.relative_to(d).as_posix()}"><figcaption>{labels[arm]}：封存后按损伤排序的自然失败案例；左为RGB，中为基础BA，右为候选＋BA，绿点为GT20。</figcaption></figure>')
    links=' '.join(f'<a href="reviews/{arm}/report.html">{label}完整自然案例</a>' for arm,label in labels.items() if arm!='coherent_regression')
    summary=dict(complete=True,scope='reused public HOT3D clip2102 development replay; no fresh generalization evidence',
        stages=stages,schedule=schedule,sampling=sampling,base_coherent=base,final_guard=load(R/'receipts/final_guard_done.json'),safe_default_admitted=False,sequence_CI=None,
        sequence_CI_reason='one development sequence, cannot estimate sequence-level confidence interval',
        architecture='native MANO rotations -> FK -> direct native-parameter BA; no 20-point IK in inference',
        training_supervision='natural GT20 fitted to MANO pseudo-parameters only in training; GT20 FK/true-time loss; not native angle GT',
        preprocessing_repair='confidence-weighted prediction-only track handedness; actual WiLoR crop rerun on 20/300 observations; other 280 exact',
        inference_protection='primary candidate: parameter interpolation before BA only; guarded candidate additionally applies original gates after BA with one scalar per track, SO3 rotations/shared shape, no GT',
        full_checkpoints_remote=str(R/'runs'),pretrained_assets_included=False)
    (d/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    baseline_temporal=stages['coherent_regression']['temporal']['BA'];repair_harm=stages['frontend_repair']['DiT_vs_BA_protection_camera']
    text=f'''<!doctype html><meta charset="utf-8"><title>原生MANO DiT v2 对照</title><style>body{{background:#14191e;color:#eee;font:16px system-ui;max-width:1500px;margin:32px auto;padding:0 24px}}a{{color:#7cd4ff;margin-right:24px}}td,th{{border:1px solid #59636b;padding:10px}}table{{border-collapse:collapse;margin:24px 0}}img{{width:100%}}pre{{white-space:pre-wrap}}.note{{background:#23323d;padding:20px;border-radius:12px}}figure{{margin:24px 0}}</style>
<h1>原生 MANO 参数 DiT：v2 修复与对照</h1>
<p class="note">当前这一开发片段，余弦日程原生DiT＋保护＋BA相机误差 {stages['coherent_cosine_dit']['metrics']['DiT_BA']['camera_mapped20_mm']['mean']:.2f} mm，基础BA为 {base['camera_mapped20_mm']['mean']:.2f} mm，出现局部正收益。仍有原准确点损伤、长尾和单序列限制，未替换生产默认。</p>
<p>150帧、300手帧、6000个映射20点。真实自然RGB，无人工遮挡。当前预测均封存后评分，但片段历史上已查看；单个序列不能给出序列级置信区间，不能当作新鲜泛化或私有5090九段结果。</p>
<h2>修复什么</h2><p>原BA只把左右手标记改成轨迹多数票，却继续使用旧裁图产生的姿态。20帧受影响，固定参数下平均几何变化192.71mm、最大294.74mm。本轮先只修复BA错误覆盖，再单独按预测置信度投票并实际重跑20帧WiLoR，重新计算表面和深度；280帧输入保持完全一致。</p>
<p>扩散原100步线性日程终点alpha=0.36356，与纯噪声采样不匹配。本轮仅改变为终点近零的余弦日程，训练数据、600步预算、初始化和批次计划完全匹配。结果如实保留；余弦日程并未稳定胜过回归或基础BA。</p>
<h2>整线逐项对照</h2><p>单位mm。历史原版与修复版分行；后四组都采用同一一致前端、同一深度、同一BA。前置位移保护为MANO参数插值，BA后不保证仍满足5/10/30mm。</p>
<table><tr><th>方法</th><th>相机均值</th><th>P95</th><th>最大</th><th>腕对齐均值</th></tr>{''.join(rows)}</table>
<h2>恢复、损伤与时序</h2><p>以一致前端＋基础BA为参照。基础BA：一阶误差P95 {baseline_temporal['error_velocity_mm_at_60Hz']['P95']:.2f} mm@60Hz，二阶P95 {baseline_temporal['error_acceleration_mm_at_60Hz2']['P95']:.2f} mm@60Hz²，腕跳变 {baseline_temporal['wrist_error_jumps_over10mm']}/{baseline_temporal['wrist_pairs']}。时序误差扣除真实运动，不屏蔽预测自身的难例。</p>
<table><tr><th>候选</th><th>&gt;20恢复至≤20</th><th>原≤10损伤&gt;1</th><th>原≤10变&gt;20</th><th>一阶P95</th><th>二阶P95</th><th>腕跳变</th><th>快速动作幅度比</th><th>骨长CV</th></tr>{''.join(temporal)}</table>
<p>主候选BA后相对基础BA位移RMSE/P95/max为3.41/6.63/13.55mm，但自投影2D重投影增加0.96px，未过0.2px线。新增BA后保护：每条完整轨迹共享一个α，在SO(3)和形状参数中插值并重新FK，按原5/10/30mm及0.2px线回退；未放宽阈值、未逐点裁剪、未按GT选择。该分支仍未通过多序列泛化验收。</p>
<p>前端修复本身相对原基础BA的正确点审计：{html.escape(json.dumps(repair_harm,ensure_ascii=False))}。降低均值不代表原正确点无损。</p>
<h2>600步匹配训练／采样对照（BA前）</h2><p>52个已复用开发窗口、1040点；原准确点83个（排除腕），困难点558个。shared仅让四个高斯采样在相邻中心帧共享，改变蒙特卡洛相关性，不是整段轨迹联合去噪。</p>
<table><tr><th>方法</th><th>腕对齐均值</th><th>P95</th><th>准确点损伤&gt;1</th><th>准确点变&gt;20</th><th>一阶P95</th></tr>{''.join(sched)}</table>
<h2>自然失败案例</h2><p>{links}</p>{''.join(cards)}
<h2>实现与验收边界</h2><p>17帧非causal、±1.6秒、每帧192×1280完整WiLoR空间特征；视觉主干冻结，适配层／空间注意力／时序模块／DiT训练。原生SO(3)姿态残差、MANO FK、直接参数BA，腕与形状不由DiT学习。训练参数监督为GT20拟合伪目标；推理无GT、无20点到MANO拟合。</p>
<p>两组新训练均验证了梯度和实际权重更新、FK缓冲不变、GT污染不改变推理、原生参数重解码、完整时间轴与JSON/H5。优化器和RNG做精确往返检查，不冒充真实中断恢复质量实验。当前BA仍使用WiLoR自投影2D，尚无独立RGB关键点观测；没有硬关节范围、硬速度／加速度或碰撞保证。RGBTrack五输出TRT补丁沿用v1，尚未构建新引擎或部署ROS。</p>
<details><summary>完整统计、哈希与边界</summary><pre>{html.escape(json.dumps(summary,ensure_ascii=False,indent=2))}</pre></details>'''
    (d/'report.html').write_text(text)
    if (R/'README.md').exists():shutil.copy2(R/'README.md',d/'README.md')
    manifest={str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest() for p in d.rglob('*') if p.is_file()}
    (d/'delivery_files.json').write_text(json.dumps(dict(complete=True,files=manifest),indent=2))
    archive=R/'delivery_native_mano_v2.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(d,arcname='delivery')
    receipt=dict(complete=True,archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(manifest))
    (R/'delivery_archive.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))

if __name__=='__main__':main()
