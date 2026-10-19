"""Archive the rejected shape-compatible pilot and its complete evidence."""
import hashlib,json,shutil,tarfile
from pathlib import Path
import torch
R=Path('/mnt/why/native_mano_shape_v6_20261019')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def main():
    assert json.loads((R/'finish_status.json').read_text())['complete']
    full=json.loads((R/'receipts/full_native_right_metrics.json').read_text());m=json.loads((R/'evaluation/metrics.json').read_text())
    out=R/'delivery';out.mkdir(exist_ok=False)
    for name in ['protocol.json','inference_lock.json','predictions_frozen.json','controller_status.json','finish_status.json','README.md','ITERATION_STATUS.json']:copy(R/name,out/name)
    for name in ['src','eval_src','receipts','evaluation','predictions']:
        shutil.copytree(R/name,out/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for arm in ['control_native','compatible','compatible_protected']:
        p=R/'runs'/arm;ck=torch.load(p/'best_development.pt',weights_only=False,map_location='cpu');(out/'models').mkdir(exist_ok=True)
        torch.save(dict(model=ck['model'],config=ck['config'],arm=ck['arm'],supervision=ck['supervision'],selected_step=ck['step'],source_checkpoint_sha256=sha(p/'best_development.pt'),right_only=True,inference_GT=False,production_admitted=False),out/'models'/f'{arm}.pt')
        for name in ['done.json','selection.json','progress.json','console.log']:copy(p/name,out/'training'/arm/name)
        (out/'training'/arm/'selected_checkpoint.json').write_text(json.dumps(dict(step=ck['step'],selection=ck['selection'],source_sha256=sha(p/'best_development.pt')),indent=2))
    for key,p in {'v5':Path('/mnt/why/native_mano_supervision_v5_20261019/src'),'v2':Path('/mnt/why/native_mano_dit_v2_20261019/src'),'v4':Path('/mnt/why/native_mano_rollout_v4_20261019/src'),'v3':Path('/mnt/why/native_mano_multiseq_v3_20261019/src')}.items():
        for f in p.glob('*.py'):copy(f,out/'reference_source'/key/f.name)
    for name in ['full_rollout_dit','full_rollout_dit_guarded']:shutil.copytree(R/'comparisons'/name/'outputs/review_clip2102',out/'reviews'/name)
    for name in ['ba_rollout_dit','ba_rollout_dit_guarded']:
        for f in (R/'full_pipeline/outputs'/name).iterdir():
            if f.is_file():copy(f,out/'full_outputs'/name/f.name)
    names={'WiLoR':'WiLoR','control_native':'原生监督精确控制','control_native_guard':'原生控制＋推理保护','compatible':'同形状姿态监督','compatible_guard':'同形状姿态＋推理保护','compatible_protected':'同形状姿态＋训练保护','compatible_protected_guard':'同形状姿态＋训练/推理保护'}
    rows=''
    for a in names:
        v=m['metrics'][a];p=v['native_relative_preservation'];rows+=f"<tr><td>{names[a]}</td><td>{v['native_relative20_mm']['mean']:.2f}</td><td>{v['camera21_mm']['mean']:.2f}</td><td>{p['good_harmed_over1mm']}/{p['base_good10']}</td><td>{p['good_to_over20']}</td><td>{p['recovered_to20']}/{p['base_hard20']}</td></tr>"
    fullnames={'BA':'基础BA','old_v2_guard':'旧v2保护','old_v5_guard':'原生v5普通版＋保护','compatible':'v6同形状姿态＋BA','compatible_guard':'v6同形状姿态＋BA后保护'};fr=''
    for a,n in fullnames.items():
        v=full['metrics'][a];p=v['preservation'];fr+=f"<tr><td>{n}</td><td>{v['camera21_mm']['mean']:.2f}</td><td>{v['native_relative20_mm']['mean']:.2f}</td><td>{p['good_harmed_over1mm']}/{p['base_good10']}</td><td>{p['recovered_to20']}/{p['base_hard20']}</td></tr>"
    page='''<!doctype html><meta charset="utf-8"><title>姿态形状解耦v6：未晋级</title><style>body{font:17px system-ui;max-width:1300px;margin:32px auto;background:#141b22;color:#eee;padding:24px}td,th{border:1px solid #6d7d88;padding:12px}table{border-collapse:collapse;margin:24px 0}a{color:#83d6ff}b{color:#ffcf75}pre{white-space:pre-wrap}</style><h1>姿态与形状解耦 v6：本轮未晋级</h1>
    <p><b>六段真实GT评分更差，保留原生v5候选，不替换默认。</b>移除形状误差对姿态损失的干扰，有明确的几何依据，但这轮小数据训练没有产生实际综合收益。完整BA后相机均值略降，相对手姿与准确点损伤反而更差。</p>
    <p>训练审计：155个右手窗口，正确姿态仅因预测形状不同仍有6.97mm相对误差；原FK在真实姿态处平均梯度7.84e-5，同预测形状的姿态监督为9.17e-11。这是训练损失诊断，不是验证收益。最终评估和准确点保护始终使用真实GT形状。</p>
    <p>三组相同样本/初始化/600更新/损失权重。控制600步模型、optimizer和所有RNG与v5逐位一致，六段部署数组也逐值一致。视觉主干冻结，自然RGB完整16×12空间特征、17非因果时隙、原生MANO旋转和实际4采样/10步均值。腕和形状固定，原生监督仅右手；不声称整段联合扩散或全产线端到端。</p>
    <h2>六录制894匹配右手帧：BA前</h2><table><tr><th>方案</th><th>腕对齐20点mm</th><th>相机21点mm</th><th>原准确点损伤&gt;1mm</th><th>原≤10→&gt;20mm</th><th>困难点恢复</th></tr>'''+rows+'''</table><p>同形状姿态保护版相对原控制恶化0.433mm，六录制95%CI[0.130,0.748]mm。三人员/六录制且此前已查看，属于开发诊断，不冒充新鲜泛化。</p>
    <p><a href="evaluation/report.html">18组自然RGB案例</a> · <a href="evaluation/metrics.json">完整误差、损伤/恢复、配对CI、真实时间运动与快速动作</a></p>
    <h2>预先锁定的完整回放：150匹配右手帧原生GT</h2><p>同RF/WiLoR、预测cuVSLAM、实际物理深度与BA，只换普通同形状姿态模型。保持原5/10/30mm和+0.2px保护线。单段反复使用，仅作诊断，不支持总体泛化声明。</p><table><tr><th>方案</th><th>相机21点mm</th><th>腕对齐20点mm</th><th>原准确点损伤&gt;1mm</th><th>困难点恢复</th></tr>'''+fr+'''</table><p><a href="receipts/full_native_right_metrics.json">原生完整回放统计</a> · <a href="reviews/full_rollout_dit/report.html">双手旧UmeTrack辅助：原始BA提案</a> · <a href="reviews/full_rollout_dit_guarded/report.html">双手旧口径：BA后保护</a></p>
    <p>旧双手UmeTrack20相机口径v6为15.80mm，v5为15.78mm。原生右手相机均值的小幅下降不抵消相对手姿及损伤恶化。当前BA仍用WiLoR自投影2D，未增加独立图像关键点，不保证关节限位/碰撞或整段硬运动可行性。</p>
    <p>下一步已经找到同六训练人员62段/2526窗口，独立v7正在准备自然像素、原生初值及空间特征缓存。保留原生监督，先比较同预算少数据/多数据；不继续放大本轮更差的损失方案。缓存准备不代表大规模训练已经完成。</p>
    <p><a href="receipts/v5_control_exact.json">600步精确控制</a> · <a href="receipts/GT_free_sampler_and_gradient.json">GT隔离/梯度</a> · <a href="receipts/actual_resume_verification.json">真实中断恢复</a> · <a href="README.md">方法与范围</a></p>
    <p>原生GT用现有RIGHT资产、官方SMPLX和WiLoR21点约定解码；未独立核验原标注资产版本，LEFT资产仍未齐备。权重均为研究候选，不称安全best。</p>'''
    (out/'report.html').write_text(page)
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()};(out/'files_sha256.json').write_text(json.dumps(files,indent=2))
    archive=R/'delivery_native_mano_v6.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(out,arcname='delivery')
    receipt=dict(complete=True,bytes=archive.stat().st_size,sha256=sha(archive),files=len(files),production_admitted=False,outcome='rejected; retain v5 native candidate')
    (R/'delivery_archive.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()
