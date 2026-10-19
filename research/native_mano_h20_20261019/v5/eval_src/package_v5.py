"""Immutable local-review package, excluding third-party MANO assets."""
import hashlib,html,json,shutil,tarfile
from pathlib import Path
import torch
R=Path('/mnt/why/native_mano_supervision_v5_20261019')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def main():
    assert json.loads((R/'finish_status.json').read_text())['complete']
    full=json.loads((R/'receipts/full_native_right_metrics.json').read_text());assert full['complete']
    out=R/'delivery';out.mkdir(exist_ok=False)
    for name in ['protocol.json','inference_lock.json','predictions_frozen.json','controller_status.json','finish_status.json','ITERATION_STATUS.json','README.md']:
        copy(R/name,out/name)
    for name in ['src','eval_src','receipts','evaluation','predictions']:
        shutil.copytree(R/name,out/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['ready.json'] :copy(R/'data'/name,out/'training_data'/name)
    for a in ['native_regression','proxy_dit','native_dit','native_protected']:
        folder=R/'runs'/a;ck=torch.load(folder/'best_development.pt',weights_only=False,map_location='cpu')
        (out/'models').mkdir(exist_ok=True);torch.save(dict(model=ck['model'],config=ck['config'],arm=ck['arm'],supervision=ck['supervision'],selected_step=ck['step'],
            inference_GT=False,source_checkpoint_sha256=sha(folder/'best_development.pt'),right_only=True,production_admitted=False),out/'models'/f'{a}.pt')
        for name in ['done.json','selection.json','progress.json','console.log']:copy(folder/name,out/'training'/a/name)
        (out/'training'/a/'selected_checkpoint.json').write_text(json.dumps(dict(step=ck['step'],selection=ck['selection'],best_score=ck['best_score'],source_sha256=sha(folder/'best_development.pt'),batch_plan_sha=ck['batch_plan_sha']),indent=2))
    # Compatible core source, not the original environments or third-party assets.
    dependencies={'base_v2':Path('/mnt/why/native_mano_dit_v2_20261019/src'),'base_v4':Path('/mnt/why/native_mano_rollout_v4_20261019/src'),'validation_v3':Path('/mnt/why/native_mano_multiseq_v3_20261019/src')}
    for name,p in dependencies.items():
        for file in p.glob('*.py'):copy(file,out/'reference_source'/name/file.name)
    for name in ['full_rollout_protected','full_rollout_guarded','full_rollout_dit','full_rollout_dit_guarded']:
        folder=R/'comparisons'/name;shutil.copytree(folder/'outputs/review_clip2102',out/'reviews'/name)
        for f in (folder/'receipts').glob('*'):
            if f.is_file():copy(f,out/'full_chain_receipts'/name/f.name)
    for name in ['ba_rollout_protected','ba_rollout_guarded','ba_rollout_dit','ba_rollout_dit_guarded']:
        for f in (R/'full_pipeline/outputs'/name).iterdir():
            if f.is_file():copy(f,out/'full_chain_outputs'/name/f.name)
    m=json.loads((R/'evaluation/metrics.json').read_text());rows=''
    for name,value in m['metrics'].items():
        p=value['native_relative_preservation'];rows+=f"<tr><td>{name}</td><td>{value['native_relative20_mm']['mean']:.2f}</td><td>{value['camera21_mm']['mean']:.2f}</td><td>{p['good_harmed_over1mm']}/{p['base_good10']}</td><td>{p['good_to_over20']}</td></tr>"
    fm=''
    names={'BA':'基础BA','old_v2_guard':'旧v2 DiT+BA+保护','native_v5':'原生保护损失DiT+BA','native_v5_guard':'原生保护损失DiT+BA后保护','native_v5_dit':'普通原生DiT+BA（事后探索）','native_v5_dit_guard':'普通原生DiT+BA后保护（事后探索）'}
    for name,v in full['metrics'].items():fm+=f"<tr><td>{names[name]}</td><td>{v['camera21_mm']['mean']:.2f}</td><td>{v['native_relative20_mm']['mean']:.2f}</td><td>{v['wrist_mm']['mean']:.2f}</td></tr>"
    page='''<!doctype html><meta charset="utf-8"><title>原生MANO监督v5复核</title><style>body{font:17px system-ui;max-width:1280px;margin:32px auto;background:#151b22;color:#eee;padding:24px}td,th{border:1px solid #6a7780;padding:12px}table{border-collapse:collapse;margin:24px 0}a{color:#7fd4ff}pre{white-space:pre-wrap}</style><h1>原生MANO参数监督 v5：训练与完整链路复核</h1>
    <p>已实现并训练直接MANO旋转参数DiT；本轮把监督从UmeTrack拟合伪目标改为数据集MANO参数。保留相同右手样本的旧目标对照。腕平移/形状固定、视觉主干冻结、自然RGB 16×12空间特征、17时隙非因果上下文。中心参数实际4采样/10步均值，尚无整轨迹联合扩散或Viterbi。</p>
    <p><b>范围：右手小规模开发试验，未准入生产。</b>155训练/25开发选择窗口，四组各600更新。缺可核验LEFT资产，网络左手零残差，BA仍按原有求解。GT使用现有RIGHT资产、官方SMPLX与WiLoR21点语义；未独立核验原标注资产版本。六验证片段已在此前查看，不能称新鲜盲测。</p>
    <p>修监督的原因：MANO与UmeTrack同名腕点平均差17.59mm，分别腕对齐后同19点GT自身差18.41mm。正确姿态下预测形状仍可带来约6.61mm相对误差。本轮只检验姿态监督，不把形状误差隐藏到新姿态里。</p>
    <h2>六录制、894匹配右手帧：BA前原生21点评估</h2><table><tr><th>方案</th><th>20点腕对齐mm</th><th>相机21点mm</th><th>原准确点损伤>1mm</th><th>原≤10→>20mm</th></tr>'''+rows+'''</table><p><a href="evaluation/report.html">预定保护版18组自然RGB案例与完整模块统计</a> · <a href="evaluation/metrics.json">难点恢复、配对区间、真实时间运动、骨长与快速动作</a></p>
    <h2>完整公开链路诊断回放：右手原生MANO GT</h2><p>保持相同RF/WiLoR、预测cuVSLAM、实际LightESS/UniDepth和BA，先复跑预先指定的原生保护模型；另因六段开发验证结果加入普通原生DiT，此项明确为事后探索。此处仍是反复查看的clip2102，不构成新泛化证据；与下面双手UmeTrack20报告口径分别列出。</p><table><tr><th>方案</th><th>相机21点mm</th><th>20点腕对齐mm</th><th>腕误差mm</th></tr>'''+fm+'''</table><p><a href="receipts/full_native_right_metrics.json">完整右手原生审计</a> · <a href="reviews/full_rollout_protected/report.html">双手UmeTrack20辅助：原生DiT+BA</a> · <a href="reviews/full_rollout_guarded/report.html">双手辅助：BA后整轨迹参数保护</a></p>
    <p><a href="reviews/full_rollout_dit/report.html">事后探索：普通原生DiT+BA</a> · <a href="reviews/full_rollout_dit_guarded/report.html">事后探索：普通原生DiT+BA后保护</a></p><p>guard保持原5/10/30mm与+0.2px线；没有放宽、GT挑候选或逐点裁剪。保护线衡量版本改动，不能保证真实正确。完整BA仍使用WiLoR自投影2D，未补上独立RGB关键点；不保证关节范围、碰撞或全轨迹硬运动可行性。</p>
    <p><a href="receipts/GT_free_sampler_and_gradient.json">GT隔离与实际采样梯度</a> · <a href="receipts/actual_resume_verification.json">真实3+3恢复一致性</a> · <a href="README.md">方法、边界及复跑入口</a></p>
    <p>不自动替换默认、不开定时任务、不发布。本交付不包含授权MANO模型、第三方环境或可跨卡复用的TRT引擎。</p>'''
    (out/'report.html').write_text(page)
    manifest={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()};(out/'files_sha256.json').write_text(json.dumps(manifest,indent=2))
    archive=R/'delivery_native_mano_v5.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(out,arcname='delivery')
    receipt=dict(complete=True,archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(manifest),production_admitted=False)
    (R/'delivery_archive.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()
