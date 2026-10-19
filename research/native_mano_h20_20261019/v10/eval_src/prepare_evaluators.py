"""Adapt existing native metric definitions, preserving their original sources."""
import hashlib,json
from pathlib import Path
R=Path('/mnt/why/native_candidate_sequence_v10_20261019');V7=Path('/mnt/why/native_mano_pool_v7_20261019');V9=Path('/mnt/why/native_rgb_observation_v9_20261019')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def change(s,a,b):assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
def main():
    assert json.loads((R/'full_status.json').read_text())['complete'];sources={}
    p=V7/'eval_src/evaluate_v7.py';s=p.read_text();sources[str(p)]=sha(p)
    s=change(s,"R=Path('/mnt/why/native_mano_pool_v7_20261019')",f"R=Path('{R}')")
    s=change(s,"NEW=[a+s for a in ['small','full'] for s in ['', '_guard']]","NEW=['frame_RGB','temporal_noRGB','temporal_RGB','temporal_noRGB_scaled','temporal_RGB_scaled']")
    s=change(s,"OLD=['WiLoR','cosine_protected_displacement','rollout_dit_displacement','v5_native_dit','v5_native_dit_guard']","OLD=['WiLoR','v5_native_dit_guard','v7_mean_guard']")
    line='            path=Path(\'/mnt/why/native_mano_supervision_v5_20261019/predictions\')/f"clip-{clip[\'clip\']:06}"/f"{a[3:]}.npz" if a.startswith(\'v5_\') else view/(\'predictions_rollout\' if a.startswith(\'rollout_\') else \'predictions\')/f\'{a}.npz\''
    replacement='''            if a=='v7_mean_guard':path=Path('/mnt/why/native_mano_pool_v7_20261019/predictions')/f"clip-{clip['clip']:06}"/'full_guard.npz'
            elif a.startswith('v5_'):path=Path('/mnt/why/native_mano_supervision_v5_20261019/predictions')/f"clip-{clip['clip']:06}"/f"{a[3:]}.npz"
            else:path=view/'predictions'/f'{a}.npz' '''
    s=change(s,line,replacement)
    s=change(s,"[('full','small'),('full_guard','small_guard')]","[('temporal_RGB','v7_mean_guard'),('temporal_RGB','frame_RGB'),('temporal_RGB','temporal_noRGB'),('temporal_RGB_scaled','v7_mean_guard'),('temporal_RGB_scaled','frame_RGB'),('temporal_RGB_scaled','temporal_noRGB_scaled')]")
    s=change(s,"for a in NEW:arrays[a]=np.load(R/'predictions'/f\"clip-{clip['clip']:06}\"/f'{a}.npz')['joints_camera21']","for a in NEW:arrays[a]=np.load(R/('likelihood_r1/predictions' if a.endswith('_scaled') else 'predictions')/f\"clip-{clip['clip']:06}\"/f'{a}.npz')['joints_camera21']")
    s=change(s,"cards=[];candidate='full_guard'","cards=[];candidate='temporal_RGB_scaled'")
    s=change(s,"('matched small data','small_guard')","('frame RGB selector','frame_RGB')")
    s=change(s,"('expanded native data',candidate)","('RGB + temporal selector',candidate)")
    s=change(s,"    allmask=np.ones(len(records),bool);", "    from capacity_diagnostic import assess_candidate_capacity\n    assess_candidate_capacity(records,GT,R)\n    allmask=np.ones(len(records),bool);")
    s=change(s,"native_relative_preservation=protection(relative['WiLoR'][mask],relative[a][mask],valid[mask]),", "native_relative_preservation=protection(relative['WiLoR'][mask],relative[a][mask],valid[mask]),preservation_vs_mean=protection(relative['v7_mean_guard'][mask],relative[a][mask],valid[mask]),")
    start=s.index("    page='''");end=s.index("    (out/'report.html').write_text(page);",start)
    page='''    page='<!doctype html><meta charset="utf-8"><title>原生MANO候选整段选择v10</title><style>body{font:17px system-ui;max-width:1400px;margin:32px auto;background:#161c22;color:#eee;padding:20px}td,th{border:1px solid #78828c;padding:10px}table{border-collapse:collapse}img{width:100%}a{color:#7fd4ff}figcaption{margin:12px}</style><h1>原生MANO候选整段选择v10</h1><p>冻结v7多数据DiT和v9独立RGB头，不新增训练或GT调参。每帧同4采样，候选包括原均值、WiLoR、4单样本；全部保持原5/10/30mm参数保护。比较逐帧RGB、仅时序、RGB+二阶时序。以真实时间计算世界朝向腕相对速度/加速度，按身份/相机段/手侧/长缺口切段。非因果整段Viterbi，不称整段联合扩散。</p><p>六历史录制894右手帧，主干冻结、原生右手监督、原标注资产版本未独立核验。这里是提供参考相机的BA前补全模块；完整预测相机/物理深度/BA回放另列。所有输出先封存再评分，不使用GT选候选。P10/P15只表示未进优化器与权重选择，不是全项目新鲜测试。</p><table><tr><th>方案</th><th>腕对齐20点mm</th><th>相机21点mm</th><th>原准确点损伤&gt;1mm</th><th>原≤10→&gt;20mm</th><th>困难点恢复</th></tr>'+table+'</table><p><a href="metrics.json">逐片段、配对区间、真实时间运动/骨长/快速动作</a></p>'+''.join(cards)
'''
    s=s[:start]+page+s[end:];module_source=s
    p=V9/'eval_src/evaluate_full.py';s=p.read_text();sources[str(p)]=sha(p)
    s=change(s,"R=Path('/mnt/why/native_rgb_observation_v9_20261019')",f"R=Path('{R}')")
    s=change(s,"ARMS=['BA_spatial','BA_pooled','DiT_spatial','DiT_pooled']","ARMS=['frame_RGB','temporal_noRGB','temporal_RGB','temporal_noRGB_scaled','temporal_RGB_scaled']")
    needle="v7_DiT_guard=V7/'full_pipeline/outputs/ba_full_guarded/parameters.npz')"
    s=change(s,needle,"v7_DiT_guard=V7/'full_pipeline/outputs/ba_full_guarded/parameters.npz',v9_RGB_DiT_guard=Path('/mnt/why/native_rgb_observation_v9_20261019/full_pipeline/outputs/ba_DiT_spatial_guarded/parameters.npz'))")
    s=change(s,"candidate='DiT_spatial_guarded'","candidate='temporal_RGB_scaled_guarded'")
    s=change(s,"relative['v7_DiT_guard'].mean(1)","relative['v9_RGB_DiT_guard'].mean(1)")
    s=change(s,"('native DiT guard','v7_DiT_guard')","('native DiT + RGB guard','v9_RGB_DiT_guard')")
    s=change(s,"('DiT + independent RGB',candidate)","('whole track selection',candidate)")
    s=s.replace('原生MANO DiT + 独立RGB二维观测：完整回放','原生MANO候选整段选择：完整回放').replace('新四组原始/保护共八条输出','新五组原始/保护共十条输出').replace('all_eight_prediction_seal_sha256','all_ten_prediction_seal_sha256').replace('独立RGB定位完整原生BA v9','原生MANO候选整段BA v10')
    needle="wrist_Z_signed_mean_mm=float((p[:,0,2]-GT[:,0,2]).mean()*1000))"
    s=change(s,needle,"wrist_Z_signed_mean_mm=float((p[:,0,2]-GT[:,0,2]).mean()*1000),relative_preservation_vs_v9=protection(relative['v9_RGB_DiT_guard'],relative[a],native_valid),absolute_preservation_vs_v9=protection(error['v9_RGB_DiT_guard'],error[a],absolute_valid))")
    outputs={'evaluate_module.py':module_source,'evaluate_full.py':s}
    for name,text in outputs.items():compile(text,name,'exec');assert not (R/'eval_src'/name).exists()
    for name,text in outputs.items():(R/'eval_src'/name).write_text(text)
    (R/'receipts/evaluator_sources.json').write_text(json.dumps(dict(reference_sha256=sources,generated_sha256={p.name:sha(p) for p in (R/'eval_src').glob('evaluate_*.py')},same_native_metrics=True,new_v9_reference_protection=True),indent=2))
if __name__=='__main__':main()
