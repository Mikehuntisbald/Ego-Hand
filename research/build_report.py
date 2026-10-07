"""Offline HTML summary with real download/audit state and GT gallery."""
import html,json
from pathlib import Path
root=Path('/mnt/why/HOT3D');a=json.loads((root/'audit_report.json').read_text());m=json.loads((root/'subset_manifest.json').read_text())
gallery=json.loads((root/'previews/index.json').read_text())
subjects=''.join(f'<tr data-split="{r["split"]}"><td>{s}</td><td>{r["split"]}</td><td>{r["sequences"]}</td><td>{r["clips"]}</td><td>{r["frames"]:,}</td></tr>' for s,r in a['subjects'].items())
sequences=''.join(f'<tr data-split="{r["split"]}"><td>{r["sequence"]}</td><td>{r["split"]}</td><td>{r["clips"]}/{r["expected_clips"]}</td><td>{r["frames"]:,}</td></tr>' for r in a['sequences'])
photos=''.join(f'<figure><a href="previews/{g["file"]}"><img src="previews/{g["file"]}" alt="验证受试者 GT 叠加"></a><figcaption>{g["subject"]} · clip {g["clip"]} · frame {g["frame"]}<br>{g["chosen_hand"]} 手建模可见比例 {g["modeled_hand_visible_fraction"]:.1%}</figcaption></figure>' for g in gallery)
stage='下载与导出已完成' if a['stage']=='complete' else '下载与导出进行中：此页面是状态快照'
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HOT3D · 3D 手姿态数据准备</title>
<style>body{{margin:0;background:#f4f6f8;color:#1c2d38;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:35px 25px}}h1{{font-size:30px;margin-bottom:6px}}h2{{font-size:21px;margin-top:35px}}.sub{{color:#5d6e78}}.cards{{display:flex;gap:15px;flex-wrap:wrap;margin:24px 0}}.card{{background:white;border:1px solid #dce4e8;border-radius:12px;padding:18px 24px;flex:1;min-width:160px}}.big{{display:block;font-size:27px;font-weight:650}}code{{background:#e5ecef;padding:3px 6px;border-radius:4px}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{border-bottom:1px solid #e1e7ec;text-align:left;padding:10px 14px}}.gallery{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}figure{{margin:0}}img{{width:100%;border-radius:10px}}figcaption{{font-size:13px;color:#5d6e78}}.flow{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}.node{{background:white;border:1px solid #dce4e8;padding:16px;border-radius:10px}}.note{{background:#eaf0f3;padding:14px 18px;border-radius:10px}}a{{color:#11646b}}select{{font:inherit;padding:5px;margin:10px 0}}@media(max-width:650px){{.gallery{{grid-template-columns:1fr}}}}</style>
<main><h1>HOT3D · 3D 手姿态数据准备</h1><div class="sub">{stage} · RGB / calibration / hand GT</div>
<div class="cards"><div class="card"><span class="big">25</span>训练源序列 · 6 个受试者</div><div class="card"><span class="big">3</span>验证受试者 · 6 个源序列</div><div class="card"><span class="big">{a['train_frames']+a['val_frames']:,}</span>已核验 RGB 帧 / 69,000</div><div class="card"><span class="big">{a['retained_archive_bytes']/1e9:.2f} GB</span>保留的原始 RGB / GT 包</div></div>
<p>服务器数据目录：<code>/mnt/why/HOT3D</code>；代码：<code>/mnt/why/hot3d_hand_residual</code>。</p>
<div class="note">本次是官方 HOT3D-Clips：每个片段 150 帧、约 5 秒，按原始 source sequence 分组。它们不是完整连续 VRS 长录像。训练与验证的受试者完全分开，验证从官方 train 留出，未使用无手部 GT 的官方 test。</div>
<h2>保留的数据与标签</h2><p>仅保留 Aria RGB 流 <code>214-1</code>、逐帧相机标定、手部姿态、手形 profile 和时间戳。跳过灰度图及物体标注主体。ModelScope 仓库只有介绍和示例；实际使用 HF 镜像入口读取官方固定版本数据，二进制重定向至官方 CDN。</p>
<p>标签包含官方 20 点的世界坐标、RGB 相机坐标和腕点相对坐标，单位米；2D 投影使用官方 FISHEYE624 模型。整手建模可见比例与逐点投影有效性分别保存，未伪造逐点遮挡标签。</p>
<h2>验证受试者的 GT 叠加</h2><div class="gallery">{photos}</div><p class="sub">绿色为左手，橙色为右手。图中是 GT 投影，不是模型预测。可见比例来自几何建模，不能视为逐点遮挡真值。</p>
<h2>直接做 3D 残差的链路</h2><div class="flow"><div class="node">YOLO26s<br>手部检测 / 预测裁剪</div>→<div class="node">3D 粗姿态模型<br>待选择、训练</div>→<div class="node">条件 DiT<br>RGB 特征 + 初始 3D 姿态</div>→<div class="node">门控 3D 残差<br>腕部平移 + 相对手指坐标</div></div>
<p>已下载 YOLO26s 检测与 2D 姿态官方权重并校验 SHA256。YOLO26 不直接给出 3D 手姿态。残差 DiT 是未训练原型，只做了前向、反向和采样检查，尚无精度收益结论。</p>
<p>GT 裁剪、GT 位姿及受试者 GT 手形只作监督，不能作推理输入。门控的目标是“修正后 3D 误差确实下降”，不把检测分数当作姿态准确度。</p>
<h2>校验</h2><p>完成片段 {a['completed_clips']}/{a['expected_clips']}；subject split 无交集；相机坐标往返最大误差 {a['camera_roundtrip_max_error_m']:.2e} 米；有效投影点 {a['valid_projected_keypoints']:,}/{a['keypoints']:,}。最终复核输出包 SHA256：{a['archive_hashes_verified']} 个；标注 SHA256：{a['annotation_hashes_verified']} 个。选择性下载未验证完整源 TAR 的 SHA256，源版本和 LFS hash 已记录。</p>
<h2>受试者与源序列</h2><label>筛选：<select id="split"><option value="all">全部</option><option value="train">训练</option><option value="val">验证</option></select></label>
<table><thead><tr><th>Subject</th><th>Split</th><th>源序列</th><th>已完成片段</th><th>帧数</th></tr></thead><tbody>{subjects}</tbody></table>
<details style="margin-top:18px"><summary>查看全部源序列</summary><table><thead><tr><th>Sequence</th><th>Split</th><th>片段</th><th>帧数</th></tr></thead><tbody>{sequences}</tbody></table></details>
<h2>文件与运行入口</h2><p><a href="subset_manifest.json">固定子集清单</a> · <a href="audit_report.json">数据核验报告</a> · <a href="experiment_plan.json">3D 实验协议</a> · <a href="README.txt">目录与命令说明</a></p>
<p class="sub">来源：<a href="https://huggingface.co/datasets/bop-benchmark/hot3d">官方 Clips</a> · <a href="https://github.com/facebookresearch/hand_tracking_toolkit">手部工具包</a> · <a href="https://docs.ultralytics.com/models/yolo26/">YOLO26</a> · <a href="https://github.com/facebookresearch/DiT">DiT</a></p></main>
<script>document.getElementById('split').onchange=function(){{document.querySelectorAll('tr[data-split]').forEach(r=>r.hidden=this.value!=='all'&&r.dataset.split!==this.value)}};</script></html>'''
(root/'report.html').write_text(page)
print(dict(report=str(root/'report.html'),stage=a['stage']))
