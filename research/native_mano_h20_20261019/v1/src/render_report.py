"""Index for native training, ABI evidence and two real full-chain comparisons."""
import html,json
from pathlib import Path
ROOT=Path('/mnt/why/native_mano_dit_20261019')
pilot=json.loads((ROOT/'evaluation/metrics.json').read_text())
rows=[]
for name,data in pilot['metrics'].items():rows.append(f'<tr><td>{name}</td><td>{data["camera"]["mean_mm"]:.2f}</td><td>{data["relative"]["mean_mm"]:.2f}</td></tr>')
comparison=[]
for arm in ['raw','protected']:
    d=json.loads((ROOT/f'comparisons/{arm}/summary.json').read_text());m=d['metrics']['DiT_BA']['camera_mapped20_mm'];h=d['harm']
    comparison.append(f'<tr><td>{arm}</td><td>{m["mean"]:.2f}</td><td>{m["P95"]:.2f}</td><td>{m["max"]:.2f}</td><td>{h["good_to_over20mm"]}</td></tr>')
doc=f'''<!doctype html><meta charset="utf-8"><title>原生MANO参数DiT v1</title><style>body{{max-width:1200px;margin:36px auto;font:17px system-ui;background:#14191e;color:#eee;line-height:1.6}}a{{color:#7ccaff}}table{{border-collapse:collapse}}td,th{{border:1px solid #666;padding:10px 18px}}pre{{white-space:pre-wrap;background:#20282f;padding:20px}}strong{{color:#ffcb7b}}</style>
<h1>原生 MANO 参数 DiT v1</h1><p><strong>实现、600步×3组训练、原生参数直接接BA均已完成；仍未通过安全默认准入。</strong></p>
<p>RGB空间网格＋17帧时序 → MANO旋转Lie残差 → MANO FK → 固定相机整段BA。推理无20点→MANO拟合。第一版固定腕平移和形状，根旋转与15个手指旋转直接预测。</p>
<p>原生参数重解码、左右手、FP32几何及梯度核对通过；输出参数能重建交付骨架。训练伪参数由训练GT20拟合，仅作loss目标，不是MANO人工角度真值。</p>
<h2>相同预算的小规模对照：52个开发回放窗口</h2><table><tr><th>版本</th><th>相机误差mm</th><th>腕对齐误差mm</th></tr>{''.join(rows)}</table>
<p>六个训练人员、304窗口，各600步；52个dev_select选checkpoint；上述另52个dev_calibrate窗口在封存后评分。历史上已查看，不能称新鲜泛化。原生回归本轮优于原生DiT。</p>
<h2>同一150帧整线回放</h2><p>基础BA：21.22mm；旧桥接DiT＋BA：19.19mm。它们与下面的新原生版为相同前端输入，但旧模型的训练预算更大。</p>
<table><tr><th>原生DiT＋BA</th><th>均值mm</th><th>P95mm</th><th>最大mm</th><th>原≤10变&gt;20点数</th></tr>{''.join(comparison)}</table>
<p>位移保护在MANO参数中插值，不裁剪单个XYZ；BA之后仍可能放大长尾。不能用均值改善宣称全面通过。</p>
<p><a href="comparisons/raw/outputs/review_clip2102/report.html">原生原始版：全统计、自然RGB和手指放大图</a></p>
<p><a href="comparisons/protected/outputs/review_clip2102/report.html">原生位移保护版：全统计、自然RGB和手指放大图</a></p>
<p><a href="README.md">结构、监督、问题处理、复跑命令及RGBTrack接口说明</a></p>
<p>RGBTrack参考固定5ecaacf。原full-TRT只有cam/21点；本轮提供5输出原生MANO导出补丁和PyTorch实际RGB验证，尚未构建新TRT引擎或部署ROS。该提交没有找到手部整段MANO BA，不能把物体FoundationPose称为手BA。</p>
<details><summary>原生训练/推理完整统计</summary><pre>{html.escape(json.dumps(pilot,ensure_ascii=False,indent=2))}</pre></details>'''
(ROOT/'report.html').write_text(doc);print('Native report index complete')
