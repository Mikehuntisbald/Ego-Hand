import base64,hashlib,json,shutil,tarfile
from pathlib import Path
import wilor_eval_common
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from offline_kp_data import RUN,save

OUT=RUN/'delivery';OUT.mkdir(exist_ok=True)
r=json.loads((RUN/'test_results.json').read_text());obj=json.loads((OUT/'example_output.json').read_text())
frames=obj['tracks'][0]['frames'];images=[]
for f in frames:
    p=Path(f['image']);images.append('data:image/jpeg;base64,'+base64.b64encode(p.read_bytes()).decode() if p.exists() else '')
payload=json.dumps(dict(result=obj,images=images),ensure_ascii=False).replace('</','<\\/')
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>离线手指关键点复核</title>
<style>body{margin:24px auto;max-width:1000px;font:16px/1.6 system-ui;background:#f5f7fb;color:#17283b}h1{font-size:25px}button,input{font:inherit}button{padding:8px 13px;margin-right:8px;border:1px solid #c4ccd8;border-radius:7px;background:white;cursor:pointer}canvas{width:min(90vw,850px);height:auto;background:#ddd;touch-action:none;border-radius:8px}#note{padding:12px;background:#fff0cf;border-radius:8px}.bar{display:flex;align-items:center;gap:14px;margin:14px 0}#frame{width:360px}#status{min-height:26px}</style>
<h1>离线手指关键点补全 · 人工复核</h1>
<p id="note">蓝色：模型补全；绿色：输入坐标；橙色：人工修正。全部自动点默认未复核。可拖动点修改位置；图像仅供人工查看，没有输入本轮补全模型。</p>
<div class="bar"><label for="frame">帧</label><input id="frame" type="range" min="0" max="16" value="8"><span id="frameLabel"></span></div>
<div class="bar"><button id="previous">上一帧</button><button id="next">下一帧</button><button id="approve">确认本帧已有坐标</button><button id="export">导出复核 JSON</button></div>
<label><input id="uncertainty" type="checkbox"> 显示补全点的不确定性范围（经验估计）</label>
<p id="status" aria-live="polite"></p><canvas id="canvas" width="1408" height="1408" aria-label="手指关键点编辑画布"></canvas>
<script id="data" type="application/json">PAYLOAD</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent), result=data.result;
const frames=result.tracks[0].frames, canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
const slider=document.getElementById('frame'),status=document.getElementById('status');
const edges=[[5,6],[6,7],[7,0],[5,8],[8,9],[9,10],[10,1],[5,11],[11,12],[12,13],[13,2],[5,14],[14,15],[15,16],[16,3],[5,17],[17,18],[18,19],[19,4]];
const imgs=data.images.map(src=>{const im=new Image();im.onload=draw;im.src=src;return im});let selected=8,drag=-1;
function draw(){ctx.clearRect(0,0,1408,1408);const im=imgs[selected];if(im.complete&&im.naturalWidth)ctx.drawImage(im,0,0,1408,1408);
const f=frames[selected];document.getElementById('frameLabel').textContent=`${selected+1}/${frames.length} · ${f.timestamp_s.toFixed(3)} s`;
for(const [a,b] of edges){const p=f.points[a].xy_px,q=f.points[b].xy_px;if(!p||!q)continue;ctx.strokeStyle='#f8fafc';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke()}
f.points.forEach((p,j)=>{if(!p.xy_px)return;const [x,y]=p.xy_px;const color=p.source==='manual'?'#ffae32':p.source==='inferred'?'#36baf2':'#40dd88';
if(document.getElementById('uncertainty').checked&&p.empirical_radius90_px){ctx.fillStyle='#36baf21f';ctx.beginPath();ctx.arc(x,y,p.empirical_radius90_px,0,2*Math.PI);ctx.fill()}
ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,7,0,Math.PI*2);ctx.fill();ctx.font='20px sans-serif';ctx.fillText(String(j),x+10,y-7)});
status.textContent=`本帧 ${f.points.filter(p=>p.source==='inferred').length} 个补全点；${f.points.filter(p=>p.reviewed).length}/20 个点已人工复核。`}
function show(n){selected=Math.max(0,Math.min(frames.length-1,n));slider.value=selected;draw()}
slider.oninput=()=>show(Number(slider.value));document.getElementById('previous').onclick=()=>show(selected-1);document.getElementById('next').onclick=()=>show(selected+1);
document.getElementById('uncertainty').onchange=draw;
function position(e){const r=canvas.getBoundingClientRect();return [(e.clientX-r.left)*1408/r.width,(e.clientY-r.top)*1408/r.height]}
canvas.onpointerdown=e=>{const [x,y]=position(e);let best=25;drag=-1;frames[selected].points.forEach((p,j)=>{if(!p.xy_px)return;const d=Math.hypot(p.xy_px[0]-x,p.xy_px[1]-y);if(d<best){best=d;drag=j}});if(drag>=0)canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(drag<0)return;const p=frames[selected].points[drag];if(!p.original_xy_px)p.original_xy_px=p.xy_px.slice();p.xy_px=position(e).map(v=>Math.max(0,Math.min(1407,v)));p.source='manual';p.reviewed=true;p.review_required=false;p.empirical_radius90_px=null;draw()};
canvas.onpointerup=()=>{drag=-1};canvas.onpointercancel=()=>{drag=-1};
document.getElementById('approve').onclick=()=>{frames[selected].points.forEach(p=>{if(p.xy_px){p.reviewed=true;p.review_required=false}});draw()};
document.getElementById('export').onclick=()=>{const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'});const u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download='reviewed_keypoints.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
draw();
</script></html>'''
(OUT/'review.html').write_text(page.replace('PAYLOAD',payload),encoding='utf-8')
fig,ax=plt.subplots(figsize=(8,4.5));gaps=[1,3,6,9]
for method,color in [('linear','#8b95a5'),('regression','#d49b28'),('dit','#1679bd')]:
    values=[r['methods']['regression' if method=='regression' else 'dit']['reports'][f'single_finger/{g}']['metrics'][method]['mean_px'] for g in gaps]
    ax.plot(np.array(gaps)/6,values,'o-',label=method,color=color)
ax.set(xlabel='Removed observation span (nominal seconds)',ylabel='Hidden-joint error (pixels at 1408 x 1408)',title='Offline finger-keypoint completion: controlled missing observations')
ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(OUT/'gap_errors.png',dpi=160);plt.close(fig)
table=[]
for g in gaps:
    d=r['methods']['dit']['reports'][f'single_finger/{g}'];reg=r['methods']['regression']['reports'][f'single_finger/{g}']
    table.append(f"<tr><td>{g/6:.2f} s</td><td>{d['metrics']['linear']['mean_px']:.2f}</td><td>{d['metrics']['dit']['mean_px']:.2f}</td><td>{reg['metrics']['regression']['mean_px']:.2f}</td></tr>")
primary=r['methods']['dit']['reports']['single_finger/all'];unc=r['methods']['dit']['uncertainty']
whole=r['methods']['dit']['reports']['all_points/all']
report=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>离线关键点 Diffusion 首轮结果</title><style>body{{font:16px/1.7 system-ui;max-width:950px;margin:32px auto;color:#17283b}}td,th{{padding:10px;border-bottom:1px solid #ddd}}img{{max-width:100%}}.notice{{background:#fff0cc;padding:15px}}</style>
<h1>离线手指关键点补全：第一版</h1><p class="notice">这是受控关键点观测缺失测试，不是自然遮挡恢复已验证。Diffusion 优于线性插值，但同条件回归误差更低。当前输出用于辅助人工标注，不能直接作为免审核真值。</p>
<p>训练为六个受试者的 2526 个窗口，开发为 P0003 的 891 个窗口，测试为 P0010/P0015 的 1909 个窗口。测试 10 个源序列、单指缺失 28768 个关节点实例（跨缺口配置重复计数）。这些是既有实验数据上的任务留出，不是首次未看过的数据。</p>
<p>模型使用双向 17 帧上下文，当前及连续缺口内的指定关节坐标全部移除，不读取 RGB、GT 或缺口内隐藏坐标。观测来自 YOLO 手框下的 WiLoR 二维预测；YOLO 本身尚无已训练的手指关键点头。当前手框关联仍可用，本版不恢复整手轨迹丢失。</p>
<h2>单指缺失误差，像素</h2><table><tr><th>缺口</th><th>线性插值</th><th>DiT</th><th>时序回归</th></tr>{''.join(table)}</table>
<p>DiT 总体 {primary['metrics']['dit']['mean_px']:.2f} px，线性插值 {primary['metrics']['linear']['mean_px']:.2f} px，改善 {primary['improvement_over_linear_pct']:.2f}%；按源序列配对 bootstrap 的变化 CI 为 {primary['ci95_change_px']} px。原输入坐标最大变动为 0。</p>
<img src="gap_errors.png"><p>DiT − 回归的平均误差差值：{r['dit_vs_regression']['change_px']:.2f} px，CI {r['dit_vs_regression']['ci95_change_px']}。正值表示 DiT 更差。</p>
<h2>失败情况：整手所有关键点都缺失</h2><p class="notice">DiT {whole['metrics']['dit']['mean_px']:.2f} px，线性插值 {whole['metrics']['linear']['mean_px']:.2f} px，DiT 更差。该压力测试未通过，不能把部分手指补全的收益推广到整手完全无当前观测。原始结果全部保留。</p>
<h2>不确定性与复核</h2><p>开发集校准的名义 90% 半径在测试上的覆盖率为 {unc['test_coverage']*100:.1f}%，平均半径 {unc['mean_radius_px']:.1f} px。这是经验校准，不保证自然遮挡或新设备分布上的覆盖率。</p>
<p><a href="review.html">打开关键点复核示例</a>。页面提供拖动和 JSON 导出代码；脚本语法检查通过，但浏览器安全策略阻止了打开本地 HTML，交互尚未实测。示例按元数据选取第一段足够长的测试片段，没有按改善程度挑选。图像只供人工复核，不是模型输入；示例推理文件不含 GT。</p>
<h2>尚未解决</h2><p>自然遮挡的逐指有效性判定、遮挡像素真正缺失时的效果、完整轨迹持续漏检、时序抖动以及下游 3D 收益尚未验证。输入 observed 表示调用方接受的坐标，不等于已证明该关节可见。自动源关键点也可能有误，需复核；若有人工关键帧，应通过同一接口输入。</p></html>'''
(OUT/'report.html').write_text(report,encoding='utf-8')
shutil.copytree(RUN/'sealed',OUT/'sealed',dirs_exist_ok=True)
for name in ['test_results.json','protocol.json','implementation_checks.json','data_done.json']:shutil.copy2(RUN/name,OUT/name)
shutil.copy2(Path(__file__).parent/'infer_offline_kp.py',OUT/'infer_offline_kp.py')
readme=f'''# 离线手指关键点补全 v1

用途：双向视频上下文辅助标注。受控缺点测试已完成，真实逐指遮挡尚未验证。
当前回归对照优于 DiT，不应只因使用 diffusion 就选择它作为默认标注器。
整手所有关键点均缺失时 DiT 比线性插值差（75.72 vs 68.14 px），压力测试失败。当前不能自动生成免审核标注。

服务器推理：
```bash
PY=/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python
$PY {OUT}/infer_offline_kp.py --run {RUN} --input {OUT}/example_input.json --output {OUT}/my_result.json --kind dit --device cuda:0
```
`--kind regression` 使用同条件回归。可用 `--device cpu`，参考评估使用 GPU BF16。
独立运行只需要 PyTorch、NumPy 和 sealed/ 内的模型代码/权重；不需要 WiLoR 大模型或 GT。

输入格式见 example_input.json：图像尺寸 1408×1408；每个轨迹有递增的 timestamp_s、20 个 xy_px、20 个 observed。
缺失点用 null 且 observed=false；即使给出隐藏坐标，模型也会先将它移除。
observed=true 表示接受这个输入坐标，不代表系统自动判断了真实可见性。它可以来自检测模型，也可以来自人工关键帧。
无历史/未来上下文的点返回 available=false 与 null；不会把无依据的坐标冒充可靠标注。
输出保留输入点，补全点带候选、经验不确定性半径和 review_required。
review.html 提供切换帧、拖动改点、确认本帧、导出复核 JSON 的代码。脚本语法检查通过；浏览器策略禁止访问本地 file URL，交互尚未实测。全部自动点默认未复核。

训练数据与脚本保留在 /mnt/why/hot3d_hand_residual/offline_kp_*.py、train_offline_kp.py、evaluate_offline_kp.py；运行目录 {RUN}。
只训练了二维轨迹补全；不等于 YOLO 已增加关键点头，也没有把补全点自动接入 WiLoR 3D 网络。
目前是 6Hz 观测间隔上的验证；接口按约 1/6s 抽取前后上下文，但未验证原始30fps完整轨迹精度。
'''
(OUT/'README.md').write_text(readme,encoding='utf-8')
manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}
save(OUT/'manifest.json',manifest)
archive=RUN/'offline_keypoint_diffusion_v1.tar.gz'
with tarfile.open(archive,'w:gz') as tar:tar.add(OUT,arcname='offline_keypoint_diffusion_v1')
save(RUN/'delivery_archive.json',dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size))
print(json.dumps(dict(report=str(OUT/'report.html'),review=str(OUT/'review.html'),archive=str(archive))))
