import json,html
from pathlib import Path
import numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from offline_rgb_encoder import crop_image
O=Path('/mnt/why/HOT3D/experiments/natural_reliability_v4/temporal_clue_audit')
data={
2:('A','窗口内有线索',[105,125,140],'边界附近暂时出视野；前后可见同一只手的指节与手指轮廓，不能归因于整段无信息。','较高'),
6:('A','强线索在窗口之外',[5,15,120],'片段开头杯侧可见更多手指轮廓；目标附近杯身遮住握持手指。较强证据比目标早约3.5–3.8秒，不在当前上下文中；远处姿态不能直接复制。','中'),
8:('A','窗口内有部分线索',[30,35,65],'进入暗处之前，手背、部分指节和抓握轮廓更清楚，f30/35实际有输入。瓶后指尖仍可能一直不可见，所以这里只能判定部分线索未充分利用。','中；混合遮挡'),
10:('A','窗口内有线索',[70,75,110],'目标受到另一只手/前臂遮挡，较早f70/75同一目标手更清楚且在实际输入中。不能把临时遮挡视为整段不可见。','较高'),
12:('A','窗口内有部分线索',[110,125,130],'目标手在玩具屋后，后续旋转露出更多掌侧与手指边缘；f125/130实际有输入。并非每个被遮挡指尖都能直接定位。','中'),
13:('A','关键邻帧未进入缓存',[95,115,135],'f95/115可见更多手指形态，但实际输入仅f120/130/135/140/145，共5帧。应先修复缺帧/关联，不能直接断言时序网络看到了却没学会。','较高'),
16:('B','整段缺乏充分直接线索',[10,90,145],'已查看f0–149逐帧联系表。掌部和拇指可见，但围绕玩具屋底座背面的抓握指节持续缺乏可辨定位线索；掌部、接触和运动仍有间接约束。','中；视觉审阅结论'),
41:('A','当前及邻帧已有线索',[55,65,80],'伸出的手指在目标及邻帧可辨，后续姿态也有变化。大误差不能全归于信息缺失；需检查定位、左右手/关节对应及裁剪。','较高')}
items=[]
for ident,(cat,sub,proof,reason,confidence) in data.items():
    q=json.loads((O/f'case_{ident}_metadata.json').read_text())
    actual=[f for f in q['actual_model_frames'] if f is not None]
    evidence=[dict(frame=f,offset_s=round((f-q['frame'])/q['fps'],3),frame_present_in_model_cache=f in actual) for f in proof]
    item={k:q[k] for k in ['id','sequence','clip','frame','side','before_px','after_px']}
    item.update(category=cat,subtype=sub,reason=reason,confidence=confidence,evidence=evidence,actual_model_frames=actual,review_scope='f0–149，约4.97秒',review_sampling='all 150 frames contact sheets' if ident==16 else 'every 5 frames; positive evidence only')
    items.append(item)
    fig,axes=plt.subplots(1,3,figsize=(12,4.4),layout='constrained')
    for ax,e in zip(axes,evidence):
        f=e['frame'];im=cv2.imread(f"/mnt/why/HOT3D/export/images/dit_v3_locked/{q['sequence']}/clip-{q['clip']:06d}/{f:06d}.jpg")
        roi=q['frames_metadata'][f]['roi'];ax.imshow(np.rot90(crop_image(im,np.asarray(roi))[:,:,::-1]));ax.axis('off')
        ax.set_title(f"f{f:03d} {e['offset_s']:+.2f}s | "+('TARGET' if f==q['frame'] else 'evidence')+'\n'+('frame in cache' if e['frame_present_in_model_cache'] else 'frame absent from cache'),fontsize=10)
    fig.suptitle(f"Case {ident} | {q['side']} hand | original RGB, audit-only GT crop",fontsize=12)
    fig.savefig(O/f'case_{ident}_evidence.png',dpi=140);plt.close(fig)
payload=dict(scope='47个hard case内，v4默认回归门控后平均手指误差>40px的8个窗口',categories={'A':7,'B':1},method='人工视觉审阅；不是逐关节可见性真值；未进行因果消融或新训练',cases=items)
(O/'classification.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
head='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>失败案例：时序线索审阅</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:36px auto;padding:0 20px;background:#f5f7fa;color:#172333}h1{font-size:29px}section{background:white;padding:24px;border-radius:12px;margin:24px 0}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}.note{color:#536174}a{color:#1269aa}</style><h1>失败案例：邻帧有线索，还是整段缺线索？</h1><p><b>审阅归类：A 类 7 例；B 类 1 例。</b>范围是47个自然 hard case 中，v4默认时序回归门控后平均手指误差仍 &gt;40 px 的8个窗口。不是全部失败分布，也不是DiT专属归因。</p><p>A：同一只手在邻帧存在可利用的直接或部分手指线索，但现有管线仍失败。B：约5秒片段内，被遮挡抓握指节持续缺乏充分直接线索。B不表示掌部、接触和运动完全无信息。</p><p class="note">每例覆盖f0–149。A类依据每5帧概览中的正向证据；B类额外查看全部150帧联系表。没有逐关节可见性真值，分类是视觉审阅，不能证明每个错误点可恢复/不可恢复。GT仅用于确认同一只手和展示裁剪；未进入推理。黑边可能来自裁剪或出视野，不作为遮挡证据。“帧在缓存”仅确认时间索引，不保证检测框正确包含线索。</p><table><tr><th>案例</th><th>类别与瓶颈</th><th>原始→修正后(px)</th><th>实际上下文帧数</th></tr>'''
rows=[];sections=[]
for i in items:
    n=i['id'];rows.append(f"<tr><td><a href='#case{n}'>#{n}</a></td><td>{i['category']} · {i['subtype']}</td><td>{i['before_px']:.1f} → {i['after_px']:.1f}</td><td>{len(i['actual_model_frames'])}/17</td></tr>")
    dense=''.join(f"<a href='case_{n}_dense_{p:03d}.png'>逐帧 {p}–{p+49}</a>　" for p in [0,50,100]) if n==16 else ''
    sections.append(f"<section id='case{n}'><h2>#{n} · {i['category']} · {i['subtype']}</h2><p>{i['reason']}</p><p class='note'>判断把握：{i['confidence']}。{i['sequence']} / clip {i['clip']} / 目标 f{i['frame']} / {i['side']}。</p><img src='case_{n}_evidence.png' alt='案例{n}原始帧证据'><p>实际缓存帧：{', '.join(map(str,i['actual_model_frames']))}</p><details><summary>约5秒完整概览</summary><img loading='lazy' src='case_{n}_overview.png'></details><p>{dense}</p></section>")
tail='''<section><h2>对改进方向的含义</h2><p>优先修复A类：先查上下文缺帧与手部关联，再验证更长的双向上下文、时序对应和RGB融合。#13已有明确输入缺口，#6暴露当前窗口长度限制；其余不能仅凭本次审阅区分裁剪、编码和融合的责任。</p><p>B类应保留不确定性、多个姿态候选和人工复核。继续增大模型不能凭空确认被遮挡指尖的唯一真值。</p><p>此次未修改模型或门控，也未用这些案例重新选择超参数。后续据此开发后，这些案例只能作为开发诊断，效果需另用独立片段验证。</p><a href='classification.json'>机器可读分类与证据帧</a> · <a href='../report.html'>返回原报告</a></section></html>'''
(O/'report.html').write_text(head+''.join(rows)+'</table>'+''.join(sections)+tail,encoding='utf-8')
assert len(items)==8 and sum(i['category']=='B' for i in items)==1
print('Saved classification.json, report.html and 8 evidence triptychs')
