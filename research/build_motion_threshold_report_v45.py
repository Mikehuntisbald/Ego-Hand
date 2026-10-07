"""Real RGB and world-3D comparisons of fixed-path motion constraints."""
import re
from pathlib import Path


def once(source,before,after):
    assert source.count(before)==1,(before,source.count(before))
    return source.replace(before,after)


def main():
    here=Path(__file__).parent;original=(here/'matched_dit_compare_v43.html').read_text()
    style=re.search(r'<style>(.*?)</style>',original,re.S).group(1)
    script=re.search(r'<script>(.*?)</script>',original,re.S).group(1)
    script=once(script,"const labels={gt:'GT',v16:'匹配时序回归',v39:'DiT 参数平均',v42:'DiT＋整段选候选'};",
        "const labels={gt:'GT',v16:'当前严格配置',v39:'全部运动阈值×3',v42:'仅加速度阈值×2'};")
    begin=script.index('const trials=[');end=script.index("document.getElementById('table')",begin)
    script=script[:begin]+"const trials=Object.values(S.methods).map(v=>[v.label,v.report,v.diagnostics,v.protection]);\n"+script[end:]
    script=once(script,'<th>快速动作保留</th></tr>','<th>快速动作保留</th><th>沿GT方向保留</th><th>原好点改坏</th></tr>')
    script=once(script,'trials.map(([label,r,d])=>','trials.map(([label,r,d,p])=>')
    script=once(script,"${d?Math.round(d.fast_amplitude_ratio.median*100)+'%':'见统计'}</td></tr>",
        "${d?Math.round(d.fast_amplitude_ratio.median*100)+'%':'见统计'}</td><td>${Math.round(d.fast_directional_retention.median*100)}%</td><td>${p.strict_good_to_bad_points} / ${p.strict_good_points}</td></tr>")
    script=script.replace('DiT 异常跳变帧对','加速度×2 异常跳变帧对').replace('DiT 骨长极端帧','加速度×2 骨长极端帧')
    script=script.replace('匹配回归原本','当前严格配置原本').replace('个在DiT变成','个在加速度×2变成')
    header='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HOT3D v45：运动阈值与过度平滑诊断</title><style>__STYLE__</style><main>
<h1>v45：快速动作被哪里压低了？</h1>
<div class="notice">同一±1.6秒DiT权重、同一4候选及已固定的整段选择路径。所有对照保留手模型、静态手型/侧别、RGB证据和350步优化；分别改变速度、加速度或平滑设置。GT只用于输出冻结后的诊断，默认v42不改。</div>
<section class="panel"><h2>结论：主要在加速度相关的优化约束</h2>
<p>95ms初始平滑将有GT快速动作幅度从约102%压到56%，后续优化恢复到74%。只放宽速度×2仍约74.5%；只放宽加速度×2升至81.3%，本批异常跳变仍0。全部放宽×3保留85.7%，但出现10对异常跳变。</p>
<p>仅末端硬恢复×2不改变前述快速动作中位数；但31帧实际输出有变化，最大208mm，其中30帧没有有效GT、1帧有GT但不进入快速帧对统计。严格版有4段姿态/6段根部被收缩，最低姿态幅度0.3、根部0.19。不能用有GT运动指标不变推断所有输出不变。</p>
<p>进一步只减弱优化中的加速度尺度、保留原硬限，最终幅度反而降至65%，因为24/96段触发整段姿态收缩。只把加速度软正则减弱到1/4也退步：幅度66%、相对误差22.85mm。优化约束和末端恢复需要协调，不能只降低一个权重。</p>
<p>GT快速点2391对中，80对（3.35%）超过0.65m/s。腕旋转加速度2279组三帧中，202组（8.86%）超过30rad/s²；原始差分含标注噪声，这些统计仅作诊断，不用于推理调阈值。</p>
<div id="cards" class="cards"></div><div style="overflow:auto"><table id="table"></table></div><p id="tradeoff"></p>
<p class="muted">表内前四行为中间诊断阶段，不能当可部署结果；其余为参数空间最终输出。幅度与方向单独统计，更多移动不等于动作更准确。2673观测、352开发中心窗、4条已用P0003序列；未进行新主体评估。</p></section>
<section class="panel"><h2>逐帧看动作恢复与放宽的代价</h2><div id="cases" class="cases"></div><h2 id="title" style="margin-top:20px"></h2><p id="description"></p><p id="meta" class="muted"></p>
<div class="controls"><button id="play">播放</button><button id="focus">定位事件帧</button><input id="slider" type="range" min="0" max="1" value="0"><span id="time" class="count"></span></div>
<p class="muted">原始RGB裁剪及投影、腕相对世界3D；拖动3D视图同步旋转。没有人工RGB遮挡。GT只用于事后挑选展示事件，类别可重叠。</p><div id="views" class="views"></div><div class="links"><a id="full" target="_blank">原始全图</a><a id="snapshot" target="_blank">静态对照</a></div><img id="curves" class="plot" alt="运动、骨长和误差">
<details><summary>事件帧静态对比</summary><img id="still" class="plot" alt="RGB及3D对比"></details></section>
<section class="panel"><h2>范围与证据</h2><p>速度保持：根1.2m/s、腕相对指点0.65m/s、腕旋转5rad/s、关节6rad/s。加速度×2对照：根24m/s²、腕相对指点20m/s²、腕旋转60rad/s²、关节80rad/s²。软正则尺度、提前惩罚和末端检查一起变；“仅末端×2”单独隔离硬恢复的作用。</p>
<p>已核对严格配置逐点复现v44、每个最终变体保存参数重解码与声明限值。候选和路径完全固定，网络未重训，也未改变原默认。95→45ms单独调整未改善最终幅度并损伤点位；联合减弱软正则也有明显退步。</p>
<p class="muted">本轮是固定候选的后处理诊断；没有重新运行原始RGB入口或修改部署配置。手模型合理与速度有界不保证碰撞、遮挡真值或完整运动正确。浏览器交互未实测。</p>
<div class="links"><a href="full_summary.json">完整对照</a><a href="stage_diagnosis.json">阶段分解</a><a href="gt_motion_audit.json">GT运动分布</a><a href="hard_restore_effect.json">硬恢复改变的全部帧</a><a href="protocol.json">实验协议</a><a href="verification.json">验证记录</a></div></section></main>
<script id="report-data" type="application/json">__REPORT_DATA__</script><script>__SCRIPT__</script></html>'''
    (here/'motion_threshold_compare_v45.html').write_text(header.replace('__STYLE__',style).replace('__SCRIPT__',script),encoding='utf-8')
    source=(here/'build_matched_dit_report_v43.py').read_text()
    changes={
        "default='matched_stability_v43/dit_side_consistent'":"default='motion_threshold_v45'",
        "out=root/'matched_dit_review_v43'":"out=root/'motion_threshold_review_v45'",
        "run/'sequence.pt'":"run/'acc_x2.pt'",
        "root/'matched_stability_v43/regression_side_consistent/mean.pt'":"run/'strict.pt'",
        "run/'mean.pt'":"run/'motion_x3.pt'",
        "TITLES=dict(gt='GT',v16='Matched regression',v39='DiT parameter mean',v42='DiT sequence')":"TITLES=dict(gt='GT',v16='Current strict',v39='All motion caps x3',v42='Acceleration caps x2')",
        "'matched_dit_compare_v43.html'":"'motion_threshold_compare_v45.html'",
        "root/'matched_dit_review_v43.zip'":"root/'motion_threshold_review_v45.zip'",
        "'v39 跳变 → v42 连续'":"'全部阈值×3出现跳变，加速度×2保持连续'",
        "'同一GT手的连续帧；GT相对移动<10mm、v39>30mm。这里检查实际输出，未剔除旧回退。'":"'同GT手连续帧：GT相对移动<10mm、全部运动阈值×3时预测>30mm。速度不放宽的加速度×2对照更稳。'",
        "'GT指端移动>10mm，而DiT保留不足一半幅度。这些点不是稳定成功的例子；平滑会损失运动细节。'":"'GT指点移动>10mm，加速度×2之后仍保留不足一半幅度。提高阈值没有解决所有信息缺失。'",
        "'匹配回归相对误差≤10mm的点被DiT改到>20mm，保留局部退步案例。'":"'当前严格配置≤10mm的点被加速度×2改到>20mm；即使平均误差接近仍有局部退步。'",
        "'DiT参数平均与整段选候选相对平均误差都>30mm。高误差不等于逐指不可见真值。'":"'全部×3与加速度×2均相对平均误差>30mm。增大运动幅度不能保证遮挡姿态正确。'",
        "'DiT：姿态误差降低'":"'加速度×2：局部姿态误差降低'",
        "(means['v16']-means['v42']>5)":"(means['v16']-means['v42']>1)",
        "'匹配回归与DiT共享同样的侧别可靠性处理、手模型和运动边界；GT只用于事后展示选择。'":"'同网络、同DiT候选和已固定路径；仅加速度尺度×2，相对均值改善>1mm。GT只用于事后展示。'",
        "'整段选候选：优于参数平均'":"'保持速度限制：优于全部阈值×3'",
        "'比较同一个DiT的参数平均与整段一致性路径；案例统计为事后诊断，不作为GT选择规则。'":"'比较加速度×2与速度/加速度全部×3；案例为事后诊断，不作为GT门控。'",
        "ax.legend()":"ax.legend(fontsize=8)",
    }
    for before,after in changes.items():
        expected=2 if before in ["root/'matched_dit_review_v43.zip'","run/'mean.pt'"] else 1
        assert source.count(before)==expected,(before,source.count(before))
        source=source.replace(before,after)
    source=source.replace('label=name,c=COLORS[name]','label=TITLES[name],c=COLORS[name]')
    source='\n'.join(line for line in source.splitlines() if not any(line.strip().startswith(prefix) for prefix in ["pair_case('root_resolved'","frame_case('bone_resolved'"]))+'\n'
    # Add a motion recovery event category that also checks the direction.
    target="    cases=[];used=set()"
    addition='''    default_ratio=moves['v16']/moves['gt'].clamp_min(1e-8)
    va=poses['v42'][b]-poses['v42'][a];vg=poses['gt'][b]-poses['gt'][a]
    direction=(va*vg).sum(-1)/vg.square().sum(-1).clamp_min(1e-10)
    recovered=fast&(default_ratio<.7)&(retention>.8)&(retention<1.2)&(direction>.6)
    pair_case('motion_recovered','加速度×2：动作幅度恢复','严格版不足70%，加速度×2保留80–120%，且沿GT方向的投影保留>60%。并非只增加无方向的抖动。',recovered,(moves['v42']-moves['v16'])*recovered)
    cases=[];used=set()'''
    source=once(source,target,addition)
    from hand3d_v8_common import V7
    out=V7.parent/'motion_threshold_review_v45';(out/'report_builder_snapshot.py').write_text(source)
    exec(compile(source,str(out/'report_builder_snapshot.py'),'exec'),dict(__name__='__main__',__file__=str(here/'build_motion_threshold_report_v45.py')))


if __name__=='__main__':main()
