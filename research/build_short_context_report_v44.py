"""Reuse the audited RGB/3D viewer with explicit temporal-window labels."""
import re
from pathlib import Path


def replace_once(source,before,after):
    assert source.count(before)==1,(before,source.count(before))
    return source.replace(before,after)


def main():
    here=Path(__file__).parent
    template=(here/'matched_dit_compare_v43.html').read_text()
    style=re.search(r'<style>(.*?)</style>',template,re.S).group(1)
    script=re.search(r'<script>(.*?)</script>',template,re.S).group(1)
    script=replace_once(script,"const labels={gt:'GT',v16:'匹配时序回归',v39:'DiT 参数平均',v42:'DiT＋整段选候选'};",
        "const labels={gt:'GT',v16:'±4秒 DiT整段选择',v39:'±1.6秒 DiT参数平均',v42:'±1.6秒 DiT整段选择'};")
    trials_start=script.index('const trials=[');trials_end=script.index("document.getElementById('table')",trials_start)
    script=script[:trials_start]+"const trials=Object.values(S.methods).map(v=>[v.label,v.report,v.diagnostics]);\n"+script[trials_end:]
    script=script.replace('匹配回归原本','±4秒DiT原本').replace('个在DiT变成','个在±1.6秒DiT变成')
    script=replace_once(script,"f.relative_mm[k]===null?'该帧GT不完整':`相对手指误差：${f.relative_mm[k].toFixed(1)} mm`",
        "f.relative_mm[k]===null?(f.focus_joint_relative_mm[k]===null?'该帧GT不完整':`GT不完整；选中关节误差 ${f.focus_joint_relative_mm[k].toFixed(1)} mm`):`相对手指误差：${f.relative_mm[k].toFixed(1)} mm`")
    header='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HOT3D v44：DiT窗口 ±4秒 vs ±1.6秒</title><style>__STYLE__</style><main>
<h1>v44：DiT 时序窗口 ±4秒 → ±1.6秒</h1>
<p>同一网络权重、同一RGB空间特征和中心帧；只改变时序采样，比较参数平均与整段候选选择。</p>
<div class="notice">这是同权重推理对照，模型仍由±4秒窗口训练。±1.6秒保持17个时间槽，近处密、远处疏，正向间隔为0.033、0.067、0.10、0.20、0.33、0.60、1.0、1.6秒，负向对称。RGB、3D初值与时间信息同步重采样，同一个风险网络在新窗口上重新计算。手模型、侧别修复、4候选/10步DDIM和运动约束相同。默认v42保留。</div>
<section class="panel"><h2>结果与代价</h2><div id="cards" class="cards"></div><div style="overflow:auto"><table id="table"></table></div><p id="tradeoff"></p>
<p>2673观测的有效时间槽平均由11.67增至13.36。最佳整段选择的相对误差17.18→17.07mm，相机误差47.88→47.90mm；14个高误差困难窗中237坏点的恢复仍为135。收益很小，不能证明短窗口训练一定更好。</p>
<p class="muted">352开发中心窗，4条已用P0003序列；没有新受试者测试。窗口范围限制网络和风险特征的上下文；候选选择与约束仍使用整段轨迹。170个额外时刻仅作参数插值；缺口超过0.55秒分段。稳定、手型合理与隐藏点位正确分开评估。</p></section>
<section class="panel"><h2>改善、退步和共同失败</h2><div id="cases" class="cases"></div><h2 id="title" style="margin-top:20px"></h2><p id="description"></p><p id="meta" class="muted"></p>
<div class="controls"><button id="play">播放</button><button id="focus">定位事件帧</button><input id="slider" type="range" min="0" max="1" value="0"><span id="time" class="count"></span></div>
<p class="muted">上排为原始RGB裁剪及投影，下排为腕相对世界3D；拖动任一3D视图可同步旋转。黄色圈为选中关节。没有人工RGB遮挡，高误差不代表真实不可见性标签。</p>
<div id="views" class="views"></div><div class="links"><a id="full" target="_blank">未裁剪原图</a><a id="snapshot" target="_blank">静态对照图</a></div><img id="curves" class="plot" alt="逐帧运动、骨长与误差">
<details><summary>事件帧静态对照</summary><img id="still" class="plot" alt="RGB与3D对比"></details></section>
<section class="panel"><h2>实验入口与验证</h2>
<p><code>complete_hand_tracks_dit_v44.py --input tracks.json --output short.json --device cuda:0</code>；默认<code>--context-s 1.6</code>，可用<code>--context-s 4</code>作长窗口对照。沿用YOLO预测框/轨迹与相机元数据，<code>--prepared</code>可使用已有WiLoR结果。</p>
<p>原±4秒生成器和风险输出逐点复现；短窗口生成器/选择器GT投毒一致；保存参数重解码和相同运动限值检查通过。实际RGB→WiLoR→短窗口DiT→整段约束的16帧入口通过。</p>
<p class="muted">静态图片、报告数据和JS语法检查；浏览器交互未实测。未进行短窗口重训，未证明完全缺乏线索时恢复真值或消除自碰撞。</p>
<div class="links"><a href="full_summary.json">完整统计</a><a href="sequence_constraint_check.json">运动边界</a><a href="verification.json">验证记录</a><a href="protocol.json">实验协议</a></div></section></main>
<script id="report-data" type="application/json">__REPORT_DATA__</script><script>__SCRIPT__</script></html>'''
    (here/'short_context_compare_v44.html').write_text(header.replace('__STYLE__',style).replace('__SCRIPT__',script),encoding='utf-8')
    source=(here/'build_matched_dit_report_v43.py').read_text()
    changes={
        "default='matched_stability_v43/dit_side_consistent'":"default='short_context_v44/dit'",
        "out=root/'matched_dit_review_v43'":"out=root/'short_context_review_v44'",
        "root/'matched_stability_v43/regression_side_consistent/mean.pt'":"root/'matched_stability_v43/dit_side_consistent/sequence.pt'",
        "TITLES=dict(gt='GT',v16='Matched regression',v39='DiT parameter mean',v42='DiT sequence')":"TITLES=dict(gt='GT',v16='DiT +/-4s sequence',v39='DiT +/-1.6s mean',v42='DiT +/-1.6s sequence')",
        "'matched_dit_compare_v43.html'":"'short_context_compare_v44.html'",
        "root/'matched_dit_review_v43.zip'":"root/'short_context_review_v44.zip'",
        "'匹配回归相对误差≤10mm的点被DiT改到>20mm，保留局部退步案例。'":"'±4秒DiT相对误差≤10mm的点被±1.6秒DiT改到>20mm，保留局部退步案例。'",
        "'DiT参数平均与整段选候选相对平均误差都>30mm。高误差不等于逐指不可见真值。'":"'±1.6秒DiT参数平均与整段选择相对平均误差都>30mm。高误差不等于逐指不可见真值。'",
        "'DiT：姿态误差降低'":"'短窗口：姿态误差降低'",
        "(means['v16']-means['v42']>5)":"(means['v16']-means['v42']>1)",
        "'匹配回归与DiT共享同样的侧别可靠性处理、手模型和运动边界；GT只用于事后展示选择。'":"'相同DiT权重、侧别处理、手模型及运动边界；比较±4秒与±1.6秒，该类帧相对均值改善>1mm。GT只用于事后挑选展示案例。'",
        "'比较同一个DiT的参数平均与整段一致性路径；案例统计为事后诊断，不作为GT选择规则。'":"'比较±1.6秒DiT的参数平均与整段一致性路径；案例统计为事后诊断，不作为GT选择规则。'",
        "hits=torch.nonzero(condition&complete).flatten().tolist();hits.sort(key=lambda j:float(severity[j]),reverse=True)":"hits=torch.nonzero(condition & (torch.ones_like(complete) if key=='accuracy_loss' else complete)).flatten().tolist();hits.sort(key=lambda j:float(severity[j]),reverse=True)",
        "f['relative_mm'][name]=float(means[name][n]) if complete[n] else None":"f['relative_mm'][name]=float(means[name][n]) if complete[n] else None\n                f.setdefault('focus_joint_relative_mm',{})[name]=float(errors[name][n,joint]) if valid[n,joint] and valid[n,5] else None",
        "cloud=np.concatenate([poses[k][ix].numpy().reshape(-1,3)*1000 for k in variants]);cloud=cloud[np.isfinite(cloud).all(-1)]":"cloud_parts=[]\n        for k in variants:\n            cloud_part=poses[k][ix].numpy().copy()*1000\n            if k=='gt':cloud_part[~valid[ix].numpy()]=np.nan\n            cloud_parts.append(cloud_part.reshape(-1,3))\n        cloud=np.concatenate(cloud_parts);cloud=cloud[np.isfinite(cloud).all(-1)]",
        "ax.legend()":"ax.legend(fontsize=8)",
        "f'Relative error: {means[name][focus]:.1f} mm'":"f'Valid-point relative error: {means[name][focus]:.1f} mm'",
    }
    for before,after in changes.items():
        if before=="root/'matched_dit_review_v43.zip'":
            assert source.count(before)==2
            source=source.replace(before,after)
        else:source=replace_once(source,before,after)
    source=source.replace('label=name,c=COLORS[name]','label=TITLES[name],c=COLORS[name]')
    # No jump/bone rescue category is valid: both long and short outputs
    # already obey exactly the same structure and motion constraints.
    source='\n'.join(line for line in source.splitlines() if not any(line.strip().startswith(prefix) for prefix in ["pair_case('jump_resolved'","pair_case('root_resolved'","frame_case('bone_resolved'"]))+'\n'
    from hand3d_v8_common import V7
    out=V7.parent/'short_context_review_v44'
    (out/'report_builder_snapshot.py').write_text(source)
    namespace=dict(__name__='__main__',__file__=str(here/'build_short_context_report_v44.py'))
    exec(compile(source,str(out/'report_builder_snapshot.py'),'exec'),namespace)


if __name__=='__main__':main()
