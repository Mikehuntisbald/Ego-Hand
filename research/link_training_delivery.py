from pathlib import Path
import json
root=Path('/mnt/why/HOT3D');run=root/'experiments/dit_lowconfidence_v1'
page=root/'report.html';text=page.read_text()
banner='<div class="note">后续训练与独立测试已完成。此页保留数据准备阶段记录；当前模型与结论请查看 <a href="experiments/dit_lowconfidence_v1/training_report.html">训练报告</a>。</div>'
if '后续训练与独立测试已完成' not in text:page.write_text(text.replace('<main>','<main>'+banner,1))
plan=root.parent/'hot3d_hand_residual/experiment_plan.json'
data=json.loads(plan.read_text());data['coarse_3d_model_status']='trained YOLO26s visual backbone plus calibrated canonical20 pose head; checkpoint in coarse_fine/best.pt'
data['current_study_status']='trained and evaluated; no positive evidence for solving the three hard groups'
plan.write_text(json.dumps(data,indent=2))
print('Training delivery linked; previous dataset page retained as preparation-stage history.')
