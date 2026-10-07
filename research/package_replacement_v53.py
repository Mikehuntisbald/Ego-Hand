"""Completed experimental frontend and honest downstream admission status."""
import json,hashlib,html,shutil
from pathlib import Path
B=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53B_20261007')
def main():
    assert json.loads((B/'finalize_status.json').read_text())['phase']=='complete'
    out=B/'review';selection=json.loads((B/'strict_review_r1/selection.json').read_text());pose=json.loads((B/'full_glove_evaluation.json').read_text());audit=json.loads((out/'pose_damage_audit.json').read_text());stats=json.loads((B/'replacement_eval/statistics.json').read_text());natural=json.loads((out/'natural_checks.json').read_text())
    config=dict(frontend='RFDETRSegSmall hand detector + instance segmentation',checkpoint=selection['checkpoint'],checkpoint_sha256=selection['checkpoint_sha256'],resolution=576,threshold=selection['threshold'],postprocess=selection['policy'],detector_development_admitted=selection['development_qualified'],full_3D_replacement_admitted=False,default_replaced=False,reason='26/156 previously correct pose points harmed; 6-source paired pose CI crosses zero; identity and 3D GT missing in natural clinical replay',primary_YOLO26_replaced_in_experimental_pipeline=True,WiLoR_internal_hand_side_YOLO_helper_retained=True,inference_environment='/mnt/why/HOT3D/experiments/rfdetr_hand_instance_v52_20261007/venv/bin/python',three_D_environment='/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python',GT_inference_inputs=False,test_selection=False,noncausal_context_s=1.6,speed_limits_unchanged=True,soft_and_hard_acceleration_multiplier=2,mode='experimental offline annotation; review required')
    (B/'selected_frontend.json').write_text(json.dumps(config,indent=2));shutil.copy2(B/'selected_frontend.json',out/'selected_frontend.json')
    page=(out/'report.html').read_text();intro='<p class="note"><strong>结论：主手框检测升级成立，整套3D尚未准入。</strong>新增100DOH 400测试视频组/770手：检出60.1%→92.5%，精确率25.8%→76.1%；恢复260/307个漏检，损失11/463个原正确检出。手套完整3D投影37.1%→43.6%，恢复53点但损伤26/156个旧正确点，组CI跨零，默认保留。</p>'
    page=page.replace('<h2>手框：完整检出、正确检出损伤与候选开销</h2>',intro+'<h2>手框：完整检出、正确检出损伤与候选开销</h2>')
    extra=f'<h2>3D受损发生在哪里</h2><p>26个旧正确点受损全部发生在两种前端均检出该手的情况下；没有来自本组新漏检。其中10点集中在手侧判断发生变化的1只手。新框改变了裁剪/初始观测，手侧改变是关联证据，尚未做因果对照；优先稳定手侧并适配新的裁剪分布。</p>'
    for v in sorted(audit['gallery'],key=lambda v:(-v['lost'],-v['recovered']))[:6]:extra+=f'<figure><img loading="lazy" src="{v["image"]}"><figcaption>{html.escape(v["id"])}：损伤{v["lost"]}点、恢复{v["recovered"]}点。左右分别为原前端和RF前端；绿色人工2D点，同一3D后端。</figcaption></figure>'
    page=page.replace('<h2>真实护理：全120帧完整3D</h2>',extra+'<h2>真实护理：全120帧完整3D</h2>')
    page=page.replace('按源视频/交互对10000次配对bootstrap。','按记录的源视频/交互对/图像组10000次配对bootstrap；CPPE只有图像组，原采集会话未知。')
    page=page.replace('另外', '另外')
    notes=f'<h2>数据审计与用途</h2><p>100DOH6000训练图实际来自5981个源视频（部分文件名带片段号）；开发300组、测试400组分别唯一，三者源视频交集0，训练与新增测试文件hash无重复。冻结数据记录保持原样，修正审计单独保存。既有其他域测试仍属于回放。护理120帧输出{natural["tracks"]}条候选轨迹，其中3条115帧；其余包含短轨迹与额外候选，需要复核。硬约束与骨长稳定不能证明身份正确或手指3D准确。速度上限不变、软硬加速度×2，保留±1.6s非因果上下文。</p><p>本轮测试标签未参与选择权重/阈值。GT mask、框和关键点不是推理输入。optimizer/scheduler/Python、NumPy、Torch、CUDA RNG已保存；精确断点执行一致性未实测。作者下载页注明100DOH仅供非商业研究，当前权重作为科研实验。</p><p><a href="selected_frontend.json">实际可运行前端配置</a> · <a href="pose_damage_audit.json">3D损伤明细</a> · <a href="source_group_audit.json">视频级分组修正</a> · <a href="fresh_test_hash_audit.json">新增测试重复检查</a> · <a href="training_verification.json">实际权重/梯度/重载核验</a> · <a href="dataset_license_download_page.html">作者下载页存档</a></p>'
    page=page.replace('</html>',notes+'</html>');(out/'report.html').write_text(page,encoding='utf-8')
    for name in ['source_group_audit.json','fresh_test_hash_audit.json','dataset_protocol.json']:
        shutil.copy2(B/name,out/name)
    license_page=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53_20261007/dataset_license_download_page.html');shutil.copy2(license_page,out/license_page.name)
    (B/'final_result.json').write_text(json.dumps(dict(training_complete=True,steps_stageA=4630,steps_stageB=7568,total_training_steps=12198,detector_upgrade_supported=True,full_3D_replacement_admitted=False,default_replaced=False,report=str(out/'report.html'),config=config,pose_damage=audit['correct_points_lost'],fresh_test=stats['statistics']['stageB']['100doh'],natural_candidate_tracks=natural['tracks']),indent=2))
    manifest={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in out.iterdir() if p.is_file()};(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('PACKAGED',out,flush=True)
if __name__=='__main__':main()
