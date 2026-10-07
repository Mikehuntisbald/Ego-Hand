import json,html
from pathlib import Path
import spatial_rgb_common as s
import numpy as np,torch,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    folder=s.RUN/'delivery';folder.mkdir(exist_ok=True)
    result=json.loads((s.RUN/'test_results.json').read_text());old=json.loads((s.OLD/'test_results.json').read_text())
    gate=json.loads((s.RUN/'rgb_localization_gate.json').read_text())
    arms=[('Old RGB DiT',old['methods']['rgb_dit']),('Old RGB regression',old['methods']['rgb_regression']),
        ('New RGB DiT',result['methods']['rgb_dit']),('New RGB regression',result['methods']['rgb_regression']),
        ('New tracks DiT',result['methods']['tracks_dit']),('New tracks regression',result['methods']['tracks_regression'])]
    modes=['partial/all','full/all','natural/high_hand_occlusion_proxy'];titles=['Partial artificial occlusion','Entire ROI occluded','Natural image / high hand occlusion proxy']
    colors=['#b6bdc8','#d0d4dc','#227ab3','#ef9b37','#70ac8a','#aa89bf']
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    for ax,mode,title in zip(axes,modes,titles):
        values=[arm['reports'][mode]['mean_px'] for _,arm in arms]
        bars=ax.barh(np.arange(len(arms)),values,color=colors);ax.invert_yaxis()
        ax.set_yticks(range(len(arms)),[a for a,_ in arms]);ax.set_xlabel('Finger keypoint error (pixels), lower is better');ax.set_title(title,fontsize=11)
        ax.bar_label(bars,fmt='%.2f',padding=3);ax.set_xlim(0,max(values)*1.2);ax.grid(axis='x',alpha=.15)
    fig.suptitle('Strong spatial RGB conditioning | Same reused task holdout: 2 subjects, 8 sequences',fontsize=13)
    fig.savefig(folder/'comparison.png',dpi=160);plt.close(fig)
    # Show a representative gap-6 sample near the median paired improvement.
    new=np.load(s.RUN/'rgb_dit_predictions.npz');before=np.load(s.OLD/'rgb_dit_predictions.npz');reg=np.load(s.RUN/'rgb_regression_predictions.npz')
    mask=new['mask'];e0=np.linalg.norm(before['predicted']-new['gt'],axis=-1)*1408;e1=np.linalg.norm(new['predicted']-new['gt'],axis=-1)*1408
    candidates=np.where((new['labels']=='partial')&(new['gaps']==6)&(mask.sum(1)>=3))[0]
    improvements=((e0-e1)*mask).sum(1)/mask.sum(1).clip(1)
    idx=int(candidates[np.argmin(np.abs(improvements[candidates]-np.median(improvements[candidates])))])
    row_index=int(new['sample_indices'][idx]);records,index=s.records_and_index();fid=int(index['feature_ids'][row_index,8]);record=records[fid-1]
    roi=index['roi'][fid].numpy()*1408;variant=int(new['variant'][idx]);rect=index['rectangles'][fid,variant].numpy()
    image=cv2.imread(record['image']);native=roi[:2]+rect.reshape(2,2)*(roi[2:]-roi[:2])/256
    lo=np.maximum(np.floor(native[0]).astype(int)-1,0);hi=np.minimum(np.ceil(native[1]).astype(int)+1,[1408,1408]);image[lo[1]:hi[1],lo[0]:hi[0]]=32+64*((record['clip']+1)%4)
    from offline_rgb_encoder import crop_image
    crop=np.rot90(crop_image(image,roi)[:,:,::-1]);fig,axes=plt.subplots(1,3,figsize=(11,4.2),layout='constrained')
    for ax,name,pred in zip(axes,['Old RGB DiT','New RGB DiT','New RGB regression'],[before['predicted'][idx],new['predicted'][idx],reg['predicted'][idx]]):
        ax.imshow(crop);ax.axis('off')
        for pts,color,marker,label in [(new['gt'][idx],'#00e6b8','o','Reference'),(pred,'#ff6b67','x','Prediction')]:
            xy=(pts*1408-roi[:2])/(roi[2:]-roi[:2])*256;xy=np.stack([xy[:,1],255-xy[:,0]],-1)
            m=mask[idx];ax.scatter(xy[m,0],xy[m,1],s=30,c=color,marker=marker,label=label,linewidths=1.4)
        error=np.linalg.norm(pred-new['gt'][idx],axis=-1)[mask[idx]].mean()*1408
        ax.set_title(f'{name}\nHidden-point error: {error:.1f} px',fontsize=11);ax.set_xlim(0,255);ax.set_ylim(255,0)
    axes[-1].legend(loc='lower right',fontsize=8)
    fig.suptitle('Artificial occlusion, 6-frame gap | Example nearest the median DiT improvement',fontsize=11)
    fig.savefig(folder/'example.png',dpi=170);plt.close(fig)
    s.save(folder/'example_figure_context.json',dict(prediction_row=idx,window_index=row_index,selection='Nearest median paired DiT improvement among partial/gap6 with >=3 hidden valid finger points',image=record['image']))
    rows=''.join('<tr><td>'+html.escape(name)+'</td>'+''.join(f'<td>{arm["reports"][m]["mean_px"]:.2f}</td>' for m in modes)+'</tr>' for name,arm in arms)
    ci=result['comparisons']['rgb_dit_minus_v2_rgb_dit']['partial/all'];ab=result['development_rgb_ablations']
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>HOT3D 强空间 RGB v3</title>
<style>body{{font:16px/1.6 system-ui;margin:36px auto;max-width:1120px;color:#203046}}h1{{font-size:28px}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:8px;border-bottom:1px solid #ddd}}img{{max-width:100%}}pre{{white-space:pre-wrap;background:#f3f5f8;padding:16px}}</style>
<h1>HOT3D：强视觉编码器与空间 RGB 条件</h1>
<p>完整 WiLoR 手部预训练 ViT（32 层），保留 16×12 共 192 个空间位置。通道压缩到 128 维，空间不池化。17 帧双向条件，RGB 热图辅助监督，20% 整窗口关键点丢弃，15% 关节点丢弃；融合层可训练，没有极小增益门控。</p>
<p>独立视觉定位验证先通过，再训练回归和 DiT；所有模型在开发集选好并冻结后评估。遮挡在编码前加入，覆盖插值支撑域，框外像素清零。给定关键点原样保留。</p>
<table><tr><th>模型</th><th>人工部分遮挡 / px</th><th>整框遮挡 / px</th><th>自然图像高整手遮挡代理组 / px</th></tr>{rows}</table>
<p>新 RGB DiT 相对旧 RGB DiT 的人工部分遮挡误差变化：{ci['change_px']:.2f} px；按来源序列配对 bootstrap 的 95% 区间：{ci['ci95'][0]:.2f} 至 {ci['ci95'][1]:.2f} px。负值表示改善。</p>
<img src="comparison.png" alt="误差对比"><img src="example.png" alt="中位改善附近的样例">
<h2>模型是否使用 RGB</h2><pre>{html.escape(json.dumps(ab,indent=2))}</pre>
<h2>独立视觉定位验证</h2><pre>{html.escape(json.dumps(gate,indent=2))}</pre>
<h2>证据范围</h2><p>2 个保留受试者、8 个序列、2082 个窗口，沿用 v2 任务划分。这些数据在此前研究中已经使用，不能称为全新独立证据；WiLoR 上游预训练与 HOT3D 的样本重叠未知。自然遮挡只有整手可见性代理，未验证真实逐指完全遮挡。当前手框和相机标定仍需提供。</p>
<p>CLI：infer_spatial_rgb.py。示例输入 example_input.json；结果 example_rgb_dit.json / example_rgb_regression.json。自动生成点均需人工复核。经验不确定性仅在人工部分遮挡开发集上校准。</p></html>'''
    (folder/'report.html').write_text(page,encoding='utf-8')
    readme=f'''HOT3D spatial RGB v3
Remote run: {s.RUN}
Report: report.html; figures: comparison.png and example.png.
Input: example_input.json. This contains no ground truth. Each frame provides
camera calibration, native image path, supplied hand box, timestamp, 20 points
and observed flags. Missing coordinates may be null. Fixed native size 1408x1408.

Run on the existing server/environment:
/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python \\
  /mnt/why/hot3d_hand_residual/infer_spatial_rgb.py \\
  --input {s.RUN}/delivery/example_input.json \\
  --output /tmp/spatial_keypoints.json --arm rgb_dit --device cuda:0

Required pretrained backbone and MANO assets remain at:
{s.common.RUN}/assets
WiLoR source is resolved by {s.common.RUN}/source_path.txt.
The archive includes fitted heads/fusion weights, their source and SHA-256
checks, results, training histories, CLI example, and verification. It does
not duplicate the 2.56 GB upstream backbone or original HOT3D images.

Both regresssion and DiT consume all 17 RGB frames and surviving coordinates.
All 192 spatial cells are retained per frame; no global/4x4 pooling.
The visual backbone and validated channel stem are frozen. Fusion, spatial
heatmap decoder, and temporal heads are trainable. Camera and supplied hand
tracking are required; this does not restore a fully missing hand box.
Observed input coordinates are copied exactly. Every predicted point is a
candidate requiring review; uncertainty is not a natural-occlusion guarantee.

Evidence scope: reused task holdout, P0010/P0015, 8 source sequences, 2082 windows.
Upstream WiLoR HOT3D pretraining overlap is unknown. Artificial occlusion and
natural whole-hand visibility proxies are reported separately.
'''
    (folder/'README.txt').write_text(readme)

if __name__=='__main__':main()
