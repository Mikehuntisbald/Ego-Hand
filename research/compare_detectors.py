"""Frozen-checkpoint HOT3D detection comparison; never trains or changes old studies."""
import argparse
import contextlib
import hashlib
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from ultralytics import YOLO


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def download_wilor(out):
    dest = out / 'wilor_detector.pt'
    receipt = out / 'wilor_download.json'
    if dest.exists() and receipt.exists():
        assert sha(dest) == json.loads(receipt.read_text())['sha256']
        return dest
    errors = []
    for domain in ['huggingface.co', 'hf-mirror.com']:
        try:
            api = requests.get(f'https://{domain}/api/spaces/rolpotamias/WiLoR', timeout=30)
            api.raise_for_status()
            revision = api.json()['sha']
            url = f'https://{domain}/spaces/rolpotamias/WiLoR/resolve/{revision}/pretrained_models/detector.pt'
            part = dest.with_suffix('.partial')
            with requests.get(url, timeout=(20, 90), stream=True) as r:
                r.raise_for_status()
                with part.open('wb') as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)
            assert part.stat().st_size > 1000000
            part.replace(dest)
            save(receipt, dict(repo='rolpotamias/WiLoR', revision=revision, url=url,
                               sha256=sha(dest), bytes=dest.stat().st_size))
            return dest
        except Exception as e:
            errors.append(f'{domain}: {type(e).__name__}: {e}')
    raise RuntimeError(errors)


def load_frames(root):
    result = []
    protocol = json.loads((root / 'experiments/dit_subject_oof_v2/protocol.json').read_text())
    paths = [('tune', p) for p in sorted((root / 'export/annotations/val').glob('P0003_*/*.jsonl'))]
    for seq in protocol['fresh_sequences']:
        paths.extend(('test', root / 'export/annotations' / seq['split'] / seq['sequence'] /
                      f'clip-{clip:06d}.jsonl') for clip in seq['clips'])
    for split, path in paths:
        for index, line in enumerate(path.read_text().splitlines()):
            if index % 5:
                continue
            row = json.loads(line)
            hands = []
            width, height = row['image_size']
            for hand in row['hands']:
                b = hand['box_amodal_xyxy']
                vis = hand['modeled_hand_visible_fraction']
                if b is None or vis is None or vis <= 0 or hand['xyz_camera_m'][5][2] <= .05:
                    continue
                b = np.clip(b, [0, 0, 0, 0], [width, height, width, height])
                if min(b[2:] - b[:2]) < 2:
                    continue
                hands.append(dict(box=b.tolist(), visibility=vis, side=hand['side'],
                                  truncated=bool(b[0] <= 0 or b[1] <= 0 or b[2] >= width or b[3] >= height)))
            image = root / 'export' / row['image']
            assert image.exists(), image
            result.append(dict(split=split, image=str(image), sequence=row['sequence'],
                               clip=row['clip'], frame=row['frame'], hands=hands))
    assert any(x['split'] == 'tune' for x in result)
    assert sum(x['split'] == 'test' for x in result) == 720
    return result


def iou(a, b):
    a = np.asarray(a).reshape(-1, 4)
    b = np.asarray(b).reshape(-1, 4)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.maximum(hi - lo, 0).prod(-1)
    return inter / (np.maximum(a[:, 2:] - a[:, :2], 0).prod(-1)[:, None] +
                    np.maximum(b[:, 2:] - b[:, :2], 0).prod(-1)[None] - inter + 1e-9)


def matches(frame, prediction, threshold, overlap=.5):
    order = np.argsort(-np.asarray(prediction['scores']), kind='stable')
    order = order[np.asarray(prediction['scores'])[order] >= threshold]
    pairs = iou([prediction['boxes'][i] for i in order], [h['box'] for h in frame['hands']])
    assigned = np.full(len(order), -1, dtype=int)
    used = np.zeros(len(frame['hands']), dtype=bool)
    for pos in range(len(order)):
        candidates = np.flatnonzero((pairs[pos] >= overlap) & ~used)
        if len(candidates):
            best = candidates[np.argmax(pairs[pos, candidates])]
            used[best] = True
            assigned[pos] = best
    return order, assigned, used


def curve(frames, predictions, overlap=.5):
    scores, correct = [], []
    for f, p in zip(frames, predictions):
        order, assigned, _ = matches(f, p, 0, overlap)
        scores.extend(p['scores'][i] for i in order)
        correct.extend(assigned >= 0)
    scores = np.asarray(scores)
    order = np.argsort(-scores, kind='stable')
    tp = np.asarray(correct, dtype=float)[order].cumsum()
    fp = np.arange(1, len(order) + 1) - tp
    total = sum(len(f['hands']) for f in frames)
    recall = tp / max(1, total)
    precision = tp / np.maximum(tp + fp, 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    ap = np.mean([envelope[recall >= r].max() if np.any(recall >= r) else 0 for r in np.linspace(0, 1, 101)])
    # Operating points are computed only at complete equal-score groups.
    ends = np.r_[np.flatnonzero(np.diff(scores[order]) != 0), len(order)-1] if len(order) else np.array([], int)
    return dict(ap=float(ap), scores=scores[order][ends], recall=recall[ends],
                precision=precision[ends], fp_per_frame=fp[ends] / len(frames))


def operating(frames, predictions, threshold):
    groups = {g: [0, 0] for g in ['all', 'visible_ge_0.5', 'occluded_lt_0.5', 'severe_lt_0.25', 'truncated']}
    tp = fp = 0
    ious_matched = []
    coverage = []
    for f, p in zip(frames, predictions):
        order, assigned, used = matches(f, p, threshold)
        tp += int(used.sum())
        fp += len(order) - int(used.sum())
        for j, h in enumerate(f['hands']):
            labels = ['all', 'visible_ge_0.5' if h['visibility'] >= .5 else 'occluded_lt_0.5']
            if h['visibility'] < .25:
                labels.append('severe_lt_0.25')
            if h['truncated']:
                labels.append('truncated')
            for g in labels:
                groups[g][0] += int(used[j])
                groups[g][1] += 1
        for pos, gt_index in enumerate(assigned):
            if gt_index < 0:
                continue
            pred = np.asarray(p['boxes'][order[pos]])
            gt = np.asarray(f['hands'][gt_index]['box'])
            ious_matched.append(float(iou([pred], [gt])[0, 0]))
            area = np.maximum(np.minimum(pred[2:], gt[2:])-np.maximum(pred[:2], gt[:2]), 0).prod()
            coverage.append(float(area / max(1e-9, np.prod(gt[2:]-gt[:2]))))
    n = groups['all'][1]
    return dict(threshold=float(threshold), frames=len(frames), gt_hands=n, tp=tp, fp=fp,
                precision=tp/max(1, tp+fp), recall=tp/max(1, n), fp_per_frame=fp/len(frames),
                matched_mean_iou=float(np.mean(ious_matched)) if ious_matched else None,
                matched_mean_gt_box_coverage=float(np.mean(coverage)) if coverage else None,
                groups={k: dict(tp=t, total=n, recall=t/n if n else None) for k, (t,n) in groups.items()})


def evaluate(frames, predictions):
    tune_ids = [i for i,f in enumerate(frames) if f['split']=='tune']
    test_ids = [i for i,f in enumerate(frames) if f['split']=='test']
    tf, tp = [frames[i] for i in tune_ids], [predictions[i] for i in tune_ids]
    ef, ep = [frames[i] for i in test_ids], [predictions[i] for i in test_ids]
    c = curve(tf, tp)
    f1 = 2*c['precision']*c['recall'] / np.maximum(c['precision']+c['recall'], 1e-9)
    thresholds = {'tune_best_f1': float(c['scores'][np.argmax(f1)]) if len(f1) else 1.000001}
    allowed = np.flatnonzero(c['fp_per_frame'] <= .05)
    thresholds['tune_fp_budget_0.05'] = float(c['scores'][allowed[np.argmax(c['recall'][allowed])]]) if len(allowed) else 1.000001
    thresholds['fixed_0.01'] = .01
    thresholds['fixed_0.3'] = .3
    aps = {f'{v:.2f}': curve(ef, ep, v)['ap'] for v in np.linspace(.5,.95,10)}
    return dict(ap50=aps['0.50'], ap75=aps['0.75'], map50_95=float(np.mean(list(aps.values()))), ap_by_iou=aps,
                operating_points={k:dict(tune=operating(tf,tp,t),test=operating(ef,ep,t)) for k,t in thresholds.items()})


def benchmark(model, frames, args):
    chosen = [f for f in frames if f['split']=='test'][::9][:80]
    images = [cv2.imread(f['image']) for f in chosen]
    settings = dict(imgsz=args.imgsz, device=args.device, conf=.001, max_det=100, iou=.7, half=False, verbose=False, save=False)
    for im in images[:10]:
        model.predict(im, **settings)
    times=[]
    for im in images:
        torch.cuda.synchronize(torch.device('cuda:' + args.device))
        start=time.perf_counter()
        model.predict(im, **settings)
        torch.cuda.synchronize(torch.device('cuda:' + args.device))
        times.append((time.perf_counter()-start)*1000)
    return dict(batch=1, samples=len(times), mean_ms=float(np.mean(times)), median_ms=float(np.median(times)),
                p95_ms=float(np.quantile(times,.95)), scope='Predecoded BGR numpy image to detections; preprocess, inference, postprocess included; disk decode excluded; shared GPU')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/mnt/why/HOT3D'))
    parser.add_argument('--out', type=Path, default=Path('/mnt/why/HOT3D/experiments/detector_compare_wilor_20261003'))
    parser.add_argument('--device', default='3')
    parser.add_argument('--imgsz', type=int, default=960)
    args=parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    frames=load_frames(args.root)
    save(args.out/'frames.json', frames)
    protocol=dict(purpose='Compare existing deployment checkpoints, not equal-training architecture superiority',
                  tune='P0003 only, frame stride 5', test='Previously evaluated 24 clips / 6 sequences of P0010/P0015; 720 frames',
                  training='YOLO26s HOT3D fine-tuned vs WiLoR released detector; WiLoR upstream training overlap not audited',
                  labels='Observable amodal clipped boxes; all predicted classes collapsed to hand for matching; native model postprocessing retained',
                  metrics='101-point interpolated single-class AP at IoU .50:.05:.95; score-ordered greedy matching; not full COCO area/maxDet suite',
                  threshold_selection='P0003 best F1 and highest recall at <=0.05 FP/frame, frozen for test',
                  imgsz=args.imgsz, precision='FP32', min_conf=.001, max_det=100, nms_iou=.7,
                  limitations=['Annotation-policy mismatch between amodal GT and pretrained visible boxes may affect IoU.',
                               'No unannotated-hand adjudication; no claim of wholly new test or controlled retraining.',
                               'Timing on shared GPU is descriptive, not isolated hardware benchmark.'])
    save(args.out/'protocol.json', protocol)
    weights={'yolo26_hot3d':args.root/'experiments/dit_lowconfidence_v1/detector/weights/best.pt',
             'wilor_released':download_wilor(args.out)}
    results={}
    for name, weight in weights.items():
        print(json.dumps(dict(stage='loading',model=name)), flush=True)
        model=YOLO(str(weight))
        info=dict(weight=str(weight), sha256=sha(weight), task=model.task, names=model.names,
                  parameters=sum(p.numel() for p in model.model.parameters()))
        cache=args.out/f'{name}_predictions.json'
        if cache.exists():
            saved=json.loads(cache.read_text())
            assert saved['model']['sha256']==info['sha256'] and saved['frames_sha256']==sha(args.out/'frames.json')
            assert saved['protocol_sha256']==sha(args.out/'protocol.json')
            predictions=saved['predictions']
        else:
            predictions=[]
            for start in range(0,len(frames),16):
                batch=frames[start:start+16]
                outputs=model.predict([f['image'] for f in batch], imgsz=args.imgsz, device=args.device, batch=16,
                                      conf=.001, max_det=100, iou=.7, half=False, verbose=False, save=False)
                for p in outputs:
                    predictions.append(dict(boxes=p.boxes.xyxy.cpu().tolist(), scores=p.boxes.conf.cpu().tolist(), classes=p.boxes.cls.cpu().tolist()))
                if start%160==0:
                    print(json.dumps(dict(model=name,done=len(predictions),total=len(frames))),flush=True)
            assert len(predictions)==len(frames)
            save(cache, dict(model=info, frames_sha256=sha(args.out/'frames.json'), protocol_sha256=sha(args.out/'protocol.json'), predictions=predictions))
        result=dict(model=info,metrics=evaluate(frames,predictions),timing=benchmark(model,frames,args))
        save(args.out/f'{name}_results.json', result)
        results[name]=result
        print(json.dumps(dict(stage='complete',model=name,results=result)),flush=True)
        del model
        torch.cuda.empty_cache()
    save(args.out/'comparison.json',dict(protocol=protocol,results=results))
    print('COMPARISON_COMPLETE',flush=True)


if __name__=='__main__':
    main()
