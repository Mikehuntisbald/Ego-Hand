"""Download pinned public runtime/data assets into this experiment only."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time
import urllib.request

ROOT = Path('/mnt/why/public_headhand_h20_20261019')
ASSETS = [
    dict(name='cuvslam', file='cuvslam-17.0.0+cu12-cp310-cp310-manylinux_2_35_x86_64.whl',
         url='https://github.com/nvidia-isaac/cuVSLAM/releases/download/v17.0.0/cuvslam-17.0.0%2Bcu12-cp310-cp310-manylinux_2_35_x86_64.whl',
         sha256='0ae3113e43902e0c15423992450510aef241f84207718ed77a1d2d39f28744c1'),
    dict(name='ess', file='dnn_stereo_disparity_v4.1.0_onnx_trt10.13.tar.gz',
         url='https://api.ngc.nvidia.com/v2/models/nvidia/isaac/dnn_stereo_disparity/versions/4.1.0_onnx_trt10.13/files/dnn_stereo_disparity_v4.1.0_onnx_trt10.13.tar.gz',
         bytes=131749133, sha256=base64.b64decode('vqc0keZzKzMtJ1RzKVg8eoiSbNcBic8CwAr+SYwScas=').hex()),
    dict(name='hot3d_clip2102', file='clip-002102-full.tar',
         url='https://hf-mirror.com/datasets/bop-benchmark/hot3d/resolve/30fe9674782f32e1e5edba98476b6ff4300132c5/train_aria/clip-002102.tar',
         bytes=99522560, sha256='67e858a5a948eeda8497b9a06a27d8c4ac4c8bd271a5b93c05ec142230db91f8')
]


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def download(item):
    path = ROOT/'assets'/item['file']
    started = time.time()
    for attempt in range(3):
        try:
            if not path.exists():
                partial = path.with_suffix(path.suffix+'.partial')
                req = urllib.request.Request(item['url'], headers={'User-Agent': 'public-headhand-reproduction/1'})
                with urllib.request.urlopen(req, timeout=60) as src, partial.open('wb') as dst:
                    while chunk := src.read(1024*1024):
                        dst.write(chunk)
                actual = digest(partial)
                assert actual == item['sha256'], (item['name'], 'SHA mismatch', actual)
                if 'bytes' in item:
                    assert partial.stat().st_size == item['bytes']
                partial.replace(path)
            assert digest(path) == item['sha256']
            result = dict(item, path=str(path), complete=True, seconds=time.time()-started)
            (ROOT/'receipts'/f"download_{item['name']}.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(dict(name=item['name'], complete=True, bytes=path.stat().st_size)), flush=True)
            return result
        except Exception as error:
            print(json.dumps(dict(name=item['name'], attempt=attempt+1, error=str(error))), flush=True)
            if attempt == 2:
                raise


def main():
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(download, ASSETS))
    ess_dir = ROOT/'assets/ess_public'
    ess_dir.mkdir(exist_ok=True)
    with tarfile.open(ROOT/'assets'/ASSETS[1]['file']) as archive:
        archive.extractall(ess_dir, filter='data')
    # Source labels remain in the sealed raw archive. Inference reads only
    # images, timestamps, and physical calibration, never reference trajectories.
    clip_dir = ROOT/'data/clip-002102'
    clip_dir.mkdir(exist_ok=True)
    with tarfile.open(ROOT/'assets'/ASSETS[2]['file']) as archive:
        names = archive.getnames()
        for member in archive.getmembers():
            if member.isfile() and (member.name.endswith('.jpg') or member.name.endswith('.info.json')):
                archive.extract(member, clip_dir, filter='data')
            elif member.isfile() and member.name.endswith('.cameras.json'):
                cameras = json.load(archive.extractfile(member))
                sanitized = {key: {'calibration': item['calibration']} for key, item in cameras.items()}
                (clip_dir/member.name).write_text(json.dumps(sanitized))
    (ROOT/'receipts/bootstrap.json').write_text(json.dumps(dict(complete=True, downloads=records,
        data_source='HOT3D public clip2102 full stereo+RGB', reference_pose_removed=True,
        hand_labels_not_extracted=True, original_archive_members=len(names)), indent=2))
    print('BOOTSTRAP_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
