"""Bounded CPU image/video decoding. No full-episode GPU frame cache."""
from collections import OrderedDict
from pathlib import Path
import threading
from .errors import RuntimeFailure


class BoundedDecoder:
    def __init__(self, window_frames=4):
        self.limit = max(1, int(window_frames))
        self.cache, self.videos = OrderedDict(), {}
        self.lock = threading.RLock()
        self.peak_cached = 0
        self.backend_calls = {'opencv_image_cpu': 0, 'opencv_ffmpeg_video_cpu': 0}

    def read(self, image, trace=None):
        if not image.get('valid', True):
            raise RuntimeFailure('source_invalid', 'Requested source image is invalid')
        key = (image['path'], int(image['source_index']))
        import cv2
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            path = Path(image['path'])
            if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
                value = cv2.imread(str(path), cv2.IMREAD_COLOR)
                self.backend_calls['opencv_image_cpu'] += 1
            else:
                if str(path) not in self.videos:
                    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
                    if not cap.isOpened() or cap.getBackendName() != 'FFMPEG':
                        raise RuntimeFailure('video_decoder', 'A real FFmpeg CPU decoder is required')
                    self.videos[str(path)] = cap
                cap = self.videos[str(path)]
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != image['source_index']:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(image['source_index']))
                okay, value = cap.read()
                if not okay:
                    value = None
                self.backend_calls['opencv_ffmpeg_video_cpu'] += 1
            if value is None:
                raise RuntimeFailure('decode', 'Source image could not be decoded', path=str(path), index=image['source_index'])
            value.setflags(write=False)
            self.cache[key] = value
            while len(self.cache)>self.limit:
                self.cache.popitem(last=False)
            self.peak_cached = max(self.peak_cached, len(self.cache))
            return value

    def close(self):
        for cap in self.videos.values():
            cap.release()
        self.videos.clear()
        self.cache.clear()
