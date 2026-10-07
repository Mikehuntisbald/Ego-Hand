import tempfile,json
from pathlib import Path
import spatial_rgb_common as s
import cv2,numpy as np
s.RUN.mkdir(exist_ok=True)
records,index=s.records_and_index();errors=[]
with tempfile.TemporaryDirectory(dir=s.RUN) as tmp:
    path=Path(tmp)/'constant.png';cv2.imwrite(str(path),np.full((1408,1408,3),231,np.uint8))
    for i in np.linspace(0,len(records)-1,20).round().astype(int):
        r=records[i];record={k:r[k] for k in ['image','camera','clip']}
        roi=index['roi'][i+1].numpy()*1408;rects=[index['rectangles'][i+1,5].numpy()]
        a,ap,*_=s.prepare(record,roi,rects);b,bp,*_=s.prepare(dict(record,image=str(path)),roi,rects)
        error=int(np.abs(a[0].astype(int)-b[0].astype(int)).max());errors.append(error)
        assert np.array_equal(ap,bp)
    assert max(errors)==0,errors
s.save(s.RUN/'pixel_checks.json',dict(full_roi_mask_independent_of_source_pixels=True,frames_checked=len(errors),max_pixel_difference=max(errors),bilinear_support_covered=True))
print(json.dumps(dict(frames=len(errors),max_pixel_difference=max(errors))))
