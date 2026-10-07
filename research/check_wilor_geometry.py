import json
import cv2
import numpy as np
from wilor_eval_common import RUN,MAPPING,from_json
samples=json.loads((RUN/'samples.json').read_text())
geo=np.load(RUN/'crop_geometry.npz')
assert json.loads((RUN/'crop_cache_done.json').read_text())['geometry_version']==2
assert geo['ray_reprojection_error'].max()<.01
assert np.allclose(np.linalg.det(geo['rotations']),1,atol=1e-5)
assert np.allclose(np.einsum('bji,bjk->bik',geo['rotations'],geo['rotations']),np.eye(3),atol=1e-5)
# Independent toolkit mapping, composed with WiLoR's OpenPose permutation.
from hand_tracking_toolkit.hand_models.mano_hand_model import MANO_TO_CANONICAL_LANDMARK_MAPPING
openpose=[0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]
assert [openpose[i] for i in MAPPING]==MANO_TO_CANONICAL_LANDMARK_MAPPING
# Numerical pixel-to-camera consistency for the actual crop center rays.
for i in np.linspace(0,len(samples)-1,40).astype(int):
    cam=from_json(samples[i]['camera']);box=np.array(samples[i]['box'])
    center=(box[:2]+box[2:])/2
    uv=cam.eye_to_window(geo['rotations'][i][:,2])
    assert np.linalg.norm(uv-center)<.01
# Pixel rotation agrees with the camera rotation, for all four options.
Q=np.array([[0,1,0],[-1,0,0],[0,0,1]])
im=np.zeros((256,256),np.uint8);im[120,170]=1
for k in range(4):
    y,x=np.argwhere(np.rot90(im,k)==1)[0]
    expected=np.linalg.matrix_power(Q,k)@np.array([170-127.5,120-127.5,1])
    assert np.allclose([x-127.5,y-127.5],expected[:2])
print('CANONICAL_MAPPING_CAMERA_RAYS_AND_ROTATION_CHECKS_PASSED')
