import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np

from ego_hand_runtime.errors import RuntimeFailure
from ego_hand_runtime.io import atomic_json, seal, verify_seal, read_json, sha256
from ego_hand_runtime.schema import Episode, Camera, ns
from ego_hand_runtime.geometry import project, unproject, rotvec_matrix, rectified_pair
from ego_hand_runtime.clock import sample_indices, motion_pairs, interpolate_native
from ego_hand_runtime.config import load_config, DEFAULT
from ego_hand_runtime.environment import child_environment, environment_delta
from ego_hand_runtime.timing import Trace
from ego_hand_runtime.pool import ModelPool, Admission


def camera(role='rgb', key='rgb', offset=0.):
    T = np.eye(4);T[0,3] = offset
    return dict(id=key,role=role,model='pinhole',width=64,height=48,
                parameters=[50,50,31.5,23.5],T_device_from_camera=T.tolist(),
                photometric='rgb' if role=='rgb' else 'mono',input_validated=False)


def manifest(folder, frames=5):
    stamps = [10**18+i*16666667 for i in range(frames)]
    return dict(schema=1,units='m',episode_id='fixture',source_id='contract-fixture',adapter='contract',
                cameras=[camera()],source_clock_ns=stamps,
                frames=[dict(timestamp_ns=t,source_index=i,images={}) for i,t in enumerate(stamps)])


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.root = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def episode(self, data=None):
        atomic_json(self.root/'episode.json',data or manifest(self.root))
        return Episode.load(self.root/'episode.json')
    def test_integer_epoch_and_sampling(self):
        t = [10**18,10**18+1,10**18+2]
        self.assertEqual(ns(t[-1]),t[-1])
        np.testing.assert_array_equal(sample_indices(t),[0,1,2])
        for bad in [True,1.2,float(10**18),-1,2**63]:
            with self.assertRaises(RuntimeFailure):ns(bad)
    def test_duplicate_clock_and_source_index(self):
        for field in ['timestamp','index']:
            d = manifest(self.root)
            if field=='timestamp':d['source_clock_ns'][1] = d['source_clock_ns'][0]
            else:d['frames'][0]['source_index'] = True
            with self.assertRaises(RuntimeFailure):self.episode(d)
    def test_missing_cameras_no_fake_capability(self):
        e = self.episode()
        self.assertFalse(e.capabilities()['stereo_vo'])
        d = manifest(self.root);d['cameras'] += [camera('side_left','side')]
        d['cameras'][1]['input_validated'] = True
        self.assertFalse(self.episode(d).capabilities()['side_rgb_recovery'])
    def test_inference_rejects_reference_pose_and_labels(self):
        for key in ['gt','labels','T_world_from_camera','qa_score']:
            d = manifest(self.root);d['provenance'] = {key:[]}
            with self.assertRaises(RuntimeFailure):self.episode(d)
    def test_per_frame_calibration_is_routed(self):
        d = manifest(self.root);d['frames'][2]['cameras'] = [camera(offset=.001)]
        e = self.episode(d)
        self.assertEqual(e.camera_at('rgb',int(e.source_clock_ns[2])).T_device_from_camera[0,3],.001)
    def test_camera_projection_roundtrip_and_baseline(self):
        c = Camera.from_dict(camera())
        points = np.array([[.01,.02,.6],[-.1,.03,.8]])
        uv = project(points,c);rays = unproject(uv,c)
        np.testing.assert_allclose(rays/rays[:,2:],points/points[:,2:],atol=1e-6)
        right = Camera.from_dict(camera('tracking_right','right',.1))
        pair = rectified_pair(c,right,64,48,50.)
        self.assertAlmostEqual(pair['baseline_m'],.1)
        # A 1m point with 50px focal and 0.1m baseline has 5px disparity.
        self.assertAlmostEqual(50*pair['baseline_m']/1.,5.)
    def test_fisheye_distortion_roundtrip(self):
        d = camera();d.update(model='fisheye624',parameters=[50,51,31,23,.1,-.02,.001,0,0,0,.001,.002,.001,0,-.001,0])
        c = Camera.from_dict(d)
        xyz = np.array([[.1,.1,.5],[-.15,.05,.6],[0,0,.8]])
        rays = unproject(project(xyz,c),c)
        np.testing.assert_allclose(rays/rays[:,2:],xyz/xyz[:,2:],atol=2e-5)
    def test_dynamic_lens_maps_preserve_fixed_virtual_geometry(self):
        from ego_hand_runtime.geometry import virtual_camera_map
        left=Camera.from_dict(camera());right=Camera.from_dict(camera('tracking_right','right',.1))
        spec=rectified_pair(left,right,64,48,50.)
        uv,_=virtual_camera_map(left,spec['T_device_from_rect'][:3,:3],64,48,50.,spec['principal'])
        np.testing.assert_array_equal(uv,spec['maps'][0])
        changed=Camera.from_dict(camera());changed.parameters[0]=55.
        mapped,_=virtual_camera_map(changed,spec['T_device_from_rect'][:3,:3],64,48,50.,spec['principal'])
        # The same virtual ray now samples the new physical lens coordinates.
        self.assertAlmostEqual(float(mapped[24,40,0]),(40-31.5)*55/50+31.5,places=5)
    def test_motion_pairs_no_identity_reset_gap_crossing(self):
        pairs,triples = motion_pairs([0,10,20,30,1000],['a']*5,[0,0,1,1,1],[True]*5,100)
        np.testing.assert_array_equal(pairs,[[0,1],[2,3]])
        self.assertEqual(len(triples),0)
    def test_parameter_interpolation_not_joint_interpolation(self):
        state = np.zeros((2,51),np.float32);state[:,2]=.5;state[1,5]=np.pi/2
        def fk(s,b,r):
            xyz = np.zeros((len(s),21,3),np.float32)
            xyz[:,0] = s[:,:3]
            xyz[:,1] = s[:,:3]+np.einsum('nij,j->ni',rotvec_matrix(s[:,3:6]),[.1,0,0])
            return xyz
        rows = interpolate_native([0,100],state,np.zeros((2,10)),[True]*2,[True]*2,[0]*2,[0,50,100,101],fk,100)
        self.assertTrue(rows[1]['interpolated']);self.assertFalse(rows[3]['valid'])
        np.testing.assert_allclose(rows[1]['joints'][1,:2],[np.sqrt(.005),np.sqrt(.005)],atol=1e-6)
        self.assertGreater(rows[1]['joints'][1,0],.05)
        rows = interpolate_native([0,100],state,np.zeros((2,10)),[True]*2,[True]*2,[0,1],[50],fk,100)
        self.assertFalse(rows[0]['valid'])
    def test_exact_k8s_cleanup_and_explicit_override(self):
        env = {'A_SERVICE_HOST':'10.0.0.1','A_SERVICE_PORT':'80','A_PORT':'tcp://10.0.0.1:80',
               'A_PORT_80_TCP':'tcp://10.0.0.1:80','A_PORT_80_TCP_PROTO':'tcp',
               'A_PORT_80_TCP_PORT':'80','A_PORT_80_TCP_ADDR':'10.0.0.1',
               'B_SERVICE_HOST':'partial','HTTPS_PROXY':'http://proxy','API_KEY':'redacted',
               'HAMR4D_SHARED_VIDEO_DECODER':'nvdec','A_PORT_81_TCP':'incomplete'}
        result = child_environment({'HAMR4D_SHARED_VIDEO_DECODER':'cpu'},env)
        self.assertNotIn('A_SERVICE_HOST',result)
        self.assertEqual(result['B_SERVICE_HOST'],'partial')
        self.assertEqual(result['A_PORT_81_TCP'],'incomplete')
        self.assertEqual(result['HTTPS_PROXY'],'http://proxy')
        self.assertEqual(result['HAMR4D_SHARED_VIDEO_DECODER'],'cpu')
    def test_incremental_env_restore(self):
        base = {f'X_{i}':str(i) for i in range(10000)}
        with patch.dict(os.environ,base,clear=True):
            with environment_delta({'X_1':'changed','NEW':'x'},remove=['X_2']) as count:
                self.assertEqual(count,3);self.assertNotIn('X_2',os.environ)
            self.assertEqual(dict(os.environ),base)
    def test_pool_reuse_and_exclusive_lease(self):
        created = []
        def factory():
            created.append(1);return object()
        pool = ModelPool('real-counter',factory,1);trace = Trace(request_id='test')
        first = None
        for _ in range(3):
            with pool.acquire(trace) as model:
                first = model if first is None else first
                self.assertIs(first,model)
        self.assertEqual(len(created),1);self.assertEqual(pool.loads,1)
        pool.close()
    def test_admission_hysteresis_no_process_mutation(self):
        cfg = copy.deepcopy(DEFAULT);cfg['device']='cpu'
        used = [13.,11.,9.]
        gate = Admission(cfg,lambda:dict(ram_used_gib=used.pop(0)))
        self.assertFalse(gate.check()[0]);self.assertFalse(gate.check()[0]);self.assertTrue(gate.check()[0])
    def test_seal_corruption_and_resume_binding(self):
        atomic_json(self.root/'x.json',dict(a=1))
        seal(self.root,['x.json'],binding='same',backend='test')
        with self.assertRaises(RuntimeFailure):verify_seal(self.root,'different')
        atomic_json(self.root/'x.json',dict(a=2))
        with self.assertRaises(RuntimeFailure):verify_seal(self.root)
    def test_scorer_does_not_open_labels_before_seal(self):
        from ego_hand_runtime.scoring import score
        with patch('ego_hand_runtime.scoring.read_json',side_effect=AssertionError('labels must stay closed')):
            with self.assertRaises(FileNotFoundError):score(self.root,self.root/'labels.json',self.root/'score.json')
    def test_live_output_lock_keeps_current_process_alive(self):
        from ego_hand_runtime.pipeline import output_lock,pid_alive
        self.assertTrue(pid_alive(os.getpid()))
        with output_lock(self.root):
            with self.assertRaises(RuntimeFailure):
                with output_lock(self.root):pass
        self.assertTrue(read_json(self.root/'run.lock')['released'])
    def test_reused_PID_can_resume_a_stale_output_lock(self):
        from ego_hand_runtime.pipeline import output_lock,process_identity
        self.assertIsNotNone(process_identity(os.getpid()))
        old=process_identity(os.getpid());old=dict(old)
        key='start_ticks' if 'start_ticks' in old else 'start_filetime';old[key]=-1
        atomic_json(self.root/'run.lock',dict(pid=os.getpid(),process_identity=old))
        with output_lock(self.root):pass
        self.assertEqual(len(list(self.root.glob('stale-lock-*.json'))),1)
    def test_environment_is_applied_before_native_import_boundary(self):
        from ego_hand_runtime.environment import reexec_for_config
        with patch.dict(os.environ,{'API_KEY':'kept','HAMR4D_OLD':'bad'},clear=True),patch('os.execve') as execute:
            reexec_for_config({'LD_LIBRARY_PATH':'/pinned/lib'},['python','job'])
            effective=execute.call_args.args[2]
            self.assertEqual(effective['API_KEY'],'kept');self.assertNotIn('HAMR4D_OLD',effective)
            self.assertEqual(effective['LD_LIBRARY_PATH'],'/pinned/lib')
    def test_lfs_is_never_a_model(self):
        from ego_hand_runtime.assets import inspect_asset
        p = self.root/'model.pth';p.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n')
        a = inspect_asset('x',dict(path=str(p),sha256=sha256(p)))
        self.assertFalse(a['valid']);self.assertTrue(a['LFS_pointer'])
    def test_asset_digest_cache_rechecks_changed_bytes(self):
        from ego_hand_runtime.assets import inspect_asset
        p=self.root/'weight';p.write_bytes(b'abcd');spec=dict(path=str(p),sha256=sha256(p))
        self.assertTrue(inspect_asset('weight',spec)['valid'])
        self.assertEqual(inspect_asset('weight',spec)['hash_verification'],'stat_guarded_process_cache')
        p.write_bytes(b'ABCDE')
        self.assertFalse(inspect_asset('weight',spec)['valid'])
    def test_nested_cpu_spans_are_not_additive(self):
        trace = Trace(request_id='abc')
        with trace.span('outer'):
            with trace.span('inner'):pass
        trace.save(self.root/'trace.json');data = read_json(self.root/'trace.json')
        self.assertFalse(data['cpu_spans_additive']);self.assertEqual(data['events'][0]['parent_id'],data['events'][1]['span_id'])
    def test_bounded_decoder_real_pixels(self):
        import cv2
        from ego_hand_runtime.decode import BoundedDecoder
        decoder = BoundedDecoder(2)
        for i in range(5):
            p = self.root/(str(i)+'.png');cv2.imwrite(str(p),np.full((6,8,3),i,np.uint8))
            image = decoder.read(dict(path=str(p),source_index=i,valid=True))
            self.assertEqual(int(image[0,0,0]),i)
        self.assertEqual(decoder.peak_cached,2);decoder.close()
    def test_dry_run_no_model_or_output_side_effect(self):
        from ego_hand_runtime.pipeline import infer
        episode = self.episode();config = copy.deepcopy(DEFAULT);config['device']='cpu'
        with patch('ego_hand_runtime.pipeline.RuntimeContext',side_effect=AssertionError('must not load models')):
            result = infer(episode.source_path,config,self.root/'output',dry_run=True)
        self.assertFalse(result['execution_ready']);self.assertFalse((self.root/'output').exists())


if __name__=='__main__':unittest.main()
