"""Real licensed MANO tests opt in with EGO_RUNTIME_TEST_MANO; no redistributable fake hand."""
import ast
import copy
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from ego_hand_runtime.config import DEFAULT
from ego_hand_runtime.io import atomic_json, seal, read_json
from ego_hand_runtime.geometry import project
from ego_hand_runtime.schema import Episode


MANO = os.environ.get('EGO_RUNTIME_TEST_MANO')


@unittest.skipUnless(MANO and Path(MANO).is_file(),'licensed MANO not provided')
class ActualMANO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ego_hand_runtime.native import NativeMano
        torch.set_num_threads(2)
        cls.codec = NativeMano(MANO)

    def fixture(self, folder):
        from test_contracts import manifest
        data = manifest(folder,5);atomic_json(folder/'episode.json',data)
        episode = Episode.load(folder/'episode.json');camera = episode.camera_for_role('rgb')
        state = np.zeros((3,51),np.float32);state[:,2]=.6;state[:,0]=[0,.001,.002]
        state[:,6:] = np.random.default_rng(2).normal(0,.05,(3,45))
        beta = np.zeros((3,10),np.float32);right = np.array([True]*3)
        xyz = self.codec(state,beta,right)
        head = [dict(timestamp_ns=int(t),valid=True,segment=0,world_from_device=np.eye(4).tolist()) for t in episode.source_clock_ns]
        obs = [dict(track_id='hand',timestamp_ns=int(episode.source_clock_ns[j*2]),camera_id='rgb',
                    right=True,owner='unknown',confidence=.9,state_camera51=state[j].tolist(),betas=beta[j].tolist(),
                    uv=project(xyz[j],camera).tolist(),point_valid=[True]*21,box_scale_px=40.,valid=True,
                    depth_valid=True,depth_m=.6,depth_weight=1.) for j in range(3)]
        return episode,head,obs

    def test_export_left_right_is_actual_FK_and_transl_is_not_wrist(self):
        state = np.zeros((2,51),np.float32);state[:,2]=.6;state[:,3:6]=[[.1,.2,.3],[-.2,.3,-.1]]
        state[:,6:] = np.random.default_rng(9).normal(0,.2,(2,45))
        beta = np.random.default_rng(4).normal(0,.2,(2,10)).astype(np.float32);right = np.array([True,False])
        buffers = {k:v.detach().clone() for k,v in self.codec.layer.named_buffers()}
        out = self.codec.export(state,beta,right)
        with torch.inference_mode():
            # Independent direct smplx forward from the EXPORTED parameters.
            from smplx.vertex_ids import vertex_ids
            raw = self.codec.layer(global_orient=torch.tensor(out['global_orient']),hand_pose=torch.tensor(out['hand_pose']),
                                   betas=torch.tensor(out['betas']),pose2rot=False)
            from ego_hand_runtime.native import OPENPOSE
            joints = torch.cat([raw.joints,raw.vertices[:,list(vertex_ids['mano'].values())]],1)[:,OPENPOSE].numpy()
            joints = (joints+out['transl'][:,None])*out['reflection'][:,None]
        np.testing.assert_allclose(joints,out['joints'],atol=1e-7)
        self.assertGreater(float(np.max(np.abs(out['transl']-state[:,:3]))),.01)
        for k,v in self.codec.layer.named_buffers():torch.testing.assert_close(v,buffers[k],atol=0,rtol=0)

    def test_projection_matches_frozen_source_function(self):
        from test_contracts import camera
        from ego_hand_runtime.schema import Camera
        from ego_hand_runtime.native import project_torch
        source = Path(__file__).resolve().parents[2]/'research/hand3d_rollout_v8.py'
        tree = ast.parse(source.read_text())
        fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='project_fisheye624')
        scope = {'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),scope)
        d = camera();d.update(model='fisheye624',parameters=[50,51,31,23,.1,-.02,.001,0,0,0,.001,.002,.001,0,-.001,0])
        c = Camera.from_dict(d);xyz = torch.tensor(np.random.default_rng(1).normal(0,.05,(2,21,3)),dtype=torch.float32);xyz[:,:,2]+=.7
        reference = scope['project_fisheye624'](xyz,torch.tensor([c.parameters]*2))
        torch.testing.assert_close(project_torch(xyz,c),reference,atol=0,rtol=0)

    @unittest.skipUnless(os.environ.get('EGO_RUNTIME_TEST_GPU')=='1' and torch.cuda.is_available(),'explicit CUDA test not enabled')
    def test_GPU_FK_survives_RF_high_precision_mode_and_restores_it(self):
        from ego_hand_runtime.native import NativeMano
        model=NativeMano(MANO,'cuda:0')
        state=np.random.default_rng(3).normal(0,.2,(2,51)).astype(np.float32);state[:,2]=.6
        beta=np.zeros((2,10),np.float32);right=np.array([True,False])
        old=torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision('high')
            result=model.export(state,beta,right)
            np.testing.assert_allclose(result['joints'],model(state,beta,right),atol=1e-6)
            self.assertEqual(torch.get_float32_matmul_precision(),'high')
        finally:torch.set_float32_matmul_precision(old)

    def test_zero_addition_BA_exact_and_single_frame_finite(self):
        from ego_hand_runtime.observations import route
        from ego_hand_runtime.ba import optimize
        with tempfile.TemporaryDirectory() as tmp:
            episode,head,obs = self.fixture(Path(tmp));seeds,routed,_ = route(obs,head,episode,100000000)
            cfg = copy.deepcopy(DEFAULT['ba']);cfg['iterations']=3
            a = optimize(seeds,routed,self.codec,cfg,disable_additions=True)
            b = optimize(seeds,routed,self.codec,cfg,disable_additions=False)
            np.testing.assert_array_equal(a['state'],b['state']);np.testing.assert_array_equal(a['joints'],b['joints'])
            out = optimize(seeds[:1],routed[:1],self.codec,cfg)
            self.assertTrue(np.isfinite(out['after']))
            self.assertLessEqual(a['after'],a['before']+1e-6)

    def test_route_rejects_owner_and_source_skew_without_seed(self):
        from ego_hand_runtime.observations import route
        with tempfile.TemporaryDirectory() as tmp:
            episode,head,obs = self.fixture(Path(tmp));bad = copy.deepcopy(obs[-1]);bad['track_id']='bad';bad['source_timestamp_ns']=bad['timestamp_ns']+10000000
            seeds,_,rejected = route(obs+[bad],head,episode,100000000)
            self.assertNotIn('bad',[x['track_id'] for x in seeds]);self.assertEqual(rejected[0]['reason'],'unsynchronized_source')

    def test_frozen_BA_objective_matches_extracted_reference(self):
        from ego_hand_runtime.observations import route
        from ego_hand_runtime.ba import optimize
        from ego_hand_runtime.schema import Camera
        with tempfile.TemporaryDirectory() as tmp:
            episode,head,obs = self.fixture(Path(tmp))
            c = episode.cameras['rgb'];c.model='fisheye624';c.parameters+= [0.]*12
            for row in obs:
                xyz=self.codec(np.array([row['state_camera51']]),np.array([row['betas']]),np.array([True]))[0]
                row['uv']=project(xyz,c).tolist()
            seeds,routed,_ = route(obs,head,episode,100000000)
            cfg=copy.deepcopy(DEFAULT['ba']);cfg['iterations']=1
            result=optimize(seeds,routed,self.codec,cfg,disable_additions=True)
            root=Path(__file__).resolve().parents[2]
            tree=ast.parse((root/'research/native_mano_h20_20261019/v2/src/native_ba.py').read_text())
            fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
            # Find by name rather than relying on module imports/environment.
            projection=next(n for n in ast.parse((root/'research/hand3d_rollout_v8.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='project_fisheye624')
            tensor=lambda x:torch.tensor(np.asarray(x),dtype=torch.float32)
            state=tensor([s['state'] for s in seeds]);beta=tensor([seeds[0]['betas']])
            stamps=[s['timestamp_ns'] for s in seeds];dt=[(stamps[1]-stamps[0])/1e9,(stamps[2]-stamps[1])/1e9]
            scope=dict(torch=torch,state=state,shape=beta,group_t=torch.tensor([0]*3),right_t=torch.tensor([True]*3),
                       decoder=self.codec.tensor,R=torch.eye(3)[None].repeat(3,1,1),T=torch.zeros(3,3),
                       params=tensor([c.parameters]*3),uv=tensor([r['uv'] for r in obs]),scale=tensor([40.]*3)[:,None,None],
                       anchor_t=tensor([.6]*3),weight_t=torch.ones(3),reference_state=state.clone(),beta0=beta.clone(),
                       pair=torch.tensor([[0,1],[1,2]]),triple=torch.tensor([[0,1,2]]),dp=tensor(dt)[:,None,None],
                       dt0=tensor([dt[0]])[:,None,None],dt1=tensor([dt[1]])[:,None,None])
            exec(compile(ast.Module(body=[projection,fn],type_ignores=[]),'<frozen reference>','exec'),scope)
            reference=float(scope['evaluate']()[0])
            self.assertAlmostEqual(result['before'],reference,places=6)

    def test_protection_gates_large_wrist_shift_and_preserves_FK(self):
        from ego_hand_runtime.quality import protect
        from ego_hand_runtime.observations import route
        with tempfile.TemporaryDirectory() as tmp:
            episode,head,obs = self.fixture(Path(tmp));seeds,routed,_ = route(obs,head,episode,100000000)
            state = np.array([s['state'] for s in seeds]);beta = np.array([s['betas'] for s in seeds]);right = np.array([True]*3)
            base = dict(state=state,betas=beta,right=right,joints=self.codec(state,beta,right))
            proposal = copy.deepcopy(base);proposal['state'][:,0]+=.2;proposal['joints']=self.codec(proposal['state'],beta,right)
            result,report = protect(base,proposal,seeds,routed,self.codec,DEFAULT['quality'])
            self.assertLess(report['tracks'][0]['alpha'],.1)
            self.assertLessEqual(report['tracks'][0]['rmse_m'],.005+1e-9)
            np.testing.assert_array_equal(result['joints'],self.codec(result['state'],result['betas'],right))
            same,report = protect(base,base,seeds,routed,self.codec,DEFAULT['quality']);self.assertIs(same,base)

    def test_explicit_invalid_source_is_not_interpolated_to_valid(self):
        from ego_hand_runtime.delivery import write
        from ego_hand_runtime.observations import route
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);episode,head,obs=self.fixture(folder);seeds,_,_=route(obs,head,episode,100000000)
            state=np.array([r['state'] for r in seeds]);beta=np.array([r['betas'] for r in seeds]);right=np.array([True]*3)
            output=write(episode,seeds,dict(state=state,betas=beta,right=right),head,self.codec,folder,DEFAULT,
                         {'hand':{int(episode.source_clock_ns[1])}})
            self.assertFalse(output['frames'][1]['hands'][0]['valid'])
            self.assertTrue(output['frames'][3]['hands'][0]['valid'])

    def test_sensor_clock_after_last_RGB_still_supports_interpolation(self):
        from ego_hand_runtime.delivery import write
        from ego_hand_runtime.observations import route
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);episode,head,obs=self.fixture(folder)
            # Tracking exposures are 100us later than RGB, including last frame.
            for h in head:h['timestamp_ns']+=100000
            seeds,_,_=route(obs,head,episode,100000000)
            state=np.array([r['state'] for r in seeds]);beta=np.array([r['betas'] for r in seeds]);right=np.array([True]*len(seeds))
            output=write(episode,seeds,dict(state=state,betas=beta,right=right),head,self.codec,folder,DEFAULT,
                         active_range=(int(episode.source_clock_ns[0]),int(episode.source_clock_ns[-1])))
            self.assertFalse(output['frames'][0]['head_valid'])
            self.assertTrue(output['frames'][-1]['head_valid']);self.assertTrue(output['frames'][-1]['hands'][0]['valid'])

    def test_new_motion_losses_are_finite_and_do_not_change_FK_buffers(self):
        from ego_hand_runtime.ba import optimize
        from ego_hand_runtime.observations import route
        with tempfile.TemporaryDirectory() as tmp:
            episode,head,obs=self.fixture(Path(tmp));seeds,routed,_=route(obs,head,episode,100000000)
            cfg=copy.deepcopy(DEFAULT['ba']);cfg.update(iterations=2,world_root_so3_weight=.01,wrist_motion_weight=.01,
                                                       observation_preservation_weight=.02,mask_outside_image=True)
            result=optimize(seeds,routed,self.codec,cfg)
            self.assertTrue(np.isfinite(result['state']).all());self.assertLessEqual(result['after'],result['before']+1e-6)

    def test_two_camera_raw_observations_share_one_native_state(self):
        from ego_hand_runtime.observations import route
        from ego_hand_runtime.ba import optimize
        from ego_hand_runtime.schema import Camera
        from test_contracts import camera
        with tempfile.TemporaryDirectory() as tmp:
            episode,head,obs=self.fixture(Path(tmp))
            spec=camera('side_left','other',.1);spec.update(model='kb4',parameters=[50,50,31.5,23.5,.01,0,0,0],photometric='rgb')
            second=Camera.from_dict(spec);episode.cameras['other']=second
            extra=[]
            for row in obs:
                s=np.array(row['state_camera51'],np.float32);s[0]-=.1
                xyz=self.codec(s[None],np.array([row['betas']]),np.array([True]))[0]
                extra.append(dict(row,camera_id='other',state_camera51=s.tolist(),uv=project(xyz,second).tolist()))
            seeds,routed,rejected=route(obs+extra,head,episode,100000000)
            self.assertEqual(len(seeds),3);self.assertEqual(len(routed),6);self.assertFalse(rejected)
            self.assertEqual(sorted(r['state_index'] for r in routed),[0,0,1,1,2,2])
            cfg=copy.deepcopy(DEFAULT['ba']);cfg['iterations']=2
            result=optimize(seeds,routed,self.codec,cfg)
            self.assertTrue(np.isfinite(result['state']).all());self.assertLessEqual(result['after'],result['before']+1e-6)

    def test_full_cli_pipeline_seal_resume_and_pool_reuse(self):
        from ego_hand_runtime.pipeline import infer, RuntimeContext
        from ego_hand_runtime.delivery import verify
        from ego_hand_runtime.io import sha256
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp);episode,head,obs = self.fixture(folder)
            replay = folder/'replay';replay.mkdir()
            atomic_json(replay/'head.json',head);atomic_json(replay/'observations.json',obs)
            atomic_json(replay/'provenance.json',dict(kind='synthetic_contract_fixture',purpose='not accuracy or live GPU evidence'))
            seal(replay,['head.json','observations.json','provenance.json'],binding='fixture',backend='test')
            cfg = copy.deepcopy(DEFAULT);cfg.update(backend='replay',device='cpu')
            cfg['replay']['bundle']=str(replay);cfg['assets']['mano_right']=dict(path=MANO,sha256=sha256(MANO))
            cfg['ba']['iterations']=2;cfg['resources'].update(ram_admit_gib=1000,ram_resume_gib=900,ram_hard_gib=1100)
            context = RuntimeContext(cfg)
            try:
                for i in range(2):
                    result = infer(episode.source_path,cfg,folder/str(i),context=context)
                    self.assertTrue(result['program_success']);self.assertFalse(result['quality_admitted'])
                    self.assertEqual(verify(folder/str(i))['frames'],5)
                self.assertEqual(context.pools['ba'].loads,1)
                before = sha256(folder/'0/bundle.h5')
                resumed = infer(episode.source_path,cfg,folder/'0',resume=True,context=context)
                self.assertTrue(resumed['resumed_complete']);self.assertEqual(before,sha256(folder/'0/bundle.h5'))
                changed = copy.deepcopy(cfg);changed['ba']['iterations']=3
                with self.assertRaises(Exception):infer(episode.source_path,changed,folder/'0',resume=True,context=context)
                (folder/'0/bundle.h5').write_bytes(b'corrupt')
                with self.assertRaises(Exception):infer(episode.source_path,cfg,folder/'0',resume=True,context=context)
            finally:context.close()


if __name__=='__main__':unittest.main()
