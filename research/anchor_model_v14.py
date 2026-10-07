from density_model_v13 import DensityTrajectoryHand3D

class AnchoredTrajectoryHand3D(DensityTrajectoryHand3D):
    def encode(self,b):
        # The denoising target/output anchor is b.xyz. All geometry observations
        # stay original; only the repeated reference pose changes to the anchor.
        original={**b,'xyz':b['observation_xyz'],'base':b['xyz'][:,8]}
        return super().encode(original)
