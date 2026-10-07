"""Reuse the audited observation path with the separately frozen v9 clip IDs."""
from hand3d_trajectory_data_v9 import RUN
import prepare_fresh_hand3d_v8 as worker
worker.RUN=RUN
if __name__=='__main__':worker.prepare()
