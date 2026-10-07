import argparse,sys
from hand3d_v8_common import V7
import train_density_risk_v13 as worker
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['control','consensus'],required=True);p.add_argument('--device',required=True);a=p.parse_args()
worker.ROOT=V7.parent/'side_data_v16'/a.variant
sys.argv=['train_density_risk_v13.py','--arm','dense','--device',a.device]
worker.main()
