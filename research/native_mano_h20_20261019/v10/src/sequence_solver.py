"""Exact second-order offline candidate selection in actual physical time."""
import itertools
import numpy as np
SPEED=.002
ACCEL=.02
def robust(x):return np.where(np.abs(x)<1,.5*x*x,np.abs(x)-.5).mean(axis=(-1,-2))
def energies(points,times):
    dt=np.diff(np.asarray(times,np.int64))/1e9;assert np.all((dt>0)&(dt<=.1))
    v=(points[1:,None]-points[:-1,:,None])/(60*dt[:,None,None,None,None]);vcost=robust(v/.01)
    ac=[]
    for t in range(1,len(v)):
        a=(v[t][None]-v[t-1][:,:,None])/(60*(dt[t]+dt[t-1])/2);ac.append(robust(a/.005))
    return vcost,np.asarray(ac)
def solve(unary,points,times,with_motion=True):
    N,K=unary.shape;assert len(points)==N and np.isfinite(unary).all() and np.isfinite(points).all()
    if N==1 or not with_motion:return unary.argmin(1),float(unary.min(1).sum())
    vc,ac=energies(points,times);dp=unary[0,:,None]+unary[1,None,:]+SPEED*vc[0];back=[]
    for t in range(2,N):
        value=dp[:,:,None]+unary[t,None,None,:]+SPEED*vc[t-1][None]+ACCEL*ac[t-2];back.append(value.argmin(0));dp=value.min(0)
    result=np.empty(N,np.int64);result[-2:]=np.unravel_index(dp.argmin(),dp.shape)
    for t in range(N-1,1,-1):result[t-2]=back[t-2][result[t-1],result[t]]
    return result,float(dp.min())
def exhaustive(unary,points,times):
    vc,ac=energies(points,times);best=None
    for path in itertools.product(range(unary.shape[1]),repeat=len(unary)):
        cost=sum(unary[t,k] for t,k in enumerate(path))+SPEED*sum(vc[t,path[t],path[t+1]] for t in range(len(path)-1))+ACCEL*sum(ac[t,path[t],path[t+1],path[t+2]] for t in range(len(path)-2))
        if best is None or cost<best[0]:best=(float(cost),np.array(path))
    return best[1],best[0]
def segments(track,segment,side,times):
    out=[]
    for tid in sorted(set(track)):
        ids=np.flatnonzero(track==tid);ids=ids[np.argsort(times[ids])];start=0
        for j in range(1,len(ids)):
            a,b=ids[j-1:j+1];delta=int(times[b])-int(times[a])
            if not(0<delta<=100_000_000) or segment[a]!=segment[b] or side[a]!=side[b]:out.append(ids[start:j]);start=j
        if len(ids):out.append(ids[start:])
    return out
def select_all(unary,points,times,track,segment,side,with_motion=True):
    result=np.empty(len(unary),np.int64);records=[]
    for ids in segments(track,segment,side,times):
        path,cost=solve(unary[ids],points[ids],times[ids],with_motion);result[ids]=path;records.append(dict(first=int(ids[0]),last=int(ids[-1]),count=len(ids),track=int(track[ids[0]]),segment=int(segment[ids[0]]),right=bool(side[ids[0]]),cost=cost,candidate_switches=int((path[1:]!=path[:-1]).sum())))
    return result,records
