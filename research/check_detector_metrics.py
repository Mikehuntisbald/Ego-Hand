"""Sanity checks for matching/AP and threshold operating points."""
import numpy as np
from compare_detectors import curve, matches, operating

f = dict(hands=[dict(box=[0,0,10,10], visibility=.2, truncated=False)])
perfect = dict(boxes=[[0,0,10,10]], scores=[.9])
assert np.isclose(curve([f],[perfect])['ap'], 1)
empty = dict(boxes=[], scores=[])
assert curve([f],[empty])['ap'] == 0
dup = dict(boxes=[[0,0,10,10],[0,0,10,10]], scores=[.9,.8])
assert matches(f,dup,0)[1].tolist() == [0,-1]
assert operating([f],[dup],0)['precision'] == .5
wrong_first = dict(boxes=[[20,20,30,30],[0,0,10,10]],scores=[.95,.9])
assert np.isclose(curve([f],[wrong_first])['ap'], .5)
assert operating([f],[perfect],.95)['recall'] == 0
assert operating([f],[perfect],0)['groups']['severe_lt_0.25']['recall'] == 1
# No predictions must be a false negative; an empty-GT image still counts FPs.
g = dict(hands=[])
assert operating([f,g],[empty,perfect],0)['fp'] == 1
assert operating([f,g],[empty,perfect],0)['recall'] == 0
# Tied confidence scores cannot create an unrealizable intermediate threshold.
tied = dict(boxes=[[0,0,10,10],[20,20,30,30]],scores=[.9,.9])
c = curve([f],[tied])
assert len(c['scores']) == 1 and c['precision'][0] == .5
print('MATCHING_AP_OPERATING_POINT_CHECKS_PASSED')
