# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
#import matplotlib.pyplot as plt
#%matplotlib inline
import matplotlib.pyplot as plt

# <codecell>

from GADGETPy import *
from cicpy import *

# <markdowncell>

# #Test the Gsnap and CIC python wrapper

# <codecell>

snap=GSnap('/Users/lyang/data/32Mpc_S1_PM_000')

# <codecell>


head=snap.GetHeader()
print head.BoxSize
npart = getUInt32Array(head.npart, 6)[1]
print npart
scalefact = head.time
print scalefact

# <codecell>

gridsize = 128
densgridsize = 128

# <codecell>

pos = np.array(snap.GetBlock("POS ", npart, 0, 0)).reshape([gridsize**3 * 3, 1]);

# <codecell>

vel = np.array(snap.GetBlock("VEL ", npart, 0, 0)).reshape([gridsize**3 * 3, 1]);
#convert to km/s
vel=vel.astype(np.float)*np.sqrt(scalefact)

# <codecell>

print pos[10 * 3 + np.array([0, 1, 2])]
print vel[10 * 3 + np.array([0, 1, 2])]
posvec = vectord(pos.T[0])
velvec = vectord(vel.T[0])
posPt = getDoublePointer(posvec)
velPt = getDoublePointer(velvec)

# <codecell>

cic = CIC(head.BoxSize, densgridsize, True) 

# <codecell>

cic.render_particle(posPt, velPt, npart, 1.0)

# <codecell>

densArray = np.array(getDoubleArray(cic.getDensityField(), densgridsize**3)).reshape([densgridsize,densgridsize,densgridsize])
velXArray = np.array(getDoubleArray(cic.getVelocityXField(), densgridsize**3)).reshape([densgridsize,densgridsize,densgridsize])
velYArray = np.array(getDoubleArray(cic.getVelocityYField(), densgridsize**3)).reshape([densgridsize,densgridsize,densgridsize])
velZArray = np.array(getDoubleArray(cic.getVelocityZField(), densgridsize**3)).reshape([densgridsize,densgridsize,densgridsize])

# <codecell>

plt.imshow(densArray[:,:,30])
plt.show()

# <codecell>

plt.imshow(velXArray[:,:,30])
plt.show()

# <codecell>

plt.imshow(velYArray[:,:,30])
plt.show()

# <codecell>

plt.imshow(velZArray[:,:,30])
plt.show()




