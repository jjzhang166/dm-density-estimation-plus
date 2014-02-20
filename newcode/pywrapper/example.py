import numpy as np
import matplotlib.pyplot as plt

from GADGETPy import *
from cicpy import *


# #Test the Gsnap and CIC python wrapper


#read GADGET data from a GADGET snapshot
snap=GSnap('/Users/lyang/data/32Mpc_S1_PM_000')


#print out the infomation of the Snapshot
head=snap.GetHeader()
print head.BoxSize
npart = getUInt32Array(head.npart, 6)[1]
print npart
scalefact = head.time
print scalefact

gridsize = 128
densgridsize = 128


#get the position array and velocity array
pos = np.array(snap.GetBlock("POS ", npart, 0, 0)).reshape([gridsize**3 * 3, 1]);
vel = np.array(snap.GetBlock("VEL ", npart, 0, 0)).reshape([gridsize**3 * 3, 1]);

#convert velocity to km/s
vel=vel.astype(np.float)*np.sqrt(scalefact)

#get the pointer of the velocity array and position array
print pos[10 * 3 + np.array([0, 1, 2])]
print vel[10 * 3 + np.array([0, 1, 2])]
posvec = vectord(pos.T[0])
velvec = vectord(vel.T[0])
posPt = getDoublePointer(posvec)
velPt = getDoublePointer(velvec)


#create a CIC object
cic = CIC(head.BoxSize, densgridsize, True) 

#render the particles in a CIC grids
cic.render_particle(posPt, velPt, npart, 1.0)


#get the result out
densArray = np.array(getDoubleArray(cic.getDensityField(), densgridsize**3)).reshape([densgridsize,densgridsize,densgridsize])
velXArray = np.array(getDoubleArray(cic.getVelocityXField(), densgridsize**3)).reshape([densgridsize,densgridsize,densgridsize])
velYArray = np.array(getDoubleArray(cic.getVelocityYField(), densgridsize**3)).reshape([densgridsize,densgridsize,densgridsize])
velZArray = np.array(getDoubleArray(cic.getVelocityZField(), densgridsize**3)).reshape([densgridsize,densgridsize,densgridsize])


#show the CIC result
plt.imshow(densArray[:,:,30])
plt.show()

plt.imshow(velXArray[:,:,30])
plt.show()

plt.imshow(velYArray[:,:,30])
plt.show()

plt.imshow(velZArray[:,:,30])
plt.show()

#Example for reading LTFE data:
import ltfepy
reader = ltfepy.LTFEReader('/Users/lyang/Documents/Projects/dm-density-estimation-plus/newcode/test/try.dens');
header = reader.getHeader();
data = np.array(reader.getDataVec());
data = data.reshape([header.xyGridSize, header.xyGridSize, header.zGridSize]);
plt.imshow(data[:,:,10])




