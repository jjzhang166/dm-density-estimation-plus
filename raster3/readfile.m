<<<<<<< HEAD
filename = 'i:\sandbox\try_spike_n256_s59_008.ltfe';
=======
filename = 'try.stream';
>>>>>>> 63ce8a0041fb59a6b0ce4cc8150d52c170bf320b
%filename = '~/data/run_200'

fid = fopen(filename, 'r');
gridsize = fread(fid, 1, 'int32');
numofcuts = fread(fid, 1, 'int32');
boxsize = fread(fid, 1, 'float32');
startz = fread(fid, 1, 'float32');
dz = fread(fid, 1, 'float32');
head = fread(fid, 59, 'int32');

<<<<<<< HEAD
data = fread(fid, gridsize * gridsize * numofcuts, 'float32');
=======
data = fread(fid, gridsize * gridsize * numofcuts, 'int32');
>>>>>>> 63ce8a0041fb59a6b0ce4cc8150d52c170bf320b
