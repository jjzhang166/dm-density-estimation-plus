
filename = 'i:\sandbox\try_spike_n256_s59_008.ltfe';
%filename = '~/data/run_200'

fid = fopen(filename, 'r');
gridsize = fread(fid, 1, 'int32');
numofcuts = fread(fid, 1, 'int32');
boxsize = fread(fid, 1, 'float32');
startz = fread(fid, 1, 'float32');
dz = fread(fid, 1, 'float32');
head = fread(fid, 59, 'int32');

data = fread(fid, gridsize * gridsize * numofcuts, 'float32');