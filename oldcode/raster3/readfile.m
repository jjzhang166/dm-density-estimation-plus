
filename = 'i:\results\snap064_red_1024.dens';
%filename = '~/data/run_200'

fid = fopen(filename, 'r');
gridsize = fread(fid, 1, 'int32');
numofcuts = fread(fid, 1, 'int32');
boxsize = fread(fid, 1, 'float32');
startz = fread(fid, 1, 'float32');
dz = fread(fid, 1, 'float32');
head = fread(fid, 59, 'int32');

numofcuts = 256
data = fread(fid, gridsize * gridsize * numofcuts, 'float32');