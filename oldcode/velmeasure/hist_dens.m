%%measure velocity disburtion from density weighted
densfile = '32Mpc_S1.dens';
velxfile = '32Mpc_S1.velx';
velyfile = '32Mpc_S1.vely';
velzfile = '32Mpc_S1.velz';

boxSize = 32000;    %kpc
radius = 5000;
gridsize = 256;

%open the files
fid = fopen(densfile, 'rb');
head = fread(fid, 64, 'int32');
dens = reshape(fread(fid, gridsize^3, 'float32'), gridsize, gridsize, gridsize);
fclose(fid);

fid = fopen(velxfile, 'rb');
head = fread(fid, 64, 'int32');
velx = reshape(fread(fid, gridsize^3, 'float32'), gridsize, gridsize, gridsize);
fclose(fid);

fid = fopen(velyfile, 'rb');
head = fread(fid, 64, 'int32');
vely = reshape(fread(fid, gridsize^3, 'float32'), gridsize, gridsize, gridsize);
fclose(fid);

fid = fopen(velzfile, 'rb');
head = fread(fid, 64, 'int32');
velz = reshape(fread(fid, gridsize^3, 'float32'), gridsize, gridsize, gridsize);
fclose(fid);


