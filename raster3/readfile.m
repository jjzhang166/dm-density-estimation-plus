filename = 'try.data';
%filename = '~/data/run_200'

fid = fopen(filename, 'r');


bytesleft = (256-6*4 - 6*8 - 8 - 8 - 2*4-6*4 -4 -4 -4*8)/4

dummy = fread(fid, 1, 'uint32')
narr = fread(fid, 6, 'int32')   % 4 * 6 = 24
marr = fread(fid, 6, 'float64') % 8 * 6 = 48
time = fread(fid, 1, 'float64') % 8
redshift = fread(fid, 1, 'float64') %8
flag_sfr = fread(fid, 1, 'int32')% 4
flag_feedback = fread(fid, 1, 'int32')%4
npartTotal = fread(fid, 6, 'int32') % 4 * 6 = 24
FlagCooling = fread(fid, 1, 'int32')% 4
NumFiles = fread(fid, 1, 'int32')% 4
Boxsize = fread(fid, 1, 'float64')% 8
Omega0 = fread(fid, 1, 'float64')% 8
OmegaLambda = fread(fid, 1, 'float64') % 8
HubbleParam = fread(fid, 1, 'float64') % 8
la = fread(fid, bytesleft, 'int32')    
dummy = fread(fid, 1, 'uint32')

dummy = fread(fid, 1, 'uint32')
px = fread(fid, 3, 'float32') 

posx(1)
posy(1)
posz(1)


fclose(fid)