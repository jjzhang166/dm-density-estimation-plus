function [data] = readpvm(filename)
%% Read the PVM (pairwise velocity measurement) data from filename 
%%
fid = fopen(filename, 'r');
pairsize = fread(fid, 1, 'uint64');
d0 = fread(fid, pairsize * 6, 'float32');
fclose(fid);
data = reshape(d0, 6, pairsize)';

end

