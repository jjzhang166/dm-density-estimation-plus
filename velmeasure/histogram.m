%measure histogram
rall = sqrt(data(:, 1) .^ 2 + data(:, 2) .^ 2 + data(:, 3) .^ 2);
rcos = data(:, 3) ./ rall;
ind = find(rall > 950 & rall < 1500);
nind = find(rcos(ind) > 0.8 );
vel0 = data(ind, 6);
vel = vel0(nind);