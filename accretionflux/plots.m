%data = artdata
data = realdata

figure

plot(data(:,1),(data(:,2)), '-*k', ...
    data(:,1),(data(:,3)), ...
    data(:,1),(data(:,4)), ...
    data(:,1),(data(:,5)), ...
    data(:,1),(data(:,6)), ...
    data(:,1),(data(:,7)), ...
    data(:,1),(data(:,8)), ...
    data(:,1),(data(:,9)))

axis([100 7000 -4000 100]);
ylabel('Accretion Rate(M_{sun}/s)')
xlabel('Radius(Code units)')

legend('LTFE', 'dr = 10', 'dr = 30', ...
    'dr = 50', 'dr = 70', 'dr = 80', 'dr = 90', 'dr = 100');

data = artdata
title('Real Data')

figure

plot(data(:,1),(data(:,2)), '-*k', ...
    data(:,1),(data(:,3)), '-rs',...
    data(:,1),(data(:,4)), ':',...
    data(:,1),(data(:,5)), ':',...
    data(:,1),(data(:,6)), ':',...
    data(:,1),(data(:,7)), ':',...
    data(:,1),(data(:,8)), ':',...
    data(:,1),(data(:,9)),':')

axis([100 15000 0 1e7]);
ylabel('Accretion Rate(M_{sun}/s)')
xlabel('Radius(Code units)')
title('Artificial Data')


legend('Accurate','LTFE', 'dr = 10', 'dr = 30', ...
    'dr = 50', 'dr = 70', 'dr = 80', 'dr = 90');
