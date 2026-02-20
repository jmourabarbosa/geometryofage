% Check neuron distributions.
% Plot with violinplot
% Junda Zhu, 20231127
clearvars
load('odr_data_both_sig_20231017.mat');
real_age = neuron_info.Neuron_age + neuron_info.mature_age;
mature_days = neuron_info.Neuron_age;
mon = neuron_info.ID;
%% age in month
avg_mat_month = 57.9;
age = real_age/365*12;
mature = mature_days/365*12+avg_mat_month;
%%
figure
my_color = linspecer(8);
vs = violinplot(mature, mon, ...
    'HalfViolin','left',...% right, left, full
    'QuartileStyle','boxplot',... % shadow, boxplot, none
    'DataStyle', 'scatter',... % histogram, scatter, none
    'ShowNotches', false,...
    'ShowMean', false,...
    'ShowMedian', false, ...
     'ViolinColor',my_color);

line([0,8], avg_mat_month*ones(2,1),'linestyle','- -',HandleVisibility='off');
title('Number of neuron from area 8a/46, aligned on maturation')
ylabel('maturation (month)')
xlabel('Animal ID')
set(gca,'fontsize',20)
disp('finished')