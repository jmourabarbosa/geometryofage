%% load data
clearvars
data1 = load('sac_data_ODR_230518.mat'); % 2019 cohort
data2 = load('sac_data_ODR_G_230522.mat'); % GRU 
% toDelete = isnan(mean(sac_data.DI,2));
% sac_data(toDelete,:) = [];
%% prep data
selected1 = data1.sac_data.TP==1;
data1.sac_data = data1.sac_data(selected1,:);% only use the first two TP of 2019 cohort
% selected2 = contains(data1.sac_data.Task,'ODR3');
% data1.sac_data = data1.sac_data(selected2,:);% only use the first two TP of 2019 cohort

data2.sac_data.TP(:)=1;
% selected3 = data2.sac_data.delay==1.5;
% data2.sac_data = data2.sac_data(selected3,:);% only use the first two TP of 2019 cohort
%% clean out outlier
DI_plt1 = sum(data1.sac_data.DI.*data1.sac_data.class_weight,2);
group_plt1 = data1.sac_data.TP;
upp1 = mean(DI_plt1,"omitnan")+std(DI_plt1,'omitnan')*3; % 3 sigma
DI_plt1(DI_plt1>upp1) = nan;
group_plt1(DI_plt1>upp1) = nan;
DI_plt2 = sum(data2.sac_data.DI.*data2.sac_data.class_weight,2);
group_plt2 = data2.sac_data.TP;
upp2 = mean(DI_plt2,"omitnan")+std(DI_plt2,'omitnan')*3; % 3 sigma
DI_plt2(DI_plt2>upp2) = nan;
group_plt2(DI_plt2>upp2) = nan;
%% group by TP
my_color = linspecer(5);
figure
vs1 = violinplot(DI_plt1, ...
    group_plt1, ...
    'HalfViolin','left',...% right, left, full
    'QuartileStyle','shadow',... % shadow, boxplot, none
    'DataStyle', 'scatter',... % histogram, scatter, none
    'ViolinColor',my_color,...
    'ShowNotches', false,...
    'ShowMean', true,...
    'ShowMedian', false);
vs2 = violinplot(DI_plt2, ...
    group_plt2, ...
    'HalfViolin','right',...% right, left, full
    'QuartileStyle','shadow',... % shadow, boxplot, none
    'DataStyle', 'scatter',... % histogram, scatter, none
    'ViolinColor',my_color(2,:)*0.7,...
    'ShowNotches', false,...
    'ShowMean', true,...
    'ShowMedian', false);

xlabel('Time Point')
ylabel('DI')
xlim([0.5 1.5])
legend({'2019 TP1','','','','','','','','GRU prerecording'})
set(gca,'fontsize',14)
set(gcf,'Position',[600,100,600,800])
%% add stats
[h,p,ci,stats] = ttest2(DI_plt1,DI_plt2);
% Define positions for significance line and text
x1 = 0.85; % position for the first violin
x2 = 1.15; % position for the second violin
yMax = max([max(DI_plt1), max(DI_plt2)])-10; % find the maximum data point and add a margin
yText = yMax + 1; % position slightly above the line for the text
% Draw the significance line above the violins
line([x1, x2], [yMax, yMax], 'Color', 'k', 'LineWidth', 2);
% Add the significance indicator
text((x1+x2)/2, yText, '**', 'HorizontalAlignment', 'center', 'FontSize', 14);