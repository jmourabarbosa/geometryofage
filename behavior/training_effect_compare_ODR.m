%% load data
clearvars
data1 = load('sac_data_ODR_230518.mat'); % 2019 cohort
data2 = load('sac_data_ODR_2012_230518.mat'); % 2012 cohort 
% toDelete = isnan(mean(sac_data.DI,2));
% sac_data(toDelete,:) = [];
%% prep data
selected1 = data1.sac_data.TP<=2;
data1.sac_data = data1.sac_data(selected1,:);% only use the first two TP of 2019 cohort
selected2 = ~contains(data1.sac_data.Task,'ODR3');
data1.sac_data = data1.sac_data(selected2,:);% only use the first two TP of 2019 cohort

data2.sac_data.TP(data2.sac_data.age<=2150)=1;
data2.sac_data.TP(data2.sac_data.age>2150)=2;% use two TP of 2012 cohort
%% clean out outlier
DI_plt1 = sum(data1.sac_data.DI.*data1.sac_data.class_weight,2);
group_plt1 = data1.sac_data.TP;
% upp1 = mean(DI_plt1,"omitnan")+std(DI_plt1,'omitnan')*3; % 3 sigma
upp1 = 62.8;
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
    'QuartileStyle','boxplot',... % shadow, boxplot, none
    'DataStyle', 'scatter',... % histogram, scatter, none
    'ViolinColor',my_color,...
    'EdgeColor', [0.8,0.8,0.8],...
        'ViolinAlpha', 0.9,...
        'ShowData', false,...
        'ShowNotches', false,...
        'ShowMean', true,...
        'ShowMedian', false, ...
        'BoxColor',[1 1 1],...
        'Width',0.3);
vs2 = violinplot(DI_plt2, ...
    group_plt2, ...
    'HalfViolin','right',...% right, left, full
    'QuartileStyle','boxplot',... % shadow, boxplot, none
    'DataStyle', 'none',... % histogram, scatter, none
    'ViolinColor',my_color*0.5,...
    'EdgeColor', [0.8,0.8,0.8],...
        'ViolinAlpha', 0.9,...
        'ShowData', false,...
        'ShowNotches', false,...
        'ShowMean', true,...
        'ShowMedian', false, ...
                'BoxColor',[1 1 1],...
        'Width',0.3);
hold on
xlabel('Time Point')
ylabel('DI')
xlim([0.5 2.5])
legend({'2019 TP1','','','','','','','','2019 TP2','','','','','','','', ...
    '2012 TP1','','','','','','','','2012 TP2'})
set(gca,'fontsize',14)
set(gcf,'Position',[600,100,600,800])
set(gca,'Box','off')
%% add stats
% Define positions for significance line and text
x1 = 1.15; % position for the first violin
x2 = 2.15; % position for the second violin
yMax = max([max(DI_plt1), max(DI_plt2)]); % find the maximum data point and add a margin
yText = yMax + 1; % position slightly above the line for the text
% Draw the significance line above the violins
line([x1, x2], [yMax, yMax], 'Color', 'k', 'LineWidth', 2);
% Add the significance indicator
text((x1+x2)/2, yText, '***', 'HorizontalAlignment', 'center', 'FontSize', 14);