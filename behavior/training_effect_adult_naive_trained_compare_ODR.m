%% load data
clearvars
data1 = load('sac_data_ODR_2012_230518.mat'); % 2012 cohort 
data2 = load('sac_data_ODR_adult_230518.mat'); % 2019 adult
% toDelete = isnan(mean(sac_data.DI,2));
% sac_data(toDelete,:) = [];
%% prep data
data1.sac_data.TP(data1.sac_data.age>2150)=1;
selected1 = data1.sac_data.TP==1;
data1.sac_data = data1.sac_data(selected1,:); % only use the second TP of 2012 cohort

data2.sac_data.TP(:)=1;
selected3 = data2.sac_data.delay==1.5;
data2.sac_data = data2.sac_data(selected3,:);
%% clean out outlier
DI_plt1 = sum(data1.sac_data.DI.*data1.sac_data.class_weight,2);
group_plt1 = data1.sac_data.TP;
upp1 = mean(DI_plt1,"omitnan")+std(DI_plt1,'omitnan')*3; % 3 sigma
% DI_plt1(DI_plt1>upp1) = nan;
% group_plt1(DI_plt1>upp1) = nan;
DI_plt2 = sum(data2.sac_data.DI.*data2.sac_data.class_weight,2);
group_plt2 = data2.sac_data.TP;
upp2 = mean(DI_plt2,"omitnan")+std(DI_plt2,'omitnan')*3; % 3 sigma
% DI_plt2(DI_plt2>upp2) = nan;
% group_plt2(DI_plt2>upp2) = nan;
%% group by TP
my_color = linspecer(5);
figure
vs1 = violinplot(DI_plt1, ...
    group_plt1, ...
    'HalfViolin','left',...% right, left, full
    'QuartileStyle','boxplot',... % shadow, boxplot, none
    'DataStyle', 'none',... % histogram, scatter, none
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
    'ViolinColor',my_color(2,:)*0.7,...
    'EdgeColor', [0.8,0.8,0.8],...
        'ViolinAlpha', 0.9,...
        'ShowData', false,...
        'ShowNotches', false,...
        'ShowMean', true,...
        'ShowMedian', false, ...
                'BoxColor',[1 1 1],...
        'Width',0.3);

xlabel('Age')
ylabel('DI')
xlim([0.5 1.5])
% legend({'2019 TP1','','','','','','','','GRU prerecording'})
set(gca,'fontsize',14)
set(gcf,'Position',[600,100,600,800])
set(gca,'Box','off')
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