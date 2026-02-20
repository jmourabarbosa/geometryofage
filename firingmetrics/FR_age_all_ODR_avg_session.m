% FR_age_all_ODR
% For ODR task, all neurons, firing rate vs age
% calculate firing rate in epochs from correct inRF trials pooled
% gourp by subjects, epochs
% Junda Zhu, 20231128
%% load data
clearvars
load('odr_data_both_sig_is_best_20240109.mat');
odr_data = odr_data_new;
%% seg data; optional
% selected1 = find(ismember(neuron_info.Neuron_area,'46'));
selected1 = find(~contains(neuron_info.ID,'PIC'));
neuron_info = neuron_info(selected1,:);
odr_data = odr_data(selected1,:);
%% find sig neuron; optional
% select_sig = find(neuron_info.is_cue_exc(:)|neuron_info.is_del_exc(:));
% select_sig = find(neuron_info.del_e(:));
select_sig = find(neuron_info.del_e(:));
neuron_info = neuron_info(select_sig,:);
odr_data = odr_data(select_sig,:);
%% label neuron data
% even time interval 4 groups
age_edge_used = linspace(min(neuron_info.Neuron_age+neuron_info.mature_age),max(neuron_info.Neuron_age+neuron_info.mature_age),5);
[~,~,age_group_used] = histcounts(neuron_info.Neuron_age+neuron_info.mature_age, age_edge_used);
neuron_info.age_group = age_group_used;
mat_edge_used = linspace(min(neuron_info.Neuron_age),max(neuron_info.Neuron_age),5);
[~,~,mat_group_used] = histcounts(neuron_info.Neuron_age, mat_edge_used);
neuron_info.mature_group = mat_group_used;
neuron_info.sess = neuron_info.Neurons(:,1);
%% FR
epochs = [-1, 0, 0.5, 3.5; -1, 0, 0.5, 2.0];
FR = table;
ntrs = [];
for n = 1:length(odr_data)
    try
        FR.age(n) = neuron_info.Neuron_age(n); % days aligned
        FR.real_age(n) = neuron_info.Neuron_age(n)+neuron_info.mature_age(n); % days aligned
        %         FR.ID(n) = convertCharsToStrings(upper(neuron_info.Neurons{n,1}(1:3))); % subject ID
        FR.ID(n) = neuron_info.ID(n); % subject ID
        FR.age_group(n) = neuron_info.age_group(n);
        FR.mature_group(n) = neuron_info.mature_group(n);
        FR.delay(n) = neuron_info.delay_duration(n);
        FR.sess(n) = neuron_info.sess(n);
        if FR.delay(n) == 3
            [FR_temp, FR_sd_temp, FR_cv_temp] = Get_all_FR_by_neuron_alignCue(odr_data(n,:),neuron_info(n,:),epochs(1,:));
        else
            [FR_temp, FR_sd_temp, FR_cv_temp] = Get_all_FR_by_neuron_alignCue(odr_data(n,:),neuron_info(n,:),epochs(2,:));
        end
        FR.rate(n,:) = FR_temp; % firing rate
        FR.sd(n,:) = FR_sd_temp;
        FR.cv(n,:) = FR_cv_temp;
        FR.rate_adj(n,:) = log(FR.rate(n,:)+exp(1));
    catch
        disp(['error processing neuron  ', neuron_info.Neurons{n,:}])
    end
end
disp('finished running')
%% age in month
FR.age = FR.age/365*12;
FR.real_age = FR.real_age/365*12;
%% average in sessions
session = unique(FR.sess);
for ns = 1:size(session,1)
    sess_neuron = FR(ismember(FR.sess,session(ns)),:);
    FR_sess(ns,:) = sess_neuron(1,:);
    FR_sess.rate(ns,:) = mean(sess_neuron.rate,1);
end
%% save data file
FR_sess = renamevars(FR_sess,["age","real_age"],["mature","age"]);
writetable(FR_sess,'rate_sess_all_trial_all_neuron_odr.csv');
%% violin plot of groups and glme on inverse mature
avg_mature_age = 57.9;
my_color = linspecer(8);
figure
data_table = splitvars(FR_sess);
data_table = sortrows(data_table,"ID");
plt_g = data_table.ID;
mon = unique(plt_g,'stable');
for n = 1:length(mon)
    data_mon = data_table(contains(data_table.ID,mon(n)),:);
    fr_plt1 = [log(data_mon.rate_3+1),data_mon.age+avg_mature_age,data_mon.mature_group];
    % [TP_age_mean,TP_num] = grpstats(DI_plt1(:,2),DI_plt1(:,3),["mean","gname"]);
    [TP_age_mean,TP_num] = grpstats(data_table.age+avg_mature_age,data_table.mature_group,["mean","gname"]);
    for i = 1:size(data_mon,1)
        fr_plt1(i,4) = TP_age_mean(str2double(TP_num)==fr_plt1(i,3));
    end
    upp1 = mean(fr_plt1(:,1),'omitnan')+std(fr_plt1(:,1),'omitnan')*3; % 3 sigma
    low1 = mean(fr_plt1(:,1),'omitnan')-std(fr_plt1(:,1),'omitnan')*3; % 3 sigma
    fr_plt1(fr_plt1(:,1)>upp1,:) = nan; fr_plt1(fr_plt1(:,1)<low1,:) = nan;
    fr_plt1 = fr_plt1(~isnan(fr_plt1(:,1)),:);
    try
        vs1 = violinplot(fr_plt1(:,1), ...
            fr_plt1(:,4), ...
            'HalfViolin','right',...% right, left, full
            'QuartileStyle','boxplot',... % shadow, boxplot, none
            'DataStyle', 'none',... % histogram, scatter, none
            'ViolinColor',my_color(n,:),...
            'EdgeColor', [0.8,0.8,0.8],...
            'ViolinAlpha', 0.25,...
            'ShowData', false,...
            'ShowNotches', false,...
            'ShowMean', false,...
            'ShowMedian', false, ...
            'Width',3);
    catch
    end
end
hold on

data_table = splitvars(FR_sess);
data_table = sortrows(data_table,"ID");
plt_y = data_table.rate_3;
plt_y = log(plt_y+1); % log of firing rate
upp = mean(plt_y,'omitnan')+std(plt_y,'omitnan')*3; % 3 sigma
% plt_y(plt_y>upp) = nan;
plt_x = (data_table.age+avg_mature_age);
plt_x_adj = plt_x.^-1;
plt_g = data_table.ID;
glme_data_tbl = table(plt_y,plt_x_adj,plt_g);
formula = 'plt_y ~ 1 + plt_x_adj + (1+plt_x_adj|plt_g)';
glme_mdl = fitglme(glme_data_tbl,formula,'Distribution', 'normal');
% Perform hypothesis tests for the fixed-effects coefficients and obtain p-values
[p_values, ~, ~, ~] = coefTest(glme_mdl)

for n = 1:length(mon)
    tbl_new = table();
    tbl_new.plt_x = linspace(min(plt_x),max(plt_x))';
    tbl_new.plt_x_adj = tbl_new.plt_x.^-1;
    tbl_new.plt_g = repmat(mon(n),100,1);
    [yhat, yCI] = predict(glme_mdl,tbl_new,'Alpha',0.05);
    h1 = line(tbl_new.plt_x,yhat,'color',my_color(n,:),'LineWidth',5);
    % h2 = plot(tbl_new.plt_x,yCI,'-.','color',my_color(n,:),'LineWidth',1);
end
gs = gscatter(plt_x,plt_y,plt_g,my_color,".",10);
title('ODR baseline rate on maturation');
% annotation('textbox',[0.7, 0.8, 0.1, 0.1],'String', "p = "+p_values)
xlabel('maturation (month)')
ylabel('Log(firing rate + 1)')
line(avg_mature_age'*ones(1,2),[0,upp],'linestyle','- -',HandleVisibility='off');
ylim([0,max(plt_y)])
set(gca,'fontsize',14)
set(gca,'Box','off')
set(gca, 'XTick', TP_age_mean, 'XTickLabels', round(TP_age_mean,1));
set(gcf,'Position',[600,100,800,800])
%% violin plot of groups and glme on inverse age
avg_mature_age = round(mean(unique(neuron_info.mature_age))/365*12);
my_color = linspecer(8);
figure(2)
data_table = splitvars(FR);
data_table = sortrows(data_table,"ID");
plt_g = data_table.ID;
mon = unique(plt_g,'stable');
for n = 1:length(mon)
    data_mon = data_table(contains(data_table.ID,mon(n)),:);
    fr_plt1 = [log(data_mon.rate_1+1),data_mon.real_age,data_mon.age_group];
    % [TP_age_mean,TP_num] = grpstats(DI_plt1(:,2),DI_plt1(:,3),["mean","gname"]);
    [TP_age_mean,TP_num] = grpstats(data_table.real_age,data_table.age_group,["mean","gname"]);
    for i = 1:size(data_mon,1)
        fr_plt1(i,4) = TP_age_mean(str2double(TP_num)==fr_plt1(i,3));
    end
    upp1 = mean(fr_plt1(:,1),'omitnan')+std(fr_plt1(:,1),'omitnan')*3; % 3 sigma
    low1 = mean(fr_plt1(:,1),'omitnan')-std(fr_plt1(:,1),'omitnan')*3; % 3 sigma
    fr_plt1(fr_plt1(:,1)>upp1,:) = nan; fr_plt1(fr_plt1(:,1)<low1,:) = nan;
    fr_plt1 = fr_plt1(~isnan(fr_plt1(:,1)),:);
    try
        vs1 = violinplot(fr_plt1(:,1), ...
            fr_plt1(:,4), ...
            'HalfViolin','right',...% right, left, full
            'QuartileStyle','boxplot',... % shadow, boxplot, none
            'DataStyle', 'none',... % histogram, scatter, none
            'ViolinColor',my_color(n,:),...
            'EdgeColor', [0.8,0.8,0.8],...
            'ViolinAlpha', 0.25,...
            'ShowData', false,...
            'ShowNotches', false,...
            'ShowMean', false,...
            'ShowMedian', false, ...
            'Width',3);
    catch
    end
end
hold on

data_table = splitvars(FR);
data_table = sortrows(data_table,"ID");
plt_y = data_table.rate_1;
plt_y = log(plt_y+1); % log of firing rate
upp = mean(plt_y,'omitnan')+std(plt_y,'omitnan')*3; % 3 sigma
% plt_y(plt_y>upp) = nan;
plt_x = data_table.real_age;
plt_x_adj = plt_x.^-1;
plt_g = data_table.ID;
glme_data_tbl = table(plt_y,plt_x_adj,plt_g);
formula = 'plt_y ~ 1 + plt_x_adj + (1+plt_x_adj|plt_g)';
glme_mdl = fitglme(glme_data_tbl,formula,'Distribution', 'normal');
% Perform hypothesis tests for the fixed-effects coefficients and obtain p-values
[p_values, ~, ~, ~] = coefTest(glme_mdl)

for n = 1:length(mon)
    tbl_new = table();
    tbl_new.plt_x = linspace(min(plt_x),max(plt_x))';
    tbl_new.plt_x_adj = tbl_new.plt_x.^-1;
    tbl_new.plt_g = repmat(mon(n),100,1);
    [yhat, yCI] = predict(glme_mdl,tbl_new,'Alpha',0.05);
    h1 = line(tbl_new.plt_x,yhat,'color',my_color(n,:),'LineWidth',5);
    % h2 = plot(tbl_new.plt_x,yCI,'-.','color',my_color(n,:),'LineWidth',1);
end
% gs = gscatter(plt_x,plt_y,plt_g,my_color,".",10);
title('ODR baseline rate on age');
% annotation('textbox',[0.7, 0.8, 0.1, 0.1],'String', "p = "+p_values)
xlabel('age (month)')
ylabel('Log(firing rate + 1)')
line(avg_mature_age'*ones(1,2),[0,upp],'linestyle','- -',HandleVisibility='off');
ylim([0,max(plt_y)])
set(gca,'fontsize',14)
set(gca,'Box','off')
set(gca, 'XTick', TP_age_mean, 'XTickLabels', round(TP_age_mean,1));
set(gcf,'Position',[600,100,800,800])


%%%%%%%%
function [FR_temp,FR_sd_temp, FR_cv_temp] = Get_all_FR_by_neuron_alignCue(datain,infoin,epochsin)
%   return the mean firing rate of every epochs of each neuron
try
    FR_all = [];
    ntrs_all = 0;
    for class = 1:length(datain)
        ntrs = 0;
        for n = 1:length(datain{class})
            try
                TS = [];
                TS = datain{class}(n).TS - datain{class}(n).Cue_onT;
                nTS =  histcounts(TS,epochsin);
                FR_trial = nTS./diff(epochsin);
                FR_all = [FR_all; FR_trial];
                ntrs = ntrs + 1;
            catch
            end
        end
        ntrs_all = ntrs_all + ntrs;
    end
catch
end
FR_temp = sum(FR_all,1,'omitmissing')/ntrs_all;
FR_sd_temp = std(FR_all,[],1);
FR_cv_temp = FR_sd_temp./FR_temp;
end