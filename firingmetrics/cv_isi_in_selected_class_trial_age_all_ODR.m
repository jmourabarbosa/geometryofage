% For ODR task, all neurons, selected correct trials, coefficient of variation (CV).
% The CV is defined as the standard deviation of interspike intervals/the mean of the interspike intervals
% Junda Zhu, 20240124
%% load data
clearvars
% load('odr_data_both_sig_20231017.mat');
load('odr_data_both_sig_is_best_20240109.mat');
odr_data = odr_data_new;
%% clean data
selected = find(neuron_info.best_cue~=0);
neuron_info = neuron_info(selected,:);
odr_data = odr_data(selected,:);
%% seg data; optional
selected1 = find(~ismember(neuron_info.ID,'PIC'));
% selected1 = find(contains(sac_data.Task,'ODR3'));
neuron_info = neuron_info(selected1,:);
odr_data = odr_data(selected1,:);
%% find sig neuron; optional
% select_sig = find(neuron_info.is_cue_exc(:)|neuron_info.is_del_exc(:));
% select_sig = find(neuron_info.del_e(:));
select_sig = find(neuron_info.cue_e(:)|neuron_info.del_e(:));
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
%% CV of isi in cue
epochs = [0.5 3.5;0.5 2.0];
isi = table;
ntrs = [];
for n = 1:length(odr_data)
    try
        isi.age(n) = neuron_info.Neuron_age(n); % days aligned
        isi.real_age(n) = neuron_info.Neuron_age(n)+neuron_info.mature_age(n); % days aligned
        isi.ID(n) = neuron_info.ID(n); % subject ID
        isi.age_group(n) = neuron_info.age_group(n);
        isi.mature_group(n) = neuron_info.mature_group(n);
        isi.delay(n) = neuron_info.delay_duration(n);
        if isi.delay(n) == 3
            [isi_mean_temp, isi_sd_temp, isi_cv_temp] = Get_CV_by_neuron_alignCue(odr_data(n,:),neuron_info.best_del(n),epochs(1,:));
        else
            [isi_mean_temp, isi_sd_temp, isi_cv_temp] = Get_CV_by_neuron_alignCue(odr_data(n,:),neuron_info.best_del(n),epochs(2,:));
        end
        isi.mean(n,:) = isi_mean_temp; % firing rate
        isi.sd(n,:) = isi_sd_temp;
        isi.cv(n,:) = isi_cv_temp;
        isi.mean_log(n,:) = log(isi.mean(n,:)+1);
    catch
        disp(['error processing neuron  ', neuron_info.Neurons{n,:}])
    end
end
disp('finished running')
% age in month
isi.age = isi.age/365*12;
isi.real_age = isi.real_age/365*12;
%% FR in ITI (optional)
epochs = [4.0, 7.0; 2.5, 5.5];
isi = table;
ntrs = [];
for n = 1:length(odr_data)
    try
        isi.age(n) = neuron_info.Neuron_age(n); % days aligned
        isi.real_age(n) = neuron_info.Neuron_age(n)+neuron_info.mature_age(n); % days aligned
        %         FR.ID(n) = convertCharsToStrings(upper(neuron_info.Neurons{n,1}(1:3))); % subject ID
        isi.ID(n) = neuron_info.ID(n); % subject ID
        isi.age_group(n) = neuron_info.age_group(n);
        isi.mature_group(n) = neuron_info.mature_group(n);
        isi.delay(n) = neuron_info.delay_duration(n);
        if isi.delay(n) == 3
            [isi_mean_temp] = Get_all_FR_by_neuron_alignCue(odr_data(n,:),neuron_info(n,:),epochs(1,:));
        else
            [isi_mean_temp] = Get_all_FR_by_neuron_alignCue(odr_data(n,:),neuron_info(n,:),epochs(2,:));
        end
        isi.mean(n,:) = isi_mean_temp; % firing rate
    catch
        disp(['error processing neuron  ', neuron_info.Neurons{n,:}])
    end
end
disp('finished running')
%% FR in ITI (optional)
% epochs = [4.0, 7.0; 2.5, 5.5];
% isi = table;
% ntrs = [];
% for n = 1:length(odr_data)
%     try
%         isi.age(n) = neuron_info.Neuron_age(n); % days aligned
%         isi.real_age(n) = neuron_info.Neuron_age(n)+neuron_info.mature_age(n); % days aligned
%         %         FR.ID(n) = convertCharsToStrings(upper(neuron_info.Neurons{n,1}(1:3))); % subject ID
%         isi.ID(n) = neuron_info.ID(n); % subject ID
%         isi.age_group(n) = neuron_info.age_group(n);
%         isi.mature_group(n) = neuron_info.mature_group(n);
%         isi.delay(n) = neuron_info.delay_duration(n);
%         if isi.delay(n) == 3
%             [isi_mean_temp] = Get_all_FR_by_neuron_alignCue(odr_data(n,:),neuron_info(n,:),epochs(1,:));
%         else
%             [isi_mean_temp] = Get_all_FR_by_neuron_alignCue(odr_data(n,:),neuron_info(n,:),epochs(2,:));
%         end
%         isi.mean(n,:) = isi_mean_temp; % firing rate
%     catch
%         disp(['error processing neuron  ', neuron_info.Neurons{n,:}])
%     end
% end
% disp('finished running')
%% remove outlier
% Grouping by ID
[G, ID] = findgroups(isi.ID);
nonOutlierIndices = false(height(isi), 1);
% Iterate over each group and filter out outliers
for i = 1:max(G)
    % Indices for the current group
    currentGroupIndices = G == i;
    % Data for the current group
    currentGroupData = isi.cv(currentGroupIndices);
    % Calculate IQR and identify non-outliers
    Q1 = quantile(currentGroupData, 0.25);
    Q3 = quantile(currentGroupData, 0.75);
    IQR = Q3 - Q1;
    % Non-outlier condition
    nonOutliers = currentGroupData >= (Q1 - 1.5 * IQR) & currentGroupData <= (Q3 + 1.5 * IQR);
    % Update the non-outlier indices array
    nonOutlierIndices(currentGroupIndices) = nonOutliers;
end
% Filtering the table to remove outliers
isi_clean = isi(nonOutlierIndices, :);
%% save data file
isi_clean = renamevars(isi_clean,["age","real_age"],["mature","age"]);
writetable(isi_clean,'isi_del_best_del_trial_all_neuron_odr_wPIC.csv');
%% violin plot of groups and glme on mature
avg_mature_age = 57.9;
my_color = linspecer(8);
figure
data_table = splitvars(isi_clean);
data_table = sortrows(data_table,"ID");
plt_g = data_table.ID;
mon = unique(plt_g,'stable');
for n = 1:length(mon)
    data_mon = data_table(contains(data_table.ID,mon(n)),:);
    fr_plt1 = [data_mon.cv,data_mon.age+avg_mature_age,data_mon.mature_group];
    [TP_age_mean,TP_num] = grpstats(data_table.age+avg_mature_age,data_table.mature_group,["mean","gname"]);
    for i = 1:size(data_mon,1)
        fr_plt1(i,4) = TP_age_mean(str2double(TP_num)==fr_plt1(i,3));
    end
    upp1 = mean(fr_plt1(:,1),'omitnan')+std(fr_plt1(:,1),'omitnan')*3; % 3 sigma
    low1 = mean(fr_plt1(:,1),'omitnan')-std(fr_plt1(:,1),'omitnan')*3; % 3 sigma
    % fr_plt1(fr_plt1(:,1)>upp1,:) = nan; fr_plt1(fr_plt1(:,1)<low1,:) = nan;
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

plt_y = data_table.cv;
upp = mean(plt_y,'omitnan')+std(plt_y,'omitnan')*3; % 3 sigma
% plt_y(plt_y>upp) = nan;
plt_x = (data_table.age+avg_mature_age);
plt_x_adj = plt_x;
plt_g = data_table.ID;
glme_data_tbl = table(plt_y,plt_x_adj,plt_g);
formula = 'plt_y ~ 1 + plt_x_adj + (1+plt_x_adj|plt_g)';
glme_mdl = fitglme(glme_data_tbl,formula,'Distribution', 'normal');
% Perform hypothesis tests for the fixed-effects coefficients and obtain p-values
[p_values, ~, ~, ~] = coefTest(glme_mdl)

for n = 1:length(mon)
    tbl_new = table();
    tbl_new.plt_x = linspace(min(plt_x),max(plt_x))';
    tbl_new.plt_x_adj = tbl_new.plt_x;
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

%%
%%%%%%%%
function [mean_isi_temp,sd_isi_temp, cv_isi_temp] = Get_CV_by_neuron_alignCue(datain,infoin,epochsin)
%   return the cv of isi of every epochs of each neuron
try
    isi_all = [];
    ntrs_all = 0;
    class = infoin;
    ntrs = 0;
    for n = 1:length(datain{class})
        try
            TS = [];
            TS = datain{class}(n).TS - datain{class}(n).Cue_onT;
            % Extract spikes within the time range
            in_range_spikes = TS(TS >= epochsin(1) & TS <= epochsin(2));
            % Compute inter-spike intervals
            isi_trial = diff(in_range_spikes);
            isi_all = [isi_all, isi_trial];
            ntrs = ntrs + 1;
        catch
        end
    end
    ntrs_all = ntrs_all + ntrs;
catch
end
if size(isi_all,2)/ntrs_all>=0.5*diff(epochsin)
    mean_isi_temp = mean(isi_all,'omitmissing');
    sd_isi_temp = std(isi_all);
    cv_isi_temp = sd_isi_temp./mean_isi_temp;
else
    [mean_isi_temp, sd_isi_temp, cv_isi_temp] = deal(nan);
end
end