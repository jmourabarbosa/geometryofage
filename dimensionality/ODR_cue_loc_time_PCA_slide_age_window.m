% ODR_cue_loc_time_PCA
% return plots of Explained variance and Effective dimensionality
% adapt from https://github.com/caroline-jahn/LAT_062923/blob/main/Codes_Figure_2/plot_Fig2_PCA
% J Zhu 20230814
%% load neuron data
clearvars
load('odr_data_both_sig_is_best_20240109.mat');
odr_data = odr_data_new;
try
    select_nan = find(neuron_info.best_cue~=0);
    neuron_info = neuron_info(select_nan,:);
    odr_data = odr_data(select_nan,:);
catch
end
%% select neuron data
selected1 = find(ismember(neuron_info.ID,'QUA'));
% selected1 = find(neuron_info.delay_duration==3.0);
neuron_info = neuron_info(selected1,:);
odr_data = odr_data(selected1,:);
% select_sig = find(neuron_info.is_cue_exc(:)|neuron_info.is_del_exc(:));
% select_sig = find(neuron_info.is_cue_exc(:)|neuron_info.is_del_exc(:)|neuron_info.is_sac_exc(:));
% neuron_info = neuron_info(select_sig,:);
% odr_data = odr_data(select_sig,:);
%% label neuron data
% even time interval 4 groups
age_edge_used = linspace(min(neuron_info.Neuron_age+neuron_info.mature_age),max(neuron_info.Neuron_age+neuron_info.mature_age),5);
[~,~,age_group_used] = histcounts(neuron_info.Neuron_age+neuron_info.mature_age, age_edge_used);
neuron_info.age_group = age_group_used;
mat_edge_used = linspace(min(neuron_info.Neuron_age),max(neuron_info.Neuron_age),5);
[~,~,mat_group_used] = histcounts(neuron_info.Neuron_age, mat_edge_used);
neuron_info.mature_group = mat_group_used;
% % pre post mature
% age_edge_used = [min(neuron_info.Neuron_age+neuron_info.mature_age),1761,max(neuron_info.Neuron_age+neuron_info.mature_age)];
% [~,~,age_group_used] = histcounts(neuron_info.Neuron_age+neuron_info.mature_age, age_edge_used);
% neuron_info.age_group = age_group_used;
% mat_edge_used = [min(neuron_info.Neuron_age),0,max(neuron_info.Neuron_age)];
% [~,~,mat_group_used] = histcounts(neuron_info.Neuron_age, mat_edge_used);
% neuron_info.mature_group = mat_group_used;
%% PSTH using chronux for all neuron
trial_start = 0.5;
trial_end = 2.0;
plt_save = table;
neuron_index = 1:size(odr_data,1);
psth_all = [];
for n=1:size(neuron_index,2) % loop neuron
    psth_temp = [];
    t_temp = [];
    E_temp = [];
    nn = [];
    neuron_data = odr_data(n,:);
    try
        for cl = 1:8 % loop conditions
            class_data = neuron_data{cl};
            cue_on = [class_data.Cue_onT];    % cueon time for all trials in the class
            spiketimes = {class_data.TS};    % spike times for all trials in the class
            temp_spiketimes = {};
            for i = 1:length(cue_on)
                temp_TS = spiketimes{i};
                temp_TS = temp_TS - cue_on(i);
                temp_spiketimes{i} = temp_TS;    % cueone aligned spike times for all trials in the struct
            end
            spikeforchronux = cell2struct(temp_spiketimes,'spiketime',1);
            [psth_temp(:,cl),t_temp(:,cl),E_temp(:,cl)] = psth(spikeforchronux,0.2,'n',[trial_start,trial_end]);
        end
    catch
        [psth_temp(:,cl),t_temp(:,cl),E_temp(:,cl)] = deal(nan);
    end
    try
        psth_temp_norm = normalize(psth_temp','zscore'); % normalize in each neuron
        psth_neuron = reshape(psth_temp_norm,1,[]);
        psth_neuron(isnan(psth_neuron))=0;
        psth_neuron(isinf(psth_neuron))=0;
        psth_all(n,:) = psth_neuron; % 2D: Neuron x (Condition x Time)
    catch
    end
end
% clear NaN
neuron_info(any(isnan(psth_all), 2),:) = [];
psth_all(any(isnan(psth_all), 2), :) = [];
disp('finished running')
%% group neuron using slide windows
% even time interval
data = neuron_info.Neuron_age;
% Parameters
data_min = min(neuron_info.Neuron_age);
data_max = max(neuron_info.Neuron_age);
num_groups = 20;
window_width = 200; % days
% Calculate step size for the groups to create overlap
total_range = data_max - data_min;
step_size = (total_range - window_width) / (num_groups - 1);
% Create intervals
intervals = (data_min:step_size:data_min + step_size * (num_groups - 1))';
% Initialize cell array to hold groups
groups = cell(num_groups, 1);
% Assign data to groups
for i = 1:num_groups
    group_start = intervals(i);
    group_end = group_start + window_width;
    groups{i} = find(data >= group_start & data <= group_end);
end

%% bootstrap to get variance
avg_mat_age = 57.9;
num_neuron_vector = 30; %number of neurons in each population
eigenvalues_boot = {};
explained_boot = {};
age_group_for_pca = [];
psth_group_for_pca = [];
for nb = 1:500
    psth_group_for_pca = {};
    for g = 1:size(groups)
        if size(groups{g},1) >0
            temp_idx_for_pca = randsample(groups{g},num_neuron_vector,'true');
            age_group_for_pca(nb,g) = mean(neuron_info.Neuron_age(temp_idx_for_pca)/365*12+avg_mat_age);
            psth_group_for_pca{g} = psth_all(temp_idx_for_pca,:);
            % PCA
            [~,~, eigenvalues_boot{g}(:,nb),~,explained_boot{g}(:,nb),~] = pca(psth_group_for_pca{g}');
        else
            eigenvalues_boot{g}(:,nb) = 0;
            explained_boot{g}(:,nb) = 0;
        end
    end
end
%% plot: Explained variance
figure
my_color = linspecer(30);
% my_color = my_color([1,10,20,30],:);

hold on
for gp = 1:size(groups)
    shadedErrorBar([],mean(explained_boot{gp}(1:10,:),2)',[prctile(explained_boot{gp}(1:10,:),97.5,2),prctile(explained_boot{gp}(1:10,:),2.5,2)]', ...
        'lineProps',{'color',my_color(gp,:),'LineWidth',2}, ...
        'patchSaturation',0, 'transparent',true)
end
xlim([0,5])
% ylim([0,100])
xlabel('PC')
% xticks([1 2 3 4 5])
ylabel('% explained variance')
hold off
%% plot: Effective dimensionality
avg_mat_age = 57.9;
figure
my_color = linspecer(30);
% my_color = my_color([1,10,20,30],:);
hold on
for g = 1:size(groups)
    g_x = age_group_for_pca(:,g);
    g_y = sum(eigenvalues_boot{g}(:,:),1).^2./sum(eigenvalues_boot{g}(:,:).^2,1);
    % errorbar(g_x,mean(sum(eigenvalues_boot{g}(:,:),1).^2./sum(eigenvalues_boot{g}(:,:).^2,1)),std(sum(eigenvalues_boot{g}(:,:),1).^2./sum(eigenvalues_boot{g}(:,:).^2,1)),'k');
    scatter(g_x,g_y);
end
% plot([avg_mat_age avg_mat_age], [0 14],'--')
xlabel('maturation age (month)')
ylabel('Effective dimensionality')
%% save data
numerator = cellfun(@(x) sum(x, 1).^2, eigenvalues_boot, 'UniformOutput', false);
denominator = cellfun(@(x) sum(x.^2, 1), eigenvalues_boot, 'UniformOutput', false);
eff_dim_all = cellfun(@(num, den) num ./ den, numerator, denominator, 'UniformOutput', false);
result_save = table;
result_save.mature = age_group_for_pca(:);
result_save.eff_dim = cell2mat(eff_dim_all)';
result_save.ID = repmat('All',size(age_group_for_pca(:)));
writetable(result_save,'eff_dim_pop_500boot_20group_200days.csv');