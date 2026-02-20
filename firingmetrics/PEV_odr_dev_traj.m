%% load data
clearvars
load('odr_data_both_sig_is_best_20240109.mat');
odr_data = odr_data_new;
try
    select_nan = find(neuron_info.best_cue~=0);
    neuron_info = neuron_info(select_nan,:);
    odr_data = odr_data(select_nan,:);
catch
end
%% seg data; optional
% selected1 = find(ismember(neuron_info.ID,'PIC'));
% selected1 = find(nonOutlierIndices);
selected1 = find(neuron_info.delay_duration==3);
neuron_info = neuron_info(selected1,:);
odr_data = odr_data(selected1,:);
%% find sig neuron; optional
% select_sig = find(neuron_info.is_cue_exc(:)|neuron_info.is_del_exc(:));
% select_sig = find(neuron_info.del_e(:));
select_sig = find(neuron_info.cue_e(:)|neuron_info.del_e(:));
neuron_info = neuron_info(select_sig,:);
odr_data = odr_data(select_sig,:);
%% group neuron using slide windows
% even time interval
data = neuron_info.Neuron_age;
% Parameters
data_min = min(neuron_info.Neuron_age);
data_max = max(neuron_info.Neuron_age);
num_groups = 20;
window_width = 150; % days
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

% % even data in each interval
% data = neuron_info.Neuron_age;
% % Number of groups
% num_groups = 30;
% % Sort the data
% [sorted_data, original_indices] = sort(data);
% % Number of elements per group
% num_elements = numel(sorted_data) / num_groups;
% % Initialize cell array to hold groups
% groups = cell(num_groups, 1);
% % Assign sorted data to groups
% for i = 1:num_groups
%     start_index = round((i - 1) * num_elements) + 1;
%     end_index = round(i * num_elements);
%     groups{i} = original_indices(start_index:end_index);
% end
%% get PEV in each group
epoch_start = -1; % in s
epoch_end = 0; % in s
binned_data_all_group = struct;
PEV = struct;
bin_width = 1000;
step_size = 1000;
for g = 1:size(groups)
    Neurons = neuron_info.Neurons(groups{g});
    group_data = odr_data(groups{g},:);
    for n = 1:length(Neurons)
        neuron_data = group_data(n,:);
        spktrain_neuron = [];
        stim_neuron = [];
        for cl = 1:length(neuron_data)
            stim_cl = [];
            try
                spktrain_cl = [];
                [spktrain_temp, ntrs_temp] = Get_spiketrain_partial_aligncue(neuron_data,cl,[epoch_start,epoch_end]);
                spktrain_neuron = [spktrain_neuron; spktrain_temp];
                stim_cl = repmat(cl,1,size(spktrain_temp,1));
                stim_neuron = [stim_neuron, stim_cl];
            catch
                disp(['error processing neuron  ', Neurons(n) '  class=' num2str(cl)])
            end
        end
        [raster_neuron, ~, ~] = spkmtx(spktrain_neuron,0,1000*[epoch_start,epoch_end]);
        start_time = 1;
        end_time = size(raster_neuron, 2);
        the_bin_start_times = start_time:step_size:(end_time - bin_width  + 1);
        the_bin_widths = bin_width .* ones(size(the_bin_start_times));
        curr_binned_data = bin_one_site(raster_neuron, the_bin_start_times, the_bin_widths);  % use the below helper function to bin the data
        % baseline_trial = (mean(curr_binned_data(:,1:2),2));
        % curr_binned_data = curr_binned_data-baseline_trial;
        % curr_binned_data_mean = mean(curr_binned_data,1);
        % curr_binned_data_std = std(curr_binned_data,1);
        % curr_binned_data_std(curr_binned_data_std == 0) = 1;
        % curr_binned_data = (curr_binned_data - curr_binned_data_mean)./curr_binned_data_std; %z score to the baseline mean and std
        binned_data{n} = curr_binned_data;
        binned_labels.stimulus_position{n} = stim_neuron;
        for nt = 1: size(curr_binned_data,2)
            omega_result = mes1way(curr_binned_data,'eta2','group',stim_neuron');
            PEV(g).group(n).omega2(nt) = omega_result.eta2;
            PEV(g).group(n).ci_lo(nt) = omega_result.eta2Ci(1);
            PEV(g).group(n).ci_hi(nt) = omega_result.eta2Ci(2);
        end
        disp(n)
    end
end
disp('finished running')
%% plot scatter
avg_mat_age = 57.9;
% define a few of colors
my_color = linspecer(30);
% my_color = my_color([1,3,5,7],:);
fig = figure;
clf
set(gcf, 'Color', 'White', 'Unit', 'Normalized', ...
    'Position', [0.1,0.1,0.6,0.6] );
t = tiledlayout('flow');
for g = 1:size(groups,1)
    my_size = length(groups{g});
    ax1 = nexttile(1);
    hold on
    if length(data(groups{g}))>10
        x = mean(data(groups{g}))/365*12+avg_mat_age;
        y = mean([PEV(g).group(:).omega2],'omitmissing');
        % y_err = mean([PEV(g).group(:).ci],'omitmissing')
        scatter(x,y,my_size,my_color(g,:),"filled")
        % errorbar(x,y,err_x1, 'horizontal', 'LineStyle', 'none');
        xlabel('maturation age (month)')
        ylabel('Fano Factor')
    end

end
title(ax1, 'cue')
% title(ax2, 'cue')
% title(ax3, 'delay')
title(t, 'ODR all neuron/class')
%% gamm data
%% label neuron data
neuron_info.sess = neuron_info.Neurons(:,1);
%% get PEV
epoch_start = 0.5; % in s
epoch_end = 2.0; % in s
neuron_index = 1:size(odr_data,1);
PEV = table;
Neurons = neuron_info.Neurons;
group_data = odr_data;
for n = 1:length(Neurons)
    neuron_data = group_data(n,:);
    rates_neuron = [];
    stim_neuron = [];
    for cl = 1:length(neuron_data)
        stim_cl = [];
        rates_cl = [];
        try
            spkdata_temp = [];
            [spktrain_temp, ntrs_temp] = Get_spiketrain_partial_aligncue(neuron_data,cl,[epoch_start,epoch_end]);
            rates_cl = cellfun(@numel,spktrain_temp);
            rates_neuron = [rates_neuron; rates_cl];
            stim_cl = repmat(cl,size(rates_cl));
            stim_neuron = [stim_neuron; stim_cl];
        catch
            disp(['error processing neuron  ', Neurons(n) '  class=' num2str(cl)])
        end
    end
    if size(stim_neuron,1)>=32 && mod(size(stim_neuron,1),8) == 0 && mean(rates_neuron) >= 0.5
        for nb = 1%:10
        % rates_neuron = randsample(rates_neuron,size(rates_neuron,1));
        omega_result = mes1way(rates_neuron,'omega2','group',stim_neuron);
        omega2(nb) = omega_result.omega2;
        end
        PEV.omega2(n) = mean(omega2,'omitmissing');
    else
        PEV.omega2(n) = nan;
    end
    PEV.age(n) = neuron_info.Neuron_age(n); % days aligned
    PEV.real_age(n) = neuron_info.Neuron_age(n)+neuron_info.mature_age(n); % days aligned
    PEV.ID(n) = neuron_info.ID(n); % subject ID
    PEV.delay(n) = neuron_info.delay_duration(n);
    PEV.sess(n) = neuron_info.sess(n);
end
disp('finished running')
% age in month
PEV.age = PEV.age/365*12;
PEV.real_age = PEV.real_age/365*12;
PEV = renamevars(PEV,["age","real_age"],["mature","age"]);
% clean up
PEV.omega2(PEV.omega2<=0) = 0;
%% remove outlier
% Grouping by ID
[G, ID] = findgroups(PEV.ID);
nonOutlierIndices = false(height(PEV), 1);
% Iterate over each group and filter out outliers
for i = 1:max(G)
    % Indices for the current group
    currentGroupIndices = G == i;
    % Data for the current group
    currentGroupData = PEV.omega2(currentGroupIndices);
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
PEV_clean = PEV(nonOutlierIndices, :);
%% average in sessions
session = unique(PEV.sess);
for ns = 1:size(session,1)
    sess_neuron = PEV(ismember(PEV.sess,session(ns)),:);
    PEV_s(ns,:) = sess_neuron(1,:);
    PEV_s.omega2(ns) = mean(sess_neuron.omega2,"omitmissing");
    % PEV_s.eta2(ns) = mean(sess_neuron.eta2,"omitmissing");
end
%% save data
writetable(PEV_s,'PEV_sess_del3_all.csv');
%% plot scatters
avg_mature_age = 57.9;
my_color = linspecer(8);
figure
data_table = splitvars(PEV_s);
data_table = sortrows(data_table,'ID');
plt_g = data_table.ID;
mon = unique(plt_g);
hold on
plt_y = data_table.omega2;
upp = mean(plt_y,'omitnan')+std(plt_y,'omitnan')*3; % 3 sigma
low = 0; % 3 sigma
% plt_y(plt_y>upp) = nan;
% plt_y(plt_y<=low) = 0;
plt_x = (data_table.mature+avg_mature_age);
plt_g = data_table.ID;
glme_data_tbl = table(plt_y,plt_x,plt_g);
formula = 'plt_y ~ 1 + plt_x + (1+plt_x|plt_g)';
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
% title('ODR baseline rate on maturation');
% annotation('textbox',[0.7, 0.8, 0.1, 0.1],'String', "p = "+p_values)
xlabel('maturation (month)')
ylabel('PEV')
line(avg_mature_age'*ones(1,2),[0,upp],'linestyle','- -',HandleVisibility='off');
% ylim([0,max(plt_y)])
set(gca,'fontsize',14)
set(gca,'Box','off')
set(gcf,'Position',[600,100,800,800])
%%
%%%%%%%%
function [spiketrain_temp, ntrs] = Get_spiketrain_partial_aligncue(datain,infoin,range)
% return time stamps in ms. poll all trials
% for ODR task
% 20230607, J Zhu
% datain: neuron data (8 classes);
% infoin: class of choice;
% range: [lo, hi] to set a range to use a subset of the trials/epoches (optional)
class = infoin;
TS_all = {};
m_counter = 0;
for n = 1:length(datain{class})
    try
        m_counter = m_counter + 1;
        TS = [];
        TS = datain{class}(n).TS - datain{class}(n).Cue_onT;
        TS_in_range = TS(TS>=range(1)&TS<range(2));
        TS_all{m_counter} = TS_in_range*1000;
    catch
    end
end
spiketrain_temp = TS_all';
ntrs = m_counter;
end
%%%%%%%%
function  binned_data = bin_one_site(raster_data, the_bin_start_times, the_bin_widths)
% a helper function that bins the data for one site
for c = 1:length(the_bin_start_times)
    binned_data(:, c) = mean(raster_data(:, the_bin_start_times(c):(the_bin_start_times(c) + the_bin_widths(c) -1)), 2);
end
end