% timescale_age_all_ODR
% For ODR task, all neurons, baseline intrinsic timescale vs development
% calculate firing rate in fixation from all trials
% gourp by subjects
% Junda Zhu, 20240517
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
% selected1 = find(~ismember(neuron_info.ID,'PIC'));
selected1 = find(contains(neuron_info.ID,'PIC')&contains(neuron_info.Neuron_area,'8a'));
neuron_info = neuron_info(selected1,:);
odr_data = odr_data(selected1,:);
%% find sig neuron; optional
% select_sig = find(neuron_info.is_cue_exc(:)|neuron_info.is_del_exc(:));
% select_sig = find(neuron_info.del_e(:));
select_sig = find(neuron_info.cue_e(:)|neuron_info.del_e(:)|neuron_info.sac_e(:));
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
%% spikeTimes
timescale = table;
epochs = [-1, 0];
bin_size = 0.05; 

for n = 1:length(odr_data)
    try
        timescale.age(n) = neuron_info.Neuron_age(n); % days aligned
        timescale.real_age(n) = neuron_info.Neuron_age(n)+neuron_info.mature_age(n); % days aligned
        timescale.ID(n) = neuron_info.ID(n); % subject ID
        timescale.age_group(n) = neuron_info.age_group(n);
        timescale.mature_group(n) = neuron_info.mature_group(n);
        timescale.delay(n) = neuron_info.delay_duration(n);
        [spike_times_temp] = Get_spike_times_by_neuron_alignCue(odr_data(n,:));
        timescale.spike_times{n} = spike_times_temp; % firing rate
    catch
        disp(['error processing neuron  ', neuron_info.Neurons{n,:}])
    end
    % Convert spike timestamps to spike counts
    timescale.spike_counts{n} = convertToSpikeCounts(spike_times_temp, bin_size, epochs);
end
disp('finished running')
%%
for n = 1:size(timescale,1)
% Calculate the intrinsic timescale
[timescale.tau(n), timescale.A(n), timescale.B(n)] = calculateIntrinsicTimescale_new(timescale.spike_counts{n}, bin_size);
end
%% age in month
timescale.mature = timescale.age/365*12;
timescale.age = timescale.real_age/365*12;
%% save results
tau = table;
tau.tau = timescale.tau;
tau.mature = timescale.mature;
tau.ID = timescale.ID;
tau.age = timescale.age;
%% export
writetable(tau, 'tau_fix_all_trial_all_neuron.csv');
%% plot
avg_mature_age = 57.9;
my_color = linspecer(8);
figure
data_table = splitvars(timescale);
data_table = sortrows(data_table,"ID");
plt_g = data_table.ID;
plt_x = (data_table.mature);
plt_y = data_table.tau;
mon = unique(plt_g,'stable');
hold on
gs = gscatter(plt_x,plt_y,plt_g,my_color,".",10);
title('ODR baseline rate on maturation');
% annotation('textbox',[0.7, 0.8, 0.1, 0.1],'String', "p = "+p_values)
xlabel('maturation (month)')
ylabel('tau')
ylim([0,max(plt_y)])
set(gca,'fontsize',14)
set(gca,'Box','off')
set(gcf,'Position',[600,100,800,800])
%%
%%%%%%%%
function [TS_all] = Get_spike_times_by_neuron_alignCue(datain)
%   return the mean firing rate of every epochs of each neuron
try
    TS_all = {};
    for cl = 1:size(datain,2)
        for n = 1:length(datain{cl})
            try
                TS = [];
                TS = datain{cl}(n).TS - datain{cl}(n).Cue_onT;
                TS_all{cl,n} = TS;
            catch
            end
        end
    end
    TS_all = reshape(TS_all,1,[]);
catch
end
end