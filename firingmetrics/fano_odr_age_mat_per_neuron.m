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
% selected1 = find(~ismember(neuron_info.ID,'PIC'));
selected1 = find(nonOutlierIndices);
% selected1 = find(neuron_info.delay_duration==3);
neuron_info = neuron_info(selected1,:);
odr_data = odr_data(selected1,:);
%% find sig neuron; optional
% select_sig = find(neuron_info.is_cue_exc(:)|neuron_info.is_del_exc(:));
% select_sig = find(neuron_info.del_e(:));
select_sig = find(neuron_info.cue_e(:)|neuron_info.del_e(:));
neuron_info = neuron_info(select_sig,:);
odr_data = odr_data(select_sig,:);
%% get spike hist count
trial_start = -1000; % in ms
trial_end = 2700; % in ms
neuron_index = 1:size(odr_data,1);
Best_class = neuron_info.best_cue;
spk_for_fano = struct;
Neurons = neuron_info.Neurons;
for n = 1:length(Neurons)
    neuron_data = odr_data(n,:);
    n_cond = 1;
    for cl = Best_class(n)%1:length(neuron_data)
        try
            spkdata_temp = [];
            [spiketrain_temp, ntrs_temp1] = Get_spiketrain_partial_aligncue(neuron_data,cl,[trial_start,trial_end]);
            [spkdata_temp, tlo, thi] = spkmtx(spiketrain_temp,0,[trial_start,trial_end]);
            spk_for_fano(n).group(n_cond).spikes = spkdata_temp;
            n_cond = n_cond + 1;
        catch
            disp(['error processing neuron  ', Neurons(n) '  Dir1=' num2str(cl)])
        end
    end
end
%% COMPUTE FANO FACTOR ETC
times = 100:100:3500;
fanoP.boxWidth = 100; % width of the sliding window in which the counts are made
fanoP.matchReps = 0;  % number of random choices regarding which points to throw away when matching distributions
fanoP.binSpacing = 0.25;% 0.25; % bin width when computing distributions of mean counts 0.25
fanoP.alignTime = 1000; % time of event that data are aligned to (in the output structure, times will be expressed relative to this)
for g = 1:length(Neurons)
    try
        Result(g) = VarVsMean_XQ(spk_for_fano(g).group, times, fanoP);
    catch
    end
end
%% save data
% save([outPutDir filename 'fanostep20box100.mat'], 'Result');%*MF.mat matched firing rate;
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
for g = 1:length(Neurons)
    ax1 = nexttile(1);
        x = neuron_info.Neuron_age(g)/365*12+avg_mat_age;
        y = mean(Result(g).FanoFactorAll(1:10),1);
        scatter(x,y)
        xlabel('maturation age (month)')
        ylabel('Fano Factor')
    hold on
    % ax2 = nexttile(2);
    %     x = mean(data(groups{g}))/365*12+avg_mat_age;
    %     y = mean(Result(g).FanoFactorAll(10:15),1);
    %     scatter(x,y,my_size,my_color(g,:),"filled")
    %     xlabel('maturation age (month)')
    %     ylabel('Fano Factor')
    % hold on
    % ax3 = nexttile(3);
    %     x = mean(data(groups{g}))/365*12+avg_mat_age;
    %     y = mean(Result(g).FanoFactorAll(16:end),1);
    %     scatter(x,y,my_size,my_color(g,:),"filled")
    %     xlabel('maturation age (month)')
    %     ylabel('Fano Factor')
    % hold on
end
title(ax1, 'pre cue fixation')
title(ax2, 'cue')
title(ax3, 'delay')
title(t, 'ODR all neuron/class')
%% plot traces
% define a few of colors
my_color = linspecer(50);
% my_color = my_color([1,3,5,7],:);
fig = figure;
clf
set(gcf, 'Color', 'White', 'Unit', 'Normalized', ...
    'Position', [0.1,0.1,0.45,0.45] );
% t = tiledlayout('flow');
for g = 1:size(groups)
    xlo = min(Result(g).times);
    xhi = max(Result(g).times);
    ylo = 1;
    yhi = 2;
    hold on

    % % plot errors as a shaded area
    % y11 = Result(g).FanoAll_95CIs(:,1);
    % y12 = Result(g).FanoAll_95CIs(:,2);
    % fill([Result(g).times; Result(g).times(end:-1:1)], [y11; y12(end:-1:1)], 'w', ...
    %     'EdgeColor', my_color(g,:), 'FaceColor', my_color(g,:),'FaceAlpha',0.1,'EdgeAlpha',0.1)

    % plot mean on top
    line(Result(g).times, Result(g).FanoFactorAll, 'linewidth', 0.5, 'color', my_color(g,:))
    % axis([xlo xhi ylo yhi])
    xlabel('Time (ms)')
    ylabel('Fano Factor')

    % if g == 4
    %     plot([0 0], [ylo yhi], "Color",'k')
    %     plot([500 500], [ylo yhi], "Color",'k')
    %     plot([3500 3500], [ylo yhi], "Color",'k')
    %     legend({'','1','','2','','3','','4'})
    % end

end
title('ODR sig neuron/class')
set(gca, 'tickdir', 'out')
origUnits = fig.Units;
fig.Units = fig.PaperUnits;
fig.PaperSize = fig.Position(3:4);% set the Page Size (figure's PaperSize) to match the figure size in the Paper units
fig.Units = origUnits;% restore the original figure Units