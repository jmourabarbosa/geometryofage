% Tuning_age_sig_ODR
% For ODR task, sig neurons, width (std of the gaussian curve) vs age
% calculate firing rate in epochs from correct inRF trials pooled
% 20240517, Junda Zhu
%% load neuron data
clearvars
load('odr_data_both_sig_is_best_20240109.mat');
odr_data = odr_data_new;
%% seg data
% selected = find(neuron_info.is_del_exc(:));
selected = find(neuron_info.is_cue_exc(:)|neuron_info.is_del_exc(:)|neuron_info.is_sac_exc(:));
neuron_info = neuron_info(selected,:);
odr_data = odr_data(selected,:);
%%
nn = 1; % num of neuron
for n = 1:length(odr_data)
    MatData = odr_data(n,:);
    rate_sac = [];
    rate_del = [];
    rate_cue = [];
    rate_fix = [];
    for j = 1:length(MatData)
        try
            rate_sac(j) = mean([MatData{j}.sacrate]);
            rate_del(j) = mean([MatData{j}.cuedelay]);
            rate_cue(j) = mean([MatData{j}.cuerate]);
            if isfield(MatData{j},'fixrate')
                rate_fix(j) = mean([MatData{j}.fixrate]);
            else
                rate_fix(j) = mean([MatData{j}.fix]);
            end
        catch
            rate_sac(j) = nan;
            rate_del(j) = nan;
            rate_cue(j) = nan;
            rate_fix(j) = nan;
        end
    end
    fix_rate(n) = nanmean(rate_fix);

    rate_test = rate_del;
    [~, max_test_class] = max(rate_test(1:8));
    %[a max_class] = min(varCue(1:8));   % for inhibit neurons

    % align max firing rate
    for n_class = 1:8
        if n_class >= max_test_class
            class_mean(n_class-max_test_class+1) = rate_test(n_class);
        else
            class_mean(n_class-max_test_class+1+8) = rate_test(n_class);
        end
    end
    class_mean_order = [5,4,3,2,1,8,7,6];
    class_mean_all(1:8,n) = class_mean(class_mean_order);
    class_mean_all(9,n) = class_mean_all(1,n);

    % gaussian fit
    try
    [~, ~, ~, ~, predicted_rate, d(n), R2(n)] = gaus_fit_8_loc(class_mean_all(:,n));
    catch
        [d(n), R2(n)] = deal(nan);
    end
    nn = nn + 1;
    ID(n) = convertCharsToStrings(upper(neuron_info.Neurons{n,1}(1:3))); % subject ID
end
% fix_rate_mean_all = nanmean(fix_rate);
%% save results
tuning_width = table;
tuning_width.d = d';
tuning_width.mature = neuron_info.Neuron_age;
tuning_width.ID = ID';
tuning_width.r2 = R2';
tuning_width.fixrate = fix_rate';
tuning_width.mature = tuning_width.mature/365*12;
tuning_width.age = (neuron_info.Neuron_age+neuron_info.mature_age)/365*12;
%% export
writetable(tuning_width, 'tuning_width_del_all_neuron_with_r2_fixrate.csv');
%% use raw rate for each neuron
nn = 1; % num of neuron
for n = 1:length(odr_data)
    MatData = odr_data(n,:);
    rate_sac = [];
    rate_del = [];
    rate_cue = [];
    rate_fix = [];
    for j = 1:length(MatData)
        try
            rate_sac(j) = mean([MatData{j}.sacrate]);
            rate_del(j) = mean([MatData{j}.cuedelay]);
            rate_cue(j) = mean([MatData{j}.cuerate]);
            if isfield(MatData{j},'fixrate')
                rate_fix(j) = mean([MatData{j}.fixrate]);
            else
                rate_fix(j) = mean([MatData{j}.fix]);
            end
        catch
            rate_sac(j) = nan;
            rate_del(j) = nan;
            rate_cue(j) = nan;
            rate_fix(j) = nan;
        end
    end
    fix_rate(n) = nanmean(rate_fix);

    rate_test = rate_cue;
    [~, max_test_class] = max(rate_test(1:8));
    %[a max_class] = min(varCue(1:8));   % for inhibit neurons

    % align max firing rate
    for n_class = 1:8
        if n_class >= max_test_class
            class_mean(n_class-max_test_class+1) = rate_test(n_class);
        else
            class_mean(n_class-max_test_class+1+8) = rate_test(n_class);
        end
    end
    class_mean_order = [5,4,3,2,1,8,7,6];
    class_mean_all(1:8,n) = class_mean(class_mean_order);
    class_mean_all(9,n) = class_mean_all(1,n);

    % gaussian fit
    try
    [~, ~, ~, ~, ~, d(n), R2(n)] = gaus_fit_8_loc(class_mean_all(:,n));
    catch
        [d(n), R2(n)] = deal(nan);
    end
    nn = nn + 1;
    ID(n) = convertCharsToStrings(upper(neuron_info.Neurons{n,1}(1:3))); % subject ID
end
% fix_rate_mean_all = nanmean(fix_rate);
%%
my_color = linspecer(8);
figure
mon = unique(plt_g);
% for n = 1:length(mon)
%     tbl_new = table();
%     tbl_new.plt_x = linspace(min(plt_x),max(plt_x))';
%     tbl_new.plt_g = repmat(mon(n),100,1);
%     [yhat, yCI] = predict(glme_mdl,tbl_new,'Alpha',0.05);
%     h1 = line(tbl_new.plt_x,yhat,'color',my_color(n,:),'LineWidth',5);
%     hold on;
%     h2 = plot(tbl_new.plt_x,yCI,'-.','color',my_color(n,:),'LineWidth',1);
% end
gs = gscatter(plt_x,plt_y,plt_g,my_color,".",20);
%% plot fitting cure
figure()
title("Sig d Del rate 46 neuron")
hold on
plot(loctt,rateyy)
errorbar(s,m,sem,'marker','o','linestyle','none')
fix_edges=[1:.5:9];
line(fix_edges,fix_rate_mean_all*ones(1,length(fix_edges)),'linestyle','- -');
%%%%%%%%%%%%
%display std
std_tuning_curve = d
annotation('textbox',[0.25, 0.75, 0.2, 0.1],'String', "std tuning curve: " + d +" n = "+length(odr_data))
xlim([0.5 9.5]);

% figure;
% rates = [real_means(6) real_means(7) real_means(8);real_means(5) real_means(9) real_means(1); real_means(4) real_means(3) real_means(2)];
% [X,Y]=meshgrid(-10:10:10, -10:10:10);
% [XI YI]=meshgrid(-10:1:10, -10:1:10);
% rateint=interp2(X,Y,rates,XI,YI);
% contourf(XI,YI,rateint,40)
% shading flat
