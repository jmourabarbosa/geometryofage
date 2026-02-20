% ODRd_cue_loc_PCA
% return plots of Explained variance and Effective dimensionality
% adapt from https://github.com/caroline-jahn/LAT_062923/blob/main/Codes_Figure_2/plot_Fig2_PCA
% J Zhu 20240206
%% load data
clearvars
load('firing_rate_cue_dis.mat');
load('group_for_cue_dis.mat');
%% save data
result_save = table;
result_save.mature = mean(group_age,2);
result_save.rotation = mean(rotation,2);
ID = repmat(1,size(group_age,1),1);
result_save.ID = mean(ID,2);
writetable(result_save,'rotation_odrd_cue_dis_500boot_10group_80%_popu_mean.csv');
%% loop each group:
sample_size = 20;
explained_boot = {};
eigenvalues_boot = {};
for g = 1:size(groups,1)
    for nb = 1:100
        % select group data
        group_neuron_rate = rate_neuron_z(:, groups{g,1});
        if ~isempty(group_neuron_rate)
            rand_idx = randsample(size(group_neuron_rate,2),size(group_neuron_rate,2),true);
            group_neuron_rate = group_neuron_rate(:,rand_idx);
            group_neuron_rate(isnan(group_neuron_rate)) = 0;
            for in = 1:size(group_neuron_rate,2)
                shuffle_class = randsample(1:4,4); % bootstrap
                group_neuron_rate(:,in) = group_neuron_rate([shuffle_class,shuffle_class+4],in);
                % shuffle_class = randsample(1:8,8); % shuffle
                % group_neuron_rate(:,in) = group_neuron_rate([shuffle_class],in);
            end
            group_age(g,nb) = mean(groups{g,2}(rand_idx));

            % Perform PCA
            [coeff, score, latent, tsquared, explained, mu] = pca(group_neuron_rate);

            % save the result
            explained_boot{g}(:,nb) = explained;
            eigenvalues_boot{g}(:,nb) = latent;
        else
            explained_boot{g}(nb,:) = nan;
            eigenvalues_boot{g}(nb,:) =  nan;
        end
    end
end
%% plot: Explained variance
figure
my_color = linspecer(20);
% my_color = my_color([1,10,20,30],:);

hold on
for gp = 1:size(groups,1)
    shadedErrorBar([],mean(explained_boot{gp}(1:5,:),2)',[prctile(explained_boot{gp}(1:5,:),97.5,2),prctile(explained_boot{gp}(1:5,:),2.5,2)]', ...
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
my_color = linspecer(20);
% my_color = my_color([1,10,20,30],:);
hold on
for g = 1:size(groups)
    g_x = group_age(g,:);
    g_y = sum(eigenvalues_boot{g}(:,:),1).^2./sum(eigenvalues_boot{g}(:,:).^2,1);
    % errorbar(g_x,mean(sum(eigenvalues_boot{g}(:,:),1).^2./sum(eigenvalues_boot{g}(:,:).^2,1)),std(sum(eigenvalues_boot{g}(:,:),1).^2./sum(eigenvalues_boot{g}(:,:).^2,1)),'k');
    scatter(g_x,g_y);
end
% plot([avg_mat_age avg_mat_age], [0 14],'--')
xlabel('maturation age (month)')
ylabel('Effective dimensionality')
%% plot: first 3 dimension
avg_mat_age = 57.9;
figure
my_color = linspecer(20);
% my_color = my_color([1,10,20,30],:);
hold on
for g = 1:size(groups)
    g_x = group_age(g,:);
    g_y = sum(explained_boot{g}(1:3,:));
    % errorbar(g_x,mean(sum(eigenvalues_boot{g}(:,:),1).^2./sum(eigenvalues_boot{g}(:,:).^2,1)),std(sum(eigenvalues_boot{g}(:,:),1).^2./sum(eigenvalues_boot{g}(:,:).^2,1)),'k');
    scatter(g_x,g_y, color = my_color(g,:));
end
% plot([avg_mat_age avg_mat_age], [0 14],'--')
xlabel('maturation age (month)')
ylabel('% explained variance by first 3 PCs')
%% save data
numerator = cellfun(@(x) sum(x, 1).^2, eigenvalues_boot, 'UniformOutput', false);
denominator = cellfun(@(x) sum(x.^2, 1), eigenvalues_boot, 'UniformOutput', false);
eff_dim_all = cellfun(@(num, den) num ./ den, numerator, denominator, 'UniformOutput', false);
result_save = table;
result_save.mature = age_group_for_pca(:);
result_save.eff_dim = cell2mat(eff_dim_all)';
result_save.ID = repmat('All',size(age_group_for_pca(:)));
writetable(result_save,'eff_dim_pop_5000boot.csv');
%%
