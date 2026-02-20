% to find example sessions and plot to show the DI
% J Zhu, 20240222
%% select example sessions
clearvars
load("sac_data_ODR_230518.mat")
file_name{1} = 'oli019_1';
file_name{2} = 'pic005_1';
file_name{3} = 'uni066_1';
%% read raw files
for i = 1:length(file_name)
    [~, class_indices, x_endpoint, y_endpoint, ~, RT] = proSaccade_alltrial(file_name{i});
    %     [~, class_indices, x_endpoint, y_endpoint, ~, RT] = proSaccade_correcttrial(file_name{i});
    class_indices(abs(y_endpoint)+abs(x_endpoint)>=30) = nan;
    x_endpoint(abs(y_endpoint)+abs(x_endpoint)>=30) = nan;
    y_endpoint(abs(y_endpoint)+abs(x_endpoint)>=30) = nan;
    if sum(~isnan(class_indices))>=30 % at least x trials completed/correct
        figure;
        for cl = 3
            try
                x_class = x_endpoint(class_indices==cl);
                y_class = y_endpoint(class_indices==cl);
            catch
                x_class = nan;
                y_class = nan;
            end
            % Plot the scatter points
            nexttile
            scatter(x_class, y_class, 'filled');
            hold on; % Keep the plot open to add the ellipse
            % Calculate the mean of the points
            mean_x = mean(x_class);
            mean_y = mean(y_class);
            % Calculate the covariance matrix and its eigenvalues and eigenvectors
            cov_matrix = cov(x_class, y_class);
            [eigenvectors, eigenvalues] = eig(cov_matrix);
            % Extract the square root of the eigenvalues (standard deviations)
            std_devs = sqrt(diag(eigenvalues));
            % Calculate the angle of rotation of the ellipse
            angle = atan2(eigenvectors(2,1), eigenvectors(1,1));
            % Number of points to define the ellipse
            num_points = 100;
            theta = linspace(0, 2*pi, num_points);
            % Ellipse parameters
            a = std_devs(1); % Semi-major axis
            b = std_devs(2); % Semi-minor axis
            % Parametric equation of the ellipse
            ellipse_x = a * cos(theta);
            ellipse_y = b * sin(theta);
            % Rotate the ellipse to align with the eigenvectors
            R = [cos(angle) -sin(angle); sin(angle) cos(angle)];
            ellipse_rotated = R * [ellipse_x; ellipse_y];
            % Plot the ellipse, shifted to the mean
            plot(ellipse_rotated(1,:) + mean_x, ellipse_rotated(2,:) + mean_y, 'r', 'LineWidth', 2);
            % Labels and legend
            xlabel('X');
            ylabel('Y');
            xlim([-5 5])
            ylim([5 15])
            % title('Scatter Plot with Standard Deviation Ellipse');
            % legend('Data Points', '1 Std Dev Ellipse');
            hold off; % Release the plot
        end
    end
end
%% save table
% Sample points in x and y coordinates
x = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];
y = [1 2 3 4 5 6 7 8 9 9 7 5 3 3 6 7 5];

% Plot the scatter points
figure;
scatter(x, y, 'filled');
hold on; % Keep the plot open to add the ellipse

% Calculate the mean of the points
mean_x = mean(x);
mean_y = mean(y);

% Calculate the covariance matrix and its eigenvalues and eigenvectors
cov_matrix = cov(x, y);
[eigenvectors, eigenvalues] = eig(cov_matrix);

% Extract the square root of the eigenvalues (standard deviations)
std_devs = sqrt(diag(eigenvalues));

% Calculate the angle of rotation of the ellipse
angle = atan2(eigenvectors(2,1), eigenvectors(1,1));

% Number of points to define the ellipse
num_points = 100;
theta = linspace(0, 2*pi, num_points);

% Ellipse parameters
a = std_devs(1); % Semi-major axis
b = std_devs(2); % Semi-minor axis

% Parametric equation of the ellipse
ellipse_x = a * cos(theta);
ellipse_y = b * sin(theta);

% Rotate the ellipse to align with the eigenvectors
R = [cos(angle) -sin(angle); sin(angle) cos(angle)];
ellipse_rotated = R * [ellipse_x; ellipse_y];

% Plot the ellipse, shifted to the mean
plot(ellipse_rotated(1,:) + mean_x, ellipse_rotated(2,:) + mean_y, 'r', 'LineWidth', 2);

% Labels and legend
xlabel('X');
ylabel('Y');
title('Scatter Plot with Standard Deviation Ellipse');
legend('Data Points', '1 Std Dev Ellipse');

hold off; % Release the plot
disp('Finish running')

xlabel('Age (days)')
ylabel('percentage correct')