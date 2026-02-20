function [tau, A, B, fitQuality] = calculateIntrinsicTimescale_new(spikeCounts, binSize)
    % Input:
    %   spikeCounts - Matrix where each row represents a trial and columns represent time bins for a single neuron
    %   binSize - The size of time bins in seconds
    % Output:
    %   tau - Estimated intrinsic timescale
    %   A, B - Fit parameters
    %   fitQuality - Structure with goodness-of-fit metrics

    % Calculate the number of trials and bins
    numTrials = size(spikeCounts, 1);
    numBins = size(spikeCounts, 2);
    
    % Exclusion criteria
    avgFiringRate = sum(spikeCounts, 'all') / (numTrials * numBins * binSize);
    if avgFiringRate < 1
        warning('Neuron excluded due to low firing rate (<1 spike/s)');
        tau = NaN;
        A = NaN;
        B = NaN;
        fitQuality = struct('rsquare', NaN, 'rmse', NaN);
        return;
    end

    if any(sum(spikeCounts, 1) == 0)
        warning('Neuron excluded due to no spikes in any time bin across all trials');
        tau = NaN;
        A = NaN;
        B = NaN;
        fitQuality = struct('rsquare', NaN, 'rmse', NaN);
        return;
    end

    % Initialize variables for calculating average autocorrelation
    R_avg = zeros(numBins-1, 1);  % To store average autocorrelation values for each lag

    % Loop through each lag and calculate autocorrelation
    for lag = 1:numBins-1
        autocorrSum = 0;
        count = 0;
        for trial = 1:numTrials
            for i = 1:numBins-lag  % Calculate autocorrelation at 'lag' for each starting point 'i'
                autocorrSum = autocorrSum + (spikeCounts(trial, i) - mean(spikeCounts(trial, :))) * (spikeCounts(trial, i+lag) - mean(spikeCounts(trial, :)));
                count = count + 1;
            end
        end
        if count > 0
            R_avg(lag) = autocorrSum / count;  % Normalizing the sum by the number of counts
        end
    end

    % Normalize by variance
    varTotal = var(spikeCounts(:));
    R_avg = R_avg / varTotal;

    % Prepare the time lags
    timeLags = (1:length(R_avg)) * binSize;

    % Only fit data starting from the first reduction in autocorrelation
    firstReductionIdx = find(diff(R_avg) < 0, 1);
    if isempty(firstReductionIdx) || firstReductionIdx > length(timeLags) - 1
        warning('No valid reduction in autocorrelation found within the time lag range.');
        tau = NaN;
        A = NaN;
        B = NaN;
        fitQuality = struct('rsquare', NaN, 'rmse', NaN);
        return;
    end

    % if timeLags(firstReductionIdx) > 0.15
    %     warning('First reduction in autocorrelation is later than 150 ms.');
    %     tau = NaN;
    %     A = NaN;
    %     B = NaN;
    %     fitQuality = struct('rsquare', NaN, 'rmse', NaN);
    %     return;
    % end

    fitTimeLags = timeLags(firstReductionIdx:end);
    fitR_avg = R_avg(firstReductionIdx:end);

    % Fit the exponential decay model: A*exp(-x/tau) + B
    try
        % % Debugging output
        % disp('Fitting data:');
        % disp(table(fitTimeLags(:), fitR_avg(:)));

        fitType = fittype('A*exp(-x/tau) + B', 'independent', 'x', 'coefficients', {'A', 'tau', 'B'});
        options = fitoptions('Method', 'NonlinearLeastSquares', ...
                             'StartPoint', [max(fitR_avg), 0.05, min(fitR_avg)], ...
                             'Lower', [0, 0, -Inf], ...
                             'Upper', [Inf, Inf, Inf]);
        [fitresult, gof] = fit(fitTimeLags(:), fitR_avg, fitType, options);
        
        % Extract parameters
        A = fitresult.A;
        tau = fitresult.tau;
        B = fitresult.B;

        % Goodness of fit metrics
        rsquare = gof.rsquare;
        rmse = gof.rmse;
        fitQuality = struct('rsquare', rsquare, 'rmse', rmse);

        if tau > 0.5 || rsquare < 0.1 % Example threshold for poor fit
            warning('Excluded due to poor fitting quality.');
            tau = NaN;
            A = NaN;
            B = NaN;
            fitQuality = struct('rsquare', NaN, 'rmse', NaN);
        end

        % % Plot the data and the fitting result
        % figure;
        % plot(fitTimeLags, fitR_avg, 'bo', 'DisplayName', 'Data');
        % hold on;
        % plot(fitTimeLags, A * exp(-fitTimeLags / tau) + B, 'r-', 'DisplayName', sprintf('Fit: A=%.2f, tau=%.2f, B=%.2f', A, tau, B));
        % xlabel('Time Lag (s)');
        % ylabel('Autocorrelation');
        % title('Exponential Decay Fit to Autocorrelation');
        % legend('show');
        % hold off;
        
        % % Debugging output
        % disp('Fitting result:');
        % disp(fitresult);
        
    catch fittingError
        warning(['Fitting error: ', fittingError.message]);
        tau = NaN;
        A = NaN;
        B = NaN;
        fitQuality = struct('rsquare', NaN, 'rmse', NaN);
    end
end
