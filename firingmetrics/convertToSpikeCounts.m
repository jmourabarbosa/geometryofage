function spikeCounts = convertToSpikeCounts(spikeTimes, binSize, epoch)
    % Inputs:
    % spikeTimes - cell array where each cell contains spike timestamps for a trial
    % binSize - size of time bins in milliseconds
    % epoch - array with two numbers specifying the start and end times of the epoch window in milliseconds
    
    % Number of bins
    epochDuration = epoch(2) - epoch(1);
    numBins = ceil(epochDuration / binSize);
    
    % Number of trials
    numTrials = length(spikeTimes);
    
    % Initialize spike counts matrix
    spikeCounts = zeros(numTrials, numBins);
    
    % Loop through each trial
    for i = 1:numTrials
        % Filter spikes to only those within the epoch
        currentSpikeTimes = spikeTimes{i};
        filteredSpikeTimes = currentSpikeTimes(currentSpikeTimes >= epoch(1) & currentSpikeTimes <= epoch(2));
        
        % Adjust filtered spike times relative to the epoch start
        adjustedSpikeTimes = filteredSpikeTimes - epoch(1);
        
        % Determine which bins the spikes fall into
        bins = ceil(adjustedSpikeTimes / binSize);
        
        % Count spikes in each bin
        for binIndex = 1:numBins
            spikeCounts(i, binIndex) = sum(bins == binIndex);
        end
    end
    
    return
end
