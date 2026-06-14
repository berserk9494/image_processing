function atmosphericLight = estimateAtmosphericLight(darkChannel)
    % Determine the number of pixels to consider (0.2% of total pixels)
    numPixels = floor(0.002 * numel(darkChannel));
    % Reshape the channel into a column vector and sort it in descending order
    darkChannelSorted = sort(darkChannel(:), 'descend');
    % Select the top pixels with the highest intensities
    topPixels = darkChannelSorted(1:numPixels);
    % Retrieve the atmospheric light from the hazy image
    atmosphericLight = double(max(topPixels));
end