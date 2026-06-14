function H = idealhp(I, cutoff)
% size: size of the filter in pixels (assumed to be square)
% cutoff: frequency cutoff for the high-pass filter

% Generate a meshgrid for the frequency domain
    [M,N] = size(I);
    % Generate a meshgrid for the frequency domain
    [X, Y] = meshgrid(1:N,1:M);
    D = ((X -  floor(N/2)).^2 + (Y - floor(M/2)).^2);

    % Create the ideal high-pass filter
    H = double(D >= cutoff);
end