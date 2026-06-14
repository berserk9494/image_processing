function H = gaushp(I, gL, gH, D0, C)
%gL     - low values
%gH     - high values
%D0     - size of the gaussian, affects more/less frequencies
%C      - as with D0, it affects the exponential in H. It is like a tug of war
%         between D0 and C, working in different ranges. D0 is a sharpness factor
%         for the big changes in the image whereas c is sharpening in a small range.


    [M,N] = size(I);
    % Generate a meshgrid for the frequency domain
    [X, Y] = meshgrid(1:N,1:M);
    gaussianNumerator = (X -  floor(N/2)).^2 + (Y - floor(M/2)).^2;
    H = (gH-gL)*(1-exp(-C*gaussianNumerator./(D0.^2)))+gL;
end

