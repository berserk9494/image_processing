function [out]=butterhp(I, D0,n)
%D0    - BHPF passes all the frequencies greater than D_{0} value without attenuation and cuts off all the frequencies less than it.
%        is the transition point between H(u, v) = 1 and H(u, v) = 0, so this is termed as cutoff frequency. But instead of making a sharp cut-off
%        (like, Ideal Highpass Filter (IHPF)), it introduces a smooth transition from 0 to 1 to reduce ringing artifacts.
%n     - filter order value

    [M,N] = size(I);
    % Generate a meshgrid for the frequency domain
    [X, Y] = meshgrid(1:N,1:M);
    D = sqrt((X -  floor(N/2)).^2 + (Y - floor(M/2)).^2);
    out=1./(1.+(D0./D).^(2*n));
end