function [Iout ,If,G] = homomorphic_filter(I,H)
tic
I_log = log(1 + I);
If = fft2(I_log);
If = fftshift(If);
G = H.*If;
Iout = real(ifft2(ifftshift(G)));
Iout = exp(Iout) - 1;
toc
end

