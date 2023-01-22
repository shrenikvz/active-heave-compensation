clear all;clc;
w=0:0.01:1;                         % Wave frequencies

S = zeros(1,numel(w));
for i = 1:51
    S(i) = 1e-3;
end

for i = 51:101
    S(i) = 0;
end

delta_w = w(2)-w(1);
w = w + delta_w .* rand(1,length(w)); 

t = 0:0.01:1500;                                                 % time vector for a duration of 10000 sec

phi = 2*pi*(rand(1,length(w)));                                % random phase of ith frequency
A = sqrt(2*S.*delta_w);                                         % amplitude of ith frequency

for i = 1:length(t)
    disturbance(i) = sum(A .* cos(w*t(i) + phi));                     % Wave Elevation
end

disturbance1 = [t' disturbance'];

subplot(2,1,1)
plot(w,S,'r');xlabel('w (rad/s)');ylabel('Spectral Density Function (m^2.s)');grid;
subplot(2,1,2)
plot(t,disturbance);xlabel('t (sec)');ylabel('Disturbance (m)');grid;
axis([0 t(end) -inf inf]);


function [S,W]=psd_from_fft(z,n,m,time_step)   %  z : discrete time signal,n : length of z, m: hamming variable 

z_frequency = fft(z);
R  = z_frequency.*conj(z_frequency)/n;
fr = (0:n-1)/n*(1/time_step);
P  = 2*R*time_step;
w  = hamming(m) ;                
w  = w/sum(w) ;                  
w  = [w(ceil((m+1)/2):m);zeros(n-m,1);w(1:ceil((m+1)/2)-1)];  
w    = fft(w) ;                    
pavg = fft(P) ;                 
pavg = ifft(w.*pavg) ; 
S = abs(pavg(1:ceil(n/2)));
F = fr(1:ceil(n/2));
S=S/(2*pi);
W=2*pi*F;

end
