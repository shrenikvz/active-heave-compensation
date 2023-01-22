close all;
clear; clc;


%% --- Sea State ---

% Sea State 7 (High)

%    Hs = 11;                                  % Wave height significant (m)
%    Tp = 12;                                  % Peak period (s)

% Sea State 6 (Very Rough)

%    Hs = 8.5;                                  % Wave height significant (m)
%    Tp = 14;                                   % Peak period (s)

% Sea State 5 (Rough)

%     Hs = 5.9;                                  % Wave height significant (m)
%     Tp = 12;                                   % Peak period (s)

% Sea State 4 (Moderate)

    Hs =  4;                               % Wave height significant (m)
    Tp =  9;                               % Peak period (s)


% Sea State 3 Slight
% 
%    Hs = 1.5;                              % Wave height significant (m)
%    Tp=  6 ;                               % Peak period (s)




%% --- PM Spectrum ---

w1=0.001:0.001:2;                         % Wave frequencies

g = 9.80665;
omega_m=2*pi/Tp;
const= w1/omega_m;
const2=1./const;
const1= Tp*Hs*Hs;

delta_w = w1(2)-w1(1);  
w1 = w1 + delta_w .* rand(1,length(w1));                        % random selection of frequencies
w3 = w1;

S=(0.3125).*(1/omega_m).*Hs.*Hs.*(const2.^5).*(exp(-1.25.*const2.^4));      % Wave Spectrum
S(1)=0;

%--------------------------------           

t = 0:0.1:10000;                                                 % time vector for a duration of 10000 sec
phi = 2*pi*(rand(1,length(w1)));                                 % random phase of ith frequency
A = sqrt(2*S.*delta_w);                                          % amplitude of ith frequency

for i = 1:length(t)
    wave(i) = sum(A .* cos(w1*t(i) + phi));                         % Wave Elevation
end

[S2,W2]=psd_from_fft(wave',length(wave),100,0.1);  % PSD from time history

figure(1)
subplot(2,1,1)
plot(w3,S,'r');xlabel('w (rad/s)');ylabel('Spectral Density Function (m^2.s)');grid;
subplot(2,1,2)
plot(t,wave);xlabel('t (sec)');ylabel('Wave Elevation (m)');grid;
axis([0 t(end) -inf inf]);



%% --- Beta=135 degree (Wave is coming at 135 degree) ---

load('mdlhydrod.mat')
w=w';
heave_rao_tf_90= rao(:,3,2);
heave_rao_tf_90=interp1(w,heave_rao_tf_90,w1);
pitch_rao_tf_90= rao(:,5,2);
pitch_rao_tf_90=interp1(w,pitch_rao_tf_90,w1);
roll_rao_tf_90= rao(:,4,2);
roll_rao_tf_90=interp1(w,roll_rao_tf_90,w1);

%% --- Calculation of Response Spectra ---

response_spectrum_heave_90 = (abs(heave_rao_tf_90)).^2.*S;      % Heave Spectrum
response_spectrum_pitch_90 = (abs(pitch_rao_tf_90)).^2.*S;      % Pitch Spectrum
response_spectrum_roll_90 = (abs(roll_rao_tf_90)).^2.*S;        % Roll Spectrum

%% --- Heave Motion ---

heave_rao_tf_90=rao(:,3,2);
WAVE=fft(wave);
[x, y]= size(t);
Tmax = t(end);
k = 1:1:y;
wk = (k-ones(1,y)).*((2*pi)/Tmax);
heave_rao_transfer_function_new = zeros(1,y);
for i = 1:y
    if wk(i)<w(1) || wk(i)>w(end)
        heave_rao_transfer_function_new(i) = 0;
    elseif wk(i) == w(end)
        heave_rao_transfer_function_new(i) = heave_rao_tf_90(end);
    elseif wk(i) == w(1)
        heave_rao_transfer_function_new(i) = heave_rao_tf_90(1);
    else
        ind1 = find(w<wk(i));
        ind1 = ind1(end);
        ind2 = find(w>wk(i));
        ind2 = ind2(1);
        heave_rao_transfer_function_new(i) =(wk(i)-w(ind1))*(heave_rao_tf_90(ind2)-heave_rao_tf_90(ind1))/(w(ind2)-w(ind1))+heave_rao_tf_90(ind1);  % Interpolation of heave RAO
    end
end
Response_heave=WAVE.*heave_rao_transfer_function_new;
j = 1;
for i=(1+y)/2+1:y
    Response_heave(i) = conj(Response_heave((end+1)/2-j));    
    j = j+1;
end
response_heave = ifft(Response_heave,'symmetric');      % Heave time history

[S3,W3]=psd_from_fft(response_heave',length(response_heave),100,0.1);   % PSD from time history

%% --- Pitch Motion ---

pitch_rao_tf_90=rao(:,5,2);
WAVE=fft(wave);
[x, y]= size(t);
Tmax = t(end);
k = 1:1:y;
wk = (k-ones(1,y)).*((2*pi)/Tmax);
pitch_rao_transfer_function_new = zeros(1,y);
for i = 1:y
    if wk(i)<w(1) || wk(i)>w(end)
        pitch_rao_transfer_function_new(i) = 0;
    elseif wk(i) == w(end)
        pitch_rao_transfer_function_new(i) = pitch_rao_tf_90(end);
    elseif wk(i) == w(1)
        pitch_rao_transfer_function_new(i) = pitch_rao_tf_90(1);
    else
        ind1 = find(w<wk(i));
        ind1 = ind1(end);
        ind2 = find(w>wk(i));
        ind2 = ind2(1);
        pitch_rao_transfer_function_new(i) =(wk(i)-w(ind1))*(pitch_rao_tf_90(ind2)-pitch_rao_tf_90(ind1))/(w(ind2)-w(ind1))+pitch_rao_tf_90(ind1);   % Interpolation of pitch RAO
    end
end
Response_pitch=WAVE.*pitch_rao_transfer_function_new;
j = 1;
for i=(1+y)/2+1:y
    Response_pitch(i) = conj(Response_pitch((end+1)/2-j));    
    j = j+1;
end   
response_pitch = ifft(Response_pitch,'symmetric');      % Pitch time history

[S4,W4]=psd_from_fft(response_pitch',length(response_pitch),100,0.1);   % PSD from time history


%% --- Roll Motion ---

roll_rao_tf_90=rao(:,4,2);
WAVE=fft(wave);
[x, y]= size(t);
Tmax = t(end);
k = 1:1:y;
wk = (k-ones(1,y)).*((2*pi)/Tmax);
roll_rao_transfer_function_new = zeros(1,y);
for i = 1:y
    if wk(i)<w(1) || wk(i)>w(end)
        roll_rao_transfer_function_new(i) = 0;
    elseif wk(i) == w(end)
        roll_rao_transfer_function_new(i) = roll_rao_tf_90(end);
    elseif wk(i) == w(1)
        roll_rao_transfer_function_new(i) = roll_rao_tf_90(1);
    else
        ind1 = find(w<wk(i));
        ind1 = ind1(end);
        ind2 = find(w>wk(i));
        ind2 = ind2(1);
        roll_rao_transfer_function_new(i) =(wk(i)-w(ind1))*(roll_rao_tf_90(ind2)-roll_rao_tf_90(ind1))/(w(ind2)-w(ind1))+roll_rao_tf_90(ind1);   % Interpolation of roll RAO
    end
end
Response_roll=WAVE.*roll_rao_transfer_function_new;
j = 1;
for i=(1+y)/2+1:y
    Response_roll(i) = conj(Response_roll((end+1)/2-j));    
    j = j+1;
end   
response_roll = ifft(Response_roll,'symmetric');        % Roll time history

[S5,W5]=psd_from_fft(response_roll',length(response_roll),100,0.1);     % PSD from time history


%% --- Combined Roll, Pitch and Heave motion ---

xcoord= 4;                   % xcoordinate of the winch
ycoord= 3;                   % ycoordinate of the winch
beta_slew = pi/3 ;           % Slewing Angle 
netcoordx = xcoord*cos(beta_slew) - ycoord*sin(beta_slew);
netcoordy = xcoord*sin(beta_slew) + ycoord*cos(beta_slew);

response = response_heave - (netcoordx*response_pitch) + (netcoordy*response_roll);    % Combined motion of roll pitch and heave

response2 = [t' response'];


%%  --- Plots --- 

figure(2)
subplot(2,1,1)
plot(w3,response_spectrum_heave_90,'r');xlabel('w (rad/s)');ylabel('Heave Spectra (m^2.s)');grid;
subplot(2,1,2)
plot(t,response_heave);grid;
xlabel('time(sec)');ylabel('Heave');


figure(3)
subplot(2,1,1)
plot(w3,response_spectrum_pitch_90,'r');xlabel('w (rad/s)');ylabel('Pitch Spectra (m^2.s)');grid;
subplot(2,1,2)
plot(t,response_pitch);grid;
xlabel('time(sec)');ylabel('Pitch');


figure(4)
subplot(2,1,1)
plot(w3,response_spectrum_roll_90,'r');xlabel('w (rad/s)');ylabel('Roll Spectra (m^2.s)');grid;
subplot(2,1,2)
plot(t,response_roll);grid;
xlabel('time(sec)');ylabel('Roll');

figure(5)
plot(t,response)
xlabel('time(sec)');ylabel('Z ship(m)');
title('Ship Response to irregular wave');grid;
response1=response';

figure(6)
subplot(4,1,1)
plot(w3,S,W2,S2,'r');xlabel('w (rad/s)');ylabel('Spectral (m^2.s)');
legend('theoretical','measured');
title('PM Spectrum Comparision');
xlim([0 5]);
subplot(4,1,2)
plot(w3,response_spectrum_heave_90,W3,S3,'r');xlabel('w (rad/s)');ylabel('Spectral (m^2.s)');
legend('theoretical','measured');
title('Heave Spectrum Comparision');
xlim([0 5]);
subplot(4,1,3)
plot(w3,response_spectrum_pitch_90,W4,S4,'r');xlabel('w (rad/s)');ylabel('Spectral (m^2.s)');
legend('theoretical','measured');
title('Pitch Spectrum Comparision');
xlim([0 5]);
subplot(4,1,4)
plot(w3,response_spectrum_roll_90,W5,S5,'r');xlabel('w (rad/s)');ylabel('Spectral (m^2.s)');
legend('theoretical','measured');
title('Roll Spectrum Comparision');
xlim([0 5]);


%% --- Generating Spectrum from time history ---

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

%--------------




