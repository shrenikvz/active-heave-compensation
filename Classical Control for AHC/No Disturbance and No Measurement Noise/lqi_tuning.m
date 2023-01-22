% clear all; clc;

g = 9.81;                   % Acceleration due to gravity
k_oil = 1.8e9;              % Bulk Modulus of Mineral Oil in Pa
Vc = 2e-3;                  % Volume of the hydraulic lines in m^3
Dp = 40e-6;                 % Maximum Pump Displacment in m^3
Dm = 4e-6;                  % Fixed Displacement Motor in m^3
k1p = 0;                    % Leakage Constant for pump
k1m = 0;                    % Leakage Constant for Motor
w_p = 45;                   % Rotation rate of Pump in Hz
Tp = 1;                     % Time constant
k = 200;                    % Gear transmission ratio
r = 0.5;                    % Radius of Winch
eta_m = 0.65;               % Efficency of motor
Jw = 150;                   % Inertia of the winch
d = 1e4;                    % Viscous friction of winch
m = 1e3;                    % Mass load


% up_bar = -(k1p + k1m)*m*g*r/(k*Dm*Dp*w_p);
zwd_max = r*k*Dm*Dp*eta_m*w_p / ((Dm*k)^2*eta_m);

% %------

A = [-1/Tp 0 0 0;
      (-2*k_oil/Vc)*Dp*w_p 0 (2*k_oil/Vc)*Dm*k/r 0;
      0 -r/(Jw + m*r^2)*Dm*k*eta_m -d/(Jw + m*r^2) 0;
      0 0 1 0;];

B = [1/Tp 0 0 0]';

ew = [0 0 r^2*m*g/(Jw + m*r^2) 0]';

C = [0 0 0 1];

D= [0];

% % -------

% Tuning of Q matrix and R matrix

Q = [1 0 0 0 0;0 1e-7 0 0 0;0 0 1 0 0;0 0 0 1 0;0 0 0 0 1e12];
R = 10;


% ----------

sys = ss(A,B,C,D);
[K,S,e] = lqi(sys,Q,R);             % Tuned value of K matrix


%---- End-----%