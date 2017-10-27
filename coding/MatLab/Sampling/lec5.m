% ---------------------------------------------------------------------- %
close all
clear variables
clc
% ---------------------------------------------------------------------- %
%->Implement a sinusoidal function in Matlab which emulates a continuous
%function
A = 1;
f = 5; %frequency
ts=0.001;
fs=1/0.001;
n1 = 0:ts:1;%small period to mimic a analog signal
y1 = A*cos(2*pi*f*n1) + A*cos(2*pi*10*n1); %"analog" signal
figure
subplot(5,1,1) % add first plot in 2 x 1 grid
plot(n1,y1) %ploting of "analog" sinusoid   
title('"Analog" sinusoid')

X = rand(size(n1)) + y1;
subplot(5,1,2)
plot(n1,X)
title('"Analog" sinusoid with noise')
% ---------------------------------------------------------------------- %
T = 1/fs;             % Sampling period       
L = size(y1,2);       % Length of signal
t = (0:L-1)*T;        % Time vector
% ---------------------------------------------------------------------- %
%fft of original signal
Y = fft(X);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = fs*(0:(L/2))/L;
subplot(5,1,3) % add first plot in 2 x 2 grid
plot(f,P1)
axis([0 150 0 1.3]);
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')
% ---------------------------------------------------------------------- %

load('/Users/kamranthomas/Documents/MATLAB/SignalProcessing/IIR_LP_Btrw.mat')

subplot(5,1,4);
Xfilt = filtfilt(SOS, G, X);
plot(n1,Xfilt)
title('filterd signal (Low pass butterworth)')
% ---------------------------------------------------------------------- %
%fft of filtered signal
Yfilt = fft(Xfilt);
P2 = abs(Yfilt/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = fs*(0:(L/2))/L;
subplot(5,1,5) % add first plot in 2 x 2 grid
plot(f,P1)
axis([0 150 0 1.3]);
title('Single-Sided Amplitude Spectrum of X(t) filtered signal')
xlabel('f (Hz)')
ylabel('|P1(f)|')
% ---------------------------------------------------------------------- %

