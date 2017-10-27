%%%%%%%%%%%%%%%%%%%%%%%%%Loading sounds sample%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
load handel.mat
filename = 'handel.wav';
audiowrite(filename,y,Fs);
clear y Fs
[y,Fs] = audioread('handel.wav');
sound(y,Fs);
%}
[y,Fs] = audioread('microphone-results.wav');
sound(y,Fs);

%%%%%%%%%%%%%%%%%%%%%%%%%Creating and ploting the spectrogram%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    y = y(:,1);
    dt = 1/Fs;
    t = 0:dt:(length(y)*dt)-dt;
    plot(t,y); xlabel('Seconds'); ylabel('Amplitude');
    figure
    plot(psd(spectrum.periodogram,y,'Fs',Fs,'NFFT',length(y)));
    figure
    spectrogram(y)
%Adding noise
   X=awgn(y,10)
    figure
    plot(X)
    sound(X,Fs)

load('C:\Users\Stefan\Documents\Mavericks\Filter1.mat')


Xfilt = filtfilt(SOS, G, X);
plot(X,Xfilt)
title('filtered signal (Low pass butterworth)')