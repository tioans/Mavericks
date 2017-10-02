%%%%%%%%%%%%%%%%%%%%%%%%%Loading sounds sample%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load handel.mat
filename = 'handel.wav';
audiowrite(filename,y,Fs);
clear y Fs
[y,Fs] = audioread('handel.wav');
sound(y,Fs);
%%%%%%%%%%%%%%%%%%%%%%%%%Ploting and creating the spectrogram%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    y = y(:,1);
    dt = 1/Fs;
    t = 0:dt:(length(y)*dt)-dt;
    plot(t,y); xlabel('Seconds'); ylabel('Amplitude');
    figure
    plot(psd(spectrum.periodogram,y,'Fs',Fs,'NFFT',length(y)));
    
    spectrogram(y)
 %filename = 'D:\School\YEAR III\Voice recognition\microphone-results.wav';
%audioread(filename,y,60);
