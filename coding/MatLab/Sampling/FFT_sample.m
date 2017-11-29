%--------------------------------------------------------
%Loading the "Hello" sample
subplot(3,1,1);
[y,Fs] = audioread('microphone-results.wav');
sound(y,Fs);
    dt = 1/Fs;
    t = 0:dt:(length(y)*dt)-dt;
    %figure();
    plot(t,y); xlabel('Seconds'); ylabel('Amplitude');
title('Hello waveform');

%--------------------------------------------------------
subplot(3,1,2);
N = length(y);
NEFT = 2^nextpow2(N)
Xk=fft(y, NEFT)/N;                % perform N-point DFT of signal
fa = 8000/2*linspace(0,1,NEFT/2+1); 
%figure();
plot(fa, 2*abs(Xk(1:NEFT/2+1)));
title('FFT Spectrum Estimates / Power Spectral Density');
xlabel('Frequency Index'),
ylabel('Amplitude');

%%%%%--------------------------------------------------------
% % % % % Creating the Spectrogram
figure();
%subplot(3,1,3);
s = spectrogram(y);
spectrogram(y,25,'yaxis')


