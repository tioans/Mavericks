filename = 'microphone-results.wav';
[y,Fs] = audioread('microphone-results.wav');
sound(y,Fs);