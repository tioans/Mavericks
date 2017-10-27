function Hd = BTW_LP
%BTW_LP Returns a discrete-time filter object.

% MATLAB Code
% Generated by MATLAB(R) 9.2 and the DSP System Toolbox 9.4.
% Generated on: 02-Oct-2017 17:39:47

% Butterworth Lowpass filter designed using FDESIGN.LOWPASS.

% All frequency values are in kHz.
Fs = 48000;  % Sampling Frequency

Fpass = 9600;        % Passband Frequency
Fstop = 12000;       % Stopband Frequency
Apass = 1;           % Passband Ripple (dB)
Astop = 80;          % Stopband Attenuation (dB)
match = 'stopband';  % Band to match exactly

% Construct an FDESIGN object and call its BUTTER method.
h  = fdesign.lowpass(Fpass, Fstop, Apass, Astop, Fs);
Hd = design(h, 'butter', 'MatchExactly', match);

% [EOF]