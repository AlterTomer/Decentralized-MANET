function [H_time, delay, H_freq] = channel_to_freq(h_channel, B, Fs)
%CHANNEL_TO_FREQ Converts time-domain channel to frequency-domain
%
% Inputs:
%   h_channel : qd_channel object
%   B         : Number of frequency bins
%   Fs        : Sampling frequency (Hz)
%
% Outputs:
%   H_time    : Time-domain channel coefficients
%   delay     : Delay of each path (in seconds)
%   H_freq    : Frequency-domain channel (CFR)

    % Get time-domain response
    H_time = h_channel.coeff;
    delay = h_channel.delay;  % delay in seconds

    % Frequency vector [Hz]
    f = linspace(0, Fs, B);  % size: 1 x B

    % Dimensions
    [N_rx, N_tx, N_paths] = size(H_time);
    N_f = B;

    % Initialize frequency domain matrix
    H_freq = zeros(N_rx, N_tx, N_f);

    for rx = 1:N_rx
        for tx = 1:N_tx
            h = squeeze(H_time(rx, tx, :));  % N_paths x 1
            H = zeros(1, N_f);
            for k = 1:N_paths
                H = H + h(k) * exp(-1j * 2 * pi * f * delay(k));
            end
            H_freq(rx, tx, :) = H;
        end
    end
end
