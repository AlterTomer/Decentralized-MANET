function [H_all, p_per_sample] = generate_quadriga_channels_with_scenario( ...
    n, p, positions, B, desired_mean, desired_std, seed, scenario, ...
    num_samples, output_dir)
%GENERATE_QUADRIGA_CHANNELS_WITH_SCENARIO Simulate MANET SISO channels.
%
%   [H_ALL, P_PER_SAMPLE] = GENERATE_QUADRIGA_CHANNELS_WITH_SCENARIO(...)
%   generates frequency-domain channel matrices using QuaDRiGa.
%
% INPUTS:
%   n             - Number of nodes.
%   p             - Scalar or vector of Erdos-Renyi edge probabilities.
%                   The generated samples are divided equally among all
%                   entries of p. If num_samples is not divisible by
%                   numel(p), it is rounded upward to the nearest divisible
%                   integer.
%   positions     - [3 x n] matrix of node positions.
%   B             - Number of frequency bins/bands.
%   desired_mean  - Desired mean of each nonzero complex channel vector.
%   desired_std   - Desired standard deviation of each nonzero channel vector.
%   seed          - Integer seed for reproducibility.
%   scenario      - QuaDRiGa propagation-scenario name.
%   num_samples   - Requested number of channel realizations.
%   output_dir    - Output directory; it is created if it does not exist.
%
% OUTPUTS:
%   H_all         - [num_samples_actual x B x n x n] complex channel tensor.
%   p_per_sample  - [num_samples_actual x 1] vector containing the p value
%                   used to generate each sample.
%
% The samples for each p value are stored in contiguous, equally sized
% groups. For example, p = [0.1 0.2 0.3] and num_samples = 2000 produce
% 2001 samples: 667 samples for each p value.

    % -------------------- Validate inputs --------------------
    validateattributes(n, {'numeric'}, {'scalar', 'integer', 'positive'}, ...
        mfilename, 'n');
    validateattributes(p, {'numeric'}, ...
        {'vector', 'real', 'finite', 'nonempty', '>=', 0, '<=', 1}, ...
        mfilename, 'p');
    validateattributes(positions, {'numeric'}, {'size', [3, n], 'finite'}, ...
        mfilename, 'positions');
    validateattributes(B, {'numeric'}, {'scalar', 'integer', 'positive'}, ...
        mfilename, 'B');
    validateattributes(num_samples, {'numeric'}, ...
        {'scalar', 'integer', 'positive'}, mfilename, 'num_samples');

    p = p(:);  % Use a column vector internally.
    num_p_values = numel(p);

    % Round upward so every p value receives exactly the same number of samples.
    num_samples_requested = num_samples;
    samples_per_p = ceil(num_samples_requested / num_p_values);
    num_samples_actual = samples_per_p * num_p_values;

    if num_samples_actual ~= num_samples_requested
        fprintf(['Requested %d samples, but %d p values require an equal split. ' ...
                 'Generating %d samples instead (%d per p value).\n'], ...
            num_samples_requested, num_p_values, num_samples_actual, samples_per_p);
    end

    % Contiguous groups: [p(1), ..., p(1), p(2), ..., p(end)].
    p_per_sample = repelem(p, samples_per_p);

    % -------------------- Setup --------------------
    H_all = complex(zeros(num_samples_actual, B, n, n));

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    simParams = qd_simulation_parameters;
    simParams.center_frequency = 2.4e9;
    simParams.sample_density = 2;
    simParams.use_absolute_delays = true;

    % -------------------- Simulate all samples --------------------
    for sample_idx = 1:num_samples_actual
        rng(seed + sample_idx);
        current_p = p_per_sample(sample_idx);

        % Bidirectional Erdos-Renyi MANET graph.
        A = triu(rand(n) < current_p, 1);
        A = A + A.';

        % Ensure that no node is isolated.
        isolated = find(sum(A, 2) == 0);
        for k = 1:numel(isolated)
            i = isolated(k);
            j = randi(n);
            while j == i
                j = randi(n);
            end
            A(i, j) = 1;
            A(j, i) = 1;
        end

        fprintf('Generating MANET sample %d/%d with p = %.6g...\n', ...
            sample_idx, num_samples_actual, current_p);

        % Local tensor for this sample: [n x n x B].
        H_sample = complex(zeros(n, n, B));

        for i = 1:n
            for j = (i + 1):n
                if A(i, j) == 1
                    % Set up a single-link QuaDRiGa layout.
                    layout = qd_layout(simParams);
                    layout.no_tx = 1;
                    layout.no_rx = 1;
                    layout.tx_position = positions(:, i);
                    layout.rx_position = positions(:, j);

                    tx_array = qd_arrayant('omni');
                    tx_array.no_elements = 1;
                    rx_array = qd_arrayant('omni');
                    rx_array.no_elements = 1;
                    layout.tx_array = tx_array;
                    layout.rx_array = rx_array;

                    layout.set_scenario(scenario);

                    % Generate the B-bin frequency-domain channel.
                    channels = layout.get_channels();
                    [~, ~, H_f] = channel_to_freq(channels, B, 1e7);  % [1 x 1 x B]

                    % Normalize the nonzero link channel.
                    H = H_f(:);
                    H = H - mean(H);
                    H_std = std(H);
                    if H_std > 0
                        H = H / H_std;
                    end
                    H = H * desired_std + desired_mean;
                    H_f = reshape(H, [1, 1, B]);

                    % Reciprocal/bidirectional channel assignment.
                    H_sample(i, j, :) = H_f;
                    H_sample(j, i, :) = H_f;
                end
            end
        end

        % Store this sample as [B x n x n].
        H_all(sample_idx, :, :, :) = permute(H_sample, [3, 1, 2]);
    end
end
