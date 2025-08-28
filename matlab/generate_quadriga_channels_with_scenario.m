function H_all = generate_quadriga_channels_with_scenario(n, p, positions, B, desired_mean, desired_std, seed, scenario, num_samples, output_dir)
%GENERATE_QUADRIGA_CHANNELS_WITH_SCENARIO Simulates SISO frequency-domain channel
% matrices for a MANET using QuaDRiGa, over multiple samples.
%
% INPUTS:
%   A             - [n x n] symmetric adjacency matrix (0/1).
%   positions     - [3 x n] matrix of 3D node positions.
%   B             - Number of frequency bins.
%   desired_mean  - Desired mean of output complex samples.
%   desired_std   - Desired std of output complex samples.
%   seed          - Integer seed for reproducibility.
%   scenario      - String. QuaDRiGa propagation scenario name.
%   num_samples   - Integer. Number of independent realizations to simulate.
%   output_dir    - String. Directory to save output.
%
% OUTPUT:
%   Saves 'H_all.mat' in output_dir:
%     H_all: [num_samples x B x n x n] complex SISO channel tensor

    % -------------------- Setup --------------------
  
    H_all = zeros(num_samples, B, n, n);  % Final tensor

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % Global simulation parameters
    simParams = qd_simulation_parameters;
    simParams.center_frequency = 2.4e9;
    simParams.sample_density = 2;
    simParams.use_absolute_delays = true;

    % -------------------- Simulate all samples --------------------
    for sample_idx = 1:num_samples
        rng(seed + sample_idx);
        % ===== Adjacency Matrix (bidirectional MANET graph) =====
        A = triu(rand(n) < p, 1);
        A = A + A';
        isolated = find(sum(A,2) == 0);
        for k = 1:numel(isolated)
            i = isolated(k);
            j = randi([1 n]);
            while j == i
                j = randi([1 n]);
            end
            A(i,j) = 1;
            A(j,i) = 1;
        end
        fprintf("Generating MANET sample %d/%d...\n", sample_idx, num_samples);

        % Local tensor for this sample: [n x n x B]
        H_sample = zeros(n, n, B);

        for i = 1:n
            for j = (i+1):n
                if A(i, j) == 1
                    % Set up layout
                    layout = qd_layout(simParams);
                    layout.no_tx = 1;
                    layout.no_rx = 1;
                    layout.tx_position = positions(:, i);
                    layout.rx_position = positions(:, j);

                    tx_array = qd_arrayant('omni'); tx_array.no_elements = 1;
                    rx_array = qd_arrayant('omni'); rx_array.no_elements = 1;
                    layout.tx_array = tx_array;
                    layout.rx_array = rx_array;

                    layout.set_scenario(scenario);

                    % Generate channels
                    channels = layout.get_channels();
                    [~, ~, H_f] = channel_to_freq(channels, B, 1e7);  % [1x1xB]

                    % Normalize
                    H = H_f(:);
                    H = H - mean(H);
                    H = H / std(H);
                    H = H * desired_std + desired_mean;
                    H_f = reshape(H, [1, 1, B]);

                    % Assign symmetrically
                    H_sample(i, j, :) = H_f;
                    H_sample(j, i, :) = H_f;
                end
            end
        end

        % Store sample in global tensor
        H_sample = permute(H_sample, [3, 1, 2]);  % [B x n x n]
        H_all(sample_idx, :, :, :) = H_sample;
    end

    % -------------------- Save --------------------

end