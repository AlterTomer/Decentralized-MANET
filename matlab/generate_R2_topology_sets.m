% GENERATE_R2_TOPOLOGY_SETS  Non-ER test topologies for the R2 generalization study.
%
% Two families of CONNECTED |V| = 10 MANETs, generated independently of the training data, for
% zero-shot evaluation of the MANET-GNN models trained on ER graphs with p = 0.3:
%
%   1) RGG - random geometric (unit-disk) graphs: a new layout per network, drawn as the training layout (100 x 100 m
%      urban block, ~20% elevated), and two nodes are linked iff their distance is at most R = 42.5 m.
%      R is set so that the hop structure matches the training graphs (connected ER, p = 0.3):
%      mean diameter ~4.0 vs. 4.1, and ~74% of many-to-many instances with all routes within 3 hops.
%   2) HUB - node-heterogeneous (hub/periphery) graphs: each node is a hub with probability 0.3
%      (weight 0.8) or a peripheral node (weight 0.1), and nodes i, j are linked with probability
%      min(1, w_i * w_j / 0.3) (Chung-Lu). The mean edge probability matches ER with p = 0.3, while the
%      node degrees are strongly non-uniform (within-graph degree std ~2.0 vs. ~1.2 for ER). HUB networks use
%      the node layout of the training data.
%
% Channels: per-link QuaDRiGa 5G-ALLSTAR_DenseUrban channels (as in the training data) with the same simulation parameters and the
% same per-link normalization as generate_quadriga_channels_with_scenario.m. Disconnected draws are
% rejected before any QuaDRiGa computation.
%
% Independence from training: training used RNG states 48313..50312; here every network uses RNG
% states >= 520001, so no network can coincide with a training network.
%
% Output (MAT v7.3, loadable by mat73 in Python), one file per family:
%   H_all [S x B x n x n] complex, A_all [S x n x n] logical, pos_all [S x 3 x n]

%% -------------------- Path setup --------------------
thisFile = mfilename('fullpath');
repoRoot = fileparts(thisFile);  % Folder containing quadriga_demo.m

addpath('C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET\Matlab');
addpath(repoRoot);
addpath(genpath(fullfile(repoRoot, 'quadriga_src')));
rehash;

%% -------------------- Parameters --------------------
OUTPUT_DIR = "C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET\Quadriga channels";
n          = 10;
B          = 6;
S          = 200;                 % networks per family
SCENARIO   = '5G-ALLSTAR_DenseUrban';   % same as the training data (quadriga_demo.m)
TRAINING_LAYOUT_SEED = 48312;          % node layout of the training data (used by the HUB family)

RGG_RADIUS = 42.5;                % [m] communication range of the unit-disk graph
HUB_FRAC   = 0.3;                 % probability that a node is a hub
HUB_W      = 0.8;                 % hub weight
LEAF_W     = 0.1;                 % peripheral-node weight
MEAN_P     = 0.3;                 % Chung-Lu normalization (= training ER edge probability)

%           family   SEED
FAMILIES = {"RGG",   520000; ...
            "HUB",   610000};

simParams = qd_simulation_parameters;
simParams.center_frequency = 2.4e9;
simParams.sample_density = 2;
simParams.use_absolute_delays = true;

% Training layout (same draw as quadriga_demo.m), used by the HUB family. The RGG family draws a new
% layout per network, since its links are defined by the node positions.
rng(TRAINING_LAYOUT_SEED);
training_positions = draw_positions(n);

for f = 1:size(FAMILIES, 1)
    family = FAMILIES{f, 1};
    SEED = FAMILIES{f, 2};
    H_all   = complex(zeros(S, B, n, n));
    A_all   = false(S, n, n);
    pos_all = zeros(S, 3, n);
    attempt = 0;
    rejected = 0;
    for s = 1:S
        % Draw (positions, topology) until the topology is connected.
        while true
            attempt = attempt + 1;
            rng(SEED + attempt);
            positions = draw_positions(n);
            switch family
                case "RGG"
                    D = squeeze(vecnorm(positions - permute(positions, [1 3 2]), 2, 1));  % [n x n]
                    A = D <= RGG_RADIUS;
                case "HUB"
                    positions = training_positions;
                    w = LEAF_W * ones(n, 1);
                    w(rand(n, 1) < HUB_FRAC) = HUB_W;
                    Pm = min(1, (w * w.') / MEAN_P);
                    A = triu(rand(n) < Pm, 1);
                    A = A | A.';
            end
            A(1:n + 1:end) = false;
            if is_connected_adjacency(A)
                break;
            end
            rejected = rejected + 1;
        end
        rng(SEED + attempt + 7919);       % reproducible QuaDRiGa randomness for this network
        H_all(s, :, :, :) = quadriga_for_adjacency(A, positions, B, SCENARIO, simParams);
        A_all(s, :, :) = A;
        pos_all(s, :, :) = positions;
        fprintf("[%s] network %d/%d (mean degree %.2f)\n", family, s, S, mean(sum(A, 2)));
    end
    fprintf("[%s] %d disconnected draws rejected\n", family, rejected);

    fname = sprintf('Test_H_%s_seed_%d_n_%d_B_%d_samples_%d.mat', family, SEED, n, B, S);
    save(fullfile(OUTPUT_DIR, fname), 'H_all', 'A_all', 'pos_all', '-v7.3');
    fprintf("Saved %s\n", fullfile(OUTPUT_DIR, fname));
end

% =====================================================================================

function H = quadriga_for_adjacency(A, positions, B, scenario, simParams)
% Per-link QuaDRiGa channels for a given adjacency; identical to the inner loop of
% generate_quadriga_channels_with_scenario.m (desired_mean = 0, desired_std = 1). Returns [B x n x n].
n = size(A, 1);
H_sample = complex(zeros(n, n, B));
for i = 1:n
    for j = (i + 1):n
        if A(i, j)
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

            channels = layout.get_channels();
            [~, ~, H_f] = channel_to_freq(channels, B, 1e7);  % [1 x 1 x B]

            h = H_f(:);
            h = h - mean(h);
            h_std = std(h);
            if h_std > 0
                h = h / h_std;
            end

            H_sample(i, j, :) = reshape(h, [1, 1, B]);
            H_sample(j, i, :) = reshape(h, [1, 1, B]);
        end
    end
end
H = permute(H_sample, [3, 1, 2]);
end

function tf = is_connected_adjacency(A)
A = double(A > 0);
A(1:size(A, 1) + 1:end) = 0;
tf = max(conncomp(graph(A))) == 1;
end

function positions = draw_positions(n)
% Street-level layout in a 100 x 100 m urban block; ~20% of the nodes elevated (second floor).
positions = [rand(1, n) * 100; ...
             rand(1, n) * 100; ...
             1.5 + (rand(1, n) < 0.2) * 1.5];
end
