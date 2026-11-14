% GENERATE_R2_TEST_SETS  Fresh, independent QuaDRiGa test sets for the R2 revision.
%
% Generates the three R2 test sets in one run (edit RUNS to change them):
%   1) Main benchmarks, |V| = 10 (F1, F3-F5): 500 networks, p = 0.3 (the training density)
%   2) Density study,   |V| = 10            : 100 networks for each p in {0.1, ..., 0.5}
%   3) Main benchmarks, |V| = 6  (F2)       : 500 networks, p = 0.3
%
% Same channel model and node layout as the training data (quadriga_demo.m):
%   * scenario 5G-ALLSTAR_DenseUrban, B frequency bins, per-link normalization;
%   * the node layout of the training data, drawn as in quadriga_demo.m (rng(48312), x, y uniform
%     in a 100 x 100 m block, 20% of the nodes at 3 m height), shared by all networks of a set.
% New ER topologies and new channel realizations are drawn for every test network.
%
% Every network is CONNECTED (as assumed by the system model): the ER adjacency that
% generate_quadriga_channels_with_scenario.m will draw is reproduced first, and seeds yielding a
% disconnected graph are skipped before any QuaDRiGa computation. After generation, each network
% is re-checked; a mismatch between the prediction and the generator stops the run with an error.
%
% Independence from training: the training files used seed 48312 with per-network RNG state
% rng(48312 + idx), idx = 1..2000 (states 48313..50312). Here every network uses the state
% rng(SEED + 100000*ip + attempt + 1) with SEED >= 31337, i.e., >= 131339, so no network can
% coincide with a training network. The seeds also differ between the three sets.
%
% Output (MAT v7.3, loadable by mat73 in Python):
%   H_all [S x B x n x n] complex, p_vec [S x 1] (ER p of each network), pos_all [S x 3 x n]

%% -------------------- Path setup --------------------
thisFile = mfilename('fullpath');
repoRoot = fileparts(thisFile);  % Folder containing quadriga_demo.m

addpath('C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET\Matlab');
addpath(repoRoot);
addpath(genpath(fullfile(repoRoot, 'quadriga_src')));
rehash;

%% -------------------- Parameters --------------------
OUTPUT_DIR          = "C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET\Quadriga channels";
B                   = 6;
SCENARIO            = '5G-ALLSTAR_DenseUrban';   % same as the training data (quadriga_demo.m)
TRAINING_LAYOUT_SEED = 48312;                    % node layout of the training data

%            name     n    P_LIST                     SAMPLES_PER_P  SEED
RUNS = { ...
    "Test",  10, 0.3,                       500,           90210; ...   % 1) main, |V| = 10
    "Test",  10, [0.1 0.2 0.3 0.4 0.5],     100,           31337; ...   % 2) density study
    "Test",   6, 0.3,                       500,           90210};      % 3) main, multicast |V| = 6

for r = 1:size(RUNS, 1)
    generate_set(RUNS{r, 1}, RUNS{r, 2}, RUNS{r, 3}, RUNS{r, 4}, RUNS{r, 5}, ...
                 B, SCENARIO, TRAINING_LAYOUT_SEED, OUTPUT_DIR);
end

% =====================================================================================

function generate_set(SET_NAME, n, P_LIST, SAMPLES_PER_P, SEED, B, SCENARIO, LAYOUT_SEED, OUTPUT_DIR)
S = numel(P_LIST) * SAMPLES_PER_P;
H_all   = complex(zeros(S, B, n, n));
p_vec   = zeros(S, 1);
pos_all = zeros(S, 3, n);

% Training layout: same draw as quadriga_demo.m (rng(seed) followed by the position draw).
rng(LAYOUT_SEED);
positions = draw_positions(n);

s = 0;
for ip = 1:numel(P_LIST)
    p = P_LIST(ip);
    attempt = 0;
    rejected = 0;
    for k = 1:SAMPLES_PER_P
        s = s + 1;
        % Draw seeds until the ER adjacency is connected (checked before calling QuaDRiGa).
        while true
            attempt = attempt + 1;
            net_seed = SEED + 100000 * ip + attempt;
            if is_connected_adjacency(predict_adjacency(n, p, net_seed + 1))
                break;
            end
            rejected = rejected + 1;
        end
        % One network per call; the generator reseeds with rng(net_seed + 1) before drawing it.
        H_one = generate_quadriga_channels_with_scenario( ...
            n, p, positions, B, 0, 1, net_seed, SCENARIO, 1, OUTPUT_DIR);
        if ~is_connected_adjacency(squeeze(sum(abs(H_one(1, :, :, :)), 2)) > 0)
            error("Generated network is not connected; adjacency prediction is out of sync with the generator.");
        end
        H_all(s, :, :, :) = H_one(1, :, :, :);
        p_vec(s) = p;
        pos_all(s, :, :) = positions;
        fprintf("[n=%d] p = %.1f | network %d/%d (total %d/%d)\n", n, p, k, SAMPLES_PER_P, s, S);
    end
    fprintf("[n=%d] p = %.1f : %d disconnected draws rejected\n", n, p, rejected);
end

% Shuffle so every contiguous block mixes all densities; p_vec keeps each network's p.
rng(SEED + 1);
perm    = randperm(S);
H_all   = H_all(perm, :, :, :);
p_vec   = p_vec(perm);
pos_all = pos_all(perm, :, :);

p_tag = strjoin(arrayfun(@(x) strrep(sprintf('%.1f', x), '.', 'p'), P_LIST, 'UniformOutput', false), '_');
fname = sprintf('%s_H_seed_%d_p_%s_n_%d_B_%d_samples_%d.mat', SET_NAME, SEED, p_tag, n, B, S);
save(fullfile(OUTPUT_DIR, fname), 'H_all', 'p_vec', 'pos_all', '-v7.3');
fprintf("Saved %s\n", fullfile(OUTPUT_DIR, fname));
end

function A = predict_adjacency(n, p, rng_seed)
% Same ER draw (and isolated-node fix) as generate_quadriga_channels_with_scenario.m,
% which calls rng(seed + sample_idx) with sample_idx = 1 before drawing the adjacency.
rng(rng_seed);
A = triu(rand(n) < p, 1);
A = A + A.';
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
end

function tf = is_connected_adjacency(A)
A = double(A > 0);
A(1:size(A, 1) + 1:end) = 0;
tf = max(conncomp(graph(A))) == 1;
end

function positions = draw_positions(n)
% Same draw as quadriga_demo.m: 100 x 100 m urban block; ~20% of the nodes elevated (second floor).
positions = [rand(1, n) * 100; ...
             rand(1, n) * 100; ...
             1.5 + (rand(1, n) < 0.2) * 1.5];
end
