% RUN_QUADRIGA_DEMO: Simulate realistic MANET SISO channel matrices with QuaDRiGa
% (scenario 5G-ALLSTAR_DenseUrban, one fixed node layout shared by all networks).
% With the parameters below it generates the TRAINING set of the paper:
%   n = 10 -> Train_H_seed_48312_n_10_B_6_samples_2000.mat (unicast, F3-F5)
%   n = 6  -> the multicast (F2) training set
% Edit output_dir and the addpath line below to your local folders.

%% -------------------- Path setup --------------------
thisFile = mfilename('fullpath');
repoRoot = fileparts(thisFile);  % Folder containing quadriga_demo.m

addpath('C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET\Matlab');
addpath(repoRoot);
addpath(genpath(fullfile(repoRoot, 'quadriga_src')));
rehash;

%% -------------------- Parameters --------------------
n = 10;                              % Number of nodes
B = 6;                               % Number of frequency bands
seed = 48312;                        % Random seed
scenario = '5G-ALLSTAR_DenseUrban';  % QuaDRiGa built-in scenario
output_dir = ['C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET\' ...
              'Quadriga channels\'];

desired_mean = 0;
desired_std = 1;
num_samples = 2000;                 % Requested number of samples
p = [0.3];                % Graph edge probabilities

%% -------------------- Node positions --------------------
% X and Y are in a 100-by-100 m urban block.
% Z is 1.5 m for 80%% of nodes and 3.0 m for the remaining 20%%.
rng(seed);
positions = [ ...
    rand(1, n) * 100; ...
    rand(1, n) * 100; ...
    1.5 + (rand(1, n) < 0.2) * 1.5 ...
];

%% -------------------- Generate channels --------------------
[H_all, p_per_sample] = generate_quadriga_channels_with_scenario( ...
    n, p, positions, B, desired_mean, desired_std, seed, scenario, ...
    num_samples, output_dir);

num_samples_actual = size(H_all, 1);

% Create a filesystem-safe label such as "0p1_0p2_0p3".
p_strings = arrayfun(@(x) strrep(sprintf('%g', x), '.', 'p'), ...
    p, 'UniformOutput', false);
p_label = strjoin(p_strings, '_');

output_filename = sprintf( ...
    'Train_H_seed_%d_n_%d_B_%d_samples_%d.mat', ...
    seed, n, B, num_samples_actual);
output_path = fullfile(output_dir, output_filename);

%% Save p and p_per_sample so every channel realization can be traced to
% the graph-density value used to generate it.
save(output_path, 'H_all', 'p', 'p_per_sample', ...
    'num_samples', 'num_samples_actual', 'positions', '-v7.3');

fprintf('Saved H_all with size [%s] to:\n%s\n', ...
    num2str(size(H_all)), output_path);
fprintf('Samples per p value: %d\n', num_samples_actual / numel(p));

% Example: inspect one B-bin channel vector between nodes 1 and 2.
% sample_idx = 1;
% figure;
% stem(abs(squeeze(H_all(sample_idx, :, 1, 2))), 'LineWidth', 1.5);
% xlabel('Frequency Band');
% ylabel('|H_{1,2}|');
% title(sprintf('Sample %d, p = %.2f', sample_idx, p_per_sample(sample_idx)));
% grid on;
