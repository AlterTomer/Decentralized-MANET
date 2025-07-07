% RUN_QUADRIGA_DEMO: Simulate realistic MANET SISO channel matrices
% using QuaDRiGa for a fixed urban microcell LOS scenario.
% addpath(genpath("C:\Users\User\Desktop\MS.c\Ph.D\Distributed OFDMA MANETs\Matlab\QuaDRiGa-main")); 
% savepath;

% ===== Parameters =====
n = 15;                  % Number of nodes
B = 10;                 % Number of frequency bands
seed = 123;             % Random seed
scenario = '3GPP_3D_UMi_LOS';  % QuaDRiGa built-in scenario
output_dir = 'quadriga_demo_output';
desired_mean = 0;
desired_std = 1;
num_samples = 1000;
% ===== Node Positions (realistic street-level layout in meters) =====
% X, Y: position in a 100x100m urban block
% Z: height in meters (ground devices = 1.5m, 2 devices on 2nd floor = 3m)
% Auto-generate positions for n nodes
positions = [ ...
    rand(1, n) * 100;        % X in [0, 100] meters
    rand(1, n) * 100;        % Y in [0, 100] meters
    1.5 + (rand(1, n) < 0.2) * 1.5  % Z: 80% ground (1.5m), 20% elevated (3.0m)
];

% ===== Adjacency Matrix (bidirectional MANET graph) =====
p = 0.3;
A = triu(rand(n) < p, 1);
A = A + A';

% ===== Call Simulation =====
generate_quadriga_channels_with_scenario(A, positions, B, desired_mean, desired_std, seed, scenario, num_samples, output_dir);

% ===== Load and Inspect Output =====
load(fullfile(output_dir, 'H_all.mat'));
disp(['Loaded H_all of size ', mat2str(size(H_all))]);

% Example: Plot magnitude response between node 1 and node 2
% figure;
% plot(abs(squeeze(H_all(1,2,:))), 'LineWidth', 1.5);
% xlabel('Frequency Bin');
% ylabel('|H_{1,2}(f)|');
% title('Channel Magnitude between Node 1 and 2');
% grid on;