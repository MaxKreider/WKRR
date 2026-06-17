%% setup

%mr clean
clc
clf
close all
clear


%% load the data

%do it
KS_data = load('ks_chaotic_data_50_trials.mat');

%timestep
dt = KS_data.dt;

%independent variables
x = KS_data.x;       


%% notational convenience

%loop parameters
noise_levels = [0.01, 0.05, 0.10, 0.20];
num_iterations = 50;

%break up trajectories 
N_train = 4500;
N_val = 5500;
N_test = 50000;

%time vectors
t_train = (0:N_train-1)*dt;
t_val   = (0:N_val-1)*dt;
t_test  = (0:N_test-1)*dt;

%data
udata = KS_data.udata_merged; 
u_all_trials = reshape(udata, 60000, num_iterations, 64);
u_all_trials = permute(u_all_trials, [1, 3, 2]);


%% Preallocation

%solution vectors: [time_steps, 3_dimensions, iteration_number]
data = cell(length(noise_levels),1);
for n = 1:4
    data{n}.u_train = zeros(N_train, 64, num_iterations);
    data{n}.u_train_true = zeros(N_train, 64, num_iterations);
    data{n}.u_val = zeros(N_val, 64, num_iterations);
    data{n}.u_val_true = zeros(N_val, 64, num_iterations);
    data{n}.u_test = zeros(N_test, 64, num_iterations);
    data{n}.u_test_true = zeros(N_test, 64, num_iterations);
    data{n}.nstrength = zeros(64,num_iterations);
end

%use the same testing indices
master_test_indices = randi([1, 47000], [500, 1]);


%% generate the data

%loop over all 100 trials
for i = 1:num_iterations
    fprintf('Generating trajectory %d/%d...\n', i, num_iterations);
    
    %extract the clean piece for this trial
    u_clean = u_all_trials(:,:,i);
    
    %loop over each noise level
    for n = 1:length(noise_levels)

        %different noise realizations
        rng(i*10 + n)

        %define noise intensity
        nlevel = noise_levels(n);
        
        %corrupt the data
        u_noisy = u_clean + randn(size(u_clean)) .* rms(u_clean) * nlevel;
                
        %form the training data
        data{n}.u_train(:,:,i) = u_noisy(1:N_train, :);
        data{n}.u_train_true(:,:,i) = u_clean(1:N_train, :);
        
        %form the validation data
        data{n}.u_val(:,:,i) = u_noisy(N_train+1:N_train+N_val, :);
        data{n}.u_val_true(:,:,i) = u_clean(N_train+1:N_train+N_val, :);
        
        %form the testing data
        data{n}.u_test(:,:,i) = u_noisy(N_train+N_val+1:N_train+N_val+N_test, :);
        data{n}.u_test_true(:,:,i) = u_clean(N_train+N_val+1:N_train+N_val+N_test, :);

        %save the noise strength
        data{n}.nstrength(:,i) = (rms(u_clean) * nlevel)';
        
    end
end

%% save the data

%file prefixes for saving
file_suffixes = {'01', '05', '10', '20'};

%loop over the noise levels
for n = 1:length(noise_levels)

    %notational nicety
    nlevel = noise_levels(n);
    fprintf('Saving file for %d%% noise level...\n', nlevel * 100);
    
    %extract arrays from struct to local workspace so save() names them correctly
    u_train = data{n}.u_train;
    u_train_true = data{n}.u_train_true;
    u_val = data{n}.u_val;
    u_val_true = data{n}.u_val_true;
    u_test = data{n}.u_test;
    u_test_true = data{n}.u_test_true;
    nstrength = data{n}.nstrength;
    
    %name the files properly
    filename = sprintf('KS_data_%s.mat', file_suffixes{n});
    
    %save
    save(filename, 'u_train', 'u_train_true', 't_train', 'u_val', 'u_val_true', 't_val', 'u_test', 'u_test_true', 't_test', 'dt', 'master_test_indices', 'nlevel','nstrength','x');
end