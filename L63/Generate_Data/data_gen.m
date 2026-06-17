%% setup

%loop parameters
noise_levels = [0.01, 0.05, 0.10, 0.20];
num_iterations = 100;

%timestep
dt = 0.01;

%system parameters
sigma = 10;
rho = 28;
beta = 8/3;

%total length of trajectory and number of samples
N_burn = 10000;
N_train = 2000 + 500;
N_val = 5000 + 500;
N_test = 50000;
N_tot = N_burn + N_train + N_val + N_test;

%time vectors
t_full = 0:dt:(N_tot-1)*dt;
t_train = 0:dt:(N_train-1)*dt;
t_val = 0:dt:(N_val-1)*dt;
t_test = 0:dt:(N_test-1)*dt;

%vector field and ODE options
F = @(t,u) [sigma*(u(2)-u(1)); u(1)*(rho-u(3))-u(2); u(1)*u(2)-beta*u(3)];
opts = odeset('RelTol',1e-10,'AbsTol',1e-12);


%% Preallocation

%solution vectors: [time_steps, 3_dimensions, iteration_number]
data = cell(4,1);
for n = 1:4
    data{n}.u_train = zeros(N_train, 3, num_iterations);
    data{n}.u_train_true = zeros(N_train, 3, num_iterations);
    data{n}.u_val = zeros(N_val, 3, num_iterations);
    data{n}.u_val_true = zeros(N_val, 3, num_iterations);
    data{n}.u_test = zeros(N_test, 3, num_iterations);
    data{n}.u_test_true = zeros(N_test, 3, num_iterations);
    data{n}.nstrength = zeros(3,num_iterations);
end

%use the same testing indices
master_test_indices = randi([1, 47000], [500, 1]);


%% generate the data

%loop over all 100 trials
for i = 1:num_iterations
    fprintf('Generating trajectory %d/%d...\n', i, num_iterations);

    %set random seed
    rng(i*10)
    
    %initial condition and solve
    x0 = randn(3,1);
    [~, u_clean] = ode113(F, t_full, x0, opts);
    
    %loop over each noise level
    for n = 1:length(noise_levels)

        %different noise realizations
        rng(i*10 + n)

        %define noise intensity
        nlevel = noise_levels(n);
        
        %corrupt the data
        u_noisy = u_clean + randn(size(u_clean)) .* rms(u_clean) * nlevel;
        
        %prune off the burn-in
        u_c_pruned = u_clean(N_burn+1:end, :);
        u_n_pruned = u_noisy(N_burn+1:end, :);
        
        %form the training data
        data{n}.u_train(:,:,i) = u_n_pruned(1:N_train, :);
        data{n}.u_train_true(:,:,i) = u_c_pruned(1:N_train, :);
        
        %form the validation data
        data{n}.u_val(:,:,i) = u_n_pruned(N_train+1:N_train+N_val, :);
        data{n}.u_val_true(:,:,i) = u_c_pruned(N_train+1:N_train+N_val, :);
        
        %form the testing data
        data{n}.u_test(:,:,i) = u_n_pruned(N_train+N_val+1:N_train+N_val+N_test, :);
        data{n}.u_test_true(:,:,i) = u_c_pruned(N_train+N_val+1:N_train+N_val+N_test, :);

        %save the noise strength
        data{n}.nstrength(:,i) = (rms(u_clean) * nlevel)';
        
    end
end

%% save the data

%file prefixes for saving
file_suffixes = {'01', '05', '10', '20'};

%loop over the noise levels
for n = 1:4

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
    filename = sprintf('L63_data_%s.mat', file_suffixes{n});
    
    %save
    save(filename, 'u_train', 'u_train_true', 't_train', 'u_val', 'u_val_true', 't_val', 'u_test', 'u_test_true', 't_test', 'dt', 'master_test_indices', 'nlevel','nstrength');
end