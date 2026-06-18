%% setup

%mr clean
clc
clf
close all
clear

%initialize cell arrays to store results
TrialResults(32).filter(1).test_vpts = [];


%% load the data

%load timestep
dt = h5read('Challenge2_1_parameters.h5', '/dt');

%spatial grid
grid_x = h5read('Challenge2_1_grid.h5', '/x');
grid_y = h5read('Challenge2_1_grid.h5', '/y');

%training data
u_train = h5read('Challenge2_1_train.h5', '/ux');
v_train = h5read('Challenge2_1_train.h5', '/uy');

%testing data
u = h5read('Challenge2_1_test_input.h5', '/ux');
v = h5read('Challenge2_1_test_input.h5', '/uy');

%POD stuff
load("POD_param_r_100.mat");


%% polynomial parameters

%polynomial tuning parameters
p = 5;
L = 10*dt;
overlap = 0.2*10*dt;


%% break data into training, validation, testing for one specific model

%model choice
model_index = 1;
nnn = 1;

%select a single testing trajectory
u_chosen = u(:,:,:,model_index);
v_chosen = v(:,:,:,model_index);

%form validation data
u_val = u_chosen(:,:,1:39);
v_val = v_chosen(:,:,1:39);

%get size for compression
[nx, ny, nt_train] = size(u_train);

%reshape the training data into (time x space) matrices
B_training = [reshape(u_train, [], nt_train)', reshape(v_train, [], nt_train)'];
t_train = (0:nt_train-1)*dt;

%reshape the validation data
u_validation = [reshape(u_val, [], 39)', reshape(v_val, [], 39)'];
t_val = (0:38)*dt;


%% create W

%create W matrix
W = create_W(t_train(1:end-1),dt,p,L,overlap);
size(W)


%% perform the POD

%form normalization mean and std
X_mean = [u_mean,v_mean];
X_std = [u_std,v_std];

%form training data by normalizing and POD projection and per mode, using a direct connection scheme
X = (B_training - X_mean) ./ X_std; 
X = X * V;
X = (X - A_mu) ./ A_std;
X_train = X(1:end-1,:);
Y_train = X(2:end,:);

%transform RHS data
Y_train = W*Y_train;

%validation
u_validation_pod = (u_validation - X_mean) ./ X_std;
u_validation_pod = u_validation_pod * V;
u_validation_pod = (u_validation_pod - A_mu) ./ A_std;


%% setup for parallel loop

%rng for validation
rng(1)

%for the parallel loop
D_sq_const = parallel.pool.Constant(-pdist2(X_train,X_train).^2);
X_sq_const = parallel.pool.Constant(sum(X_train.^2,2)');


%% prepare for validation

%epsilons
epsilon_mesh = logspace(2,4,20);

%lambdas
lambda_mesh = logspace(-16,-13,20);

%solution matrix for storing computing VPTs for each parameter pair
param_heat_map_mean = zeros(length(epsilon_mesh),length(lambda_mesh));


%% validate

%do a coarse search, then a refined search
N_search_passes = 2;

%break validation into multiple segments
Nv = 15;

%loop over the passes
for pass = 1:N_search_passes

    %number of validation trajectories
    if pass == 1
        N_segments = 3;
    elseif pass == 2
        N_segments = 5;
    end
    
    %select validation indices
    val_start_indices = randi([1 size(u_validation_pod,1)-Nv], [1 N_segments]);

    tic
    %do the loop
    parfor iii = 1:length(epsilon_mesh)

        %temporary row storage for heat plot
        temp_row_mean = zeros(1, length(lambda_mesh));


        %% solve for the normalizations of the kernel q and q_hat

        %first, construct the rbf gaussian kernel
        K_hat = (exp(D_sq_const.Value/(4*epsilon_mesh(iii))));

        %convenience of notation
        N = length(K_hat);

        %do sampling density bias normalization
        q = (sum(K_hat,1)/N);
        K_hat = (K_hat./q)./q';

        %do transition normalization
        q_hat = (sum(K_hat,1)/N);
        K_hat = (K_hat.*q_hat.^(-1/2)).*q_hat'.^(-1/2);

        %transform coordinates
        K_hat = W*K_hat*W';


        %% do the KRR

        %do the other loop
        for jjj = 1:length(lambda_mesh)

            %solve for alpha
            warning('off','all')
            alpha = W'*((K_hat + lambda_mesh(jjj)*eye(size(K_hat)))\Y_train);

            %to accumulate the mean vpt
            vpt_accumulator = zeros(1, length(val_start_indices));


            %% generate signal and compare with u_val in different regions

            %loop over each validation region
            for kkk = 1:length(val_start_indices)

                %define u_val segment
                start_idx = val_start_indices(kkk);
                u_val_segment = u_validation_pod(start_idx : start_idx + Nv, :);
                t_val_segment = t_val(start_idx : start_idx + Nv) - t_val(start_idx);

                %estimated kernel solution
                k_sol = zeros(size(u_val_segment));

                %initial condition
                x_current = (u_val_segment(1,:));
                k_sol(1,:) = x_current;

                %loop it
                for i=1:(size(k_sol, 1) - 1)

                    %iterate the model
                    k_sol(i+1,:) = kernel_func(x_current,X_train,N,q,q_hat,epsilon_mesh(iii),alpha,X_sq_const.Value);

                    %update for next step
                    x_current = k_sol(i+1,:);
                end


                %% compute the VPT

                % Record the VPT for this trajectory
                vpt_accumulator(kkk) = mean(mean((k_sol-u_val_segment).^2,2) ./ mean(u_val_segment.^2,2));

            end

            %update the parameter heat map
            temp_row_mean(jjj) = mean(vpt_accumulator);

        end

        %update heat map
        param_heat_map_mean(iii,:) = temp_row_mean;

    end
    toc

    %find best of this pass
    [Val_VPT, idx_robust] = min(param_heat_map_mean(:));
    [iii_best, jjj_best] = ind2sub(size(param_heat_map_mean), idx_robust);

    %record the values
    epsilon_best = epsilon_mesh(iii_best);
    lambda_best  = lambda_mesh(jjj_best);

    %update bounds for the "Fine" pass
    if pass == 1

        %store coarse data
        TrialResults(nnn).filter(1).coarse_grid = param_heat_map_mean;
        TrialResults(nnn).filter(1).coarse_params = [epsilon_best, lambda_best];
        TrialResults(nnn).filter(1).coarse_vpt = Val_VPT;
        TrialResults(nnn).filter(1).coarse_eps_mesh = epsilon_mesh;
        TrialResults(nnn).filter(1).coarse_lam_mesh = lambda_mesh;

        %zoom factor: shrink the search window by 10x centered on the winner
        zoom = 0.75;

        %update the mesh
        epsilon_mesh = logspace(log10(epsilon_best)-zoom, log10(epsilon_best)+zoom, 20);
        lambda_mesh = logspace(log10(lambda_best)-zoom, log10(lambda_best)+zoom, 20);

    else

        %store fine data
        TrialResults(nnn).filter(1).fine_grid = param_heat_map_mean;
        TrialResults(nnn).filter(1).fine_params = [epsilon_best, lambda_best];
        TrialResults(nnn).filter(1).fine_vpt = Val_VPT;
        TrialResults(nnn).filter(1).fine_eps_mesh = epsilon_mesh;
        TrialResults(nnn).filter(1).fine_lam_mesh = lambda_mesh;

        %final variables for testing
        epsilon_final = epsilon_best;
        lambda_final  = lambda_best;
    end
end

%look at the mean VPT heat map
figure(1)
imagesc(log10(TrialResults(nnn).filter(1).coarse_eps_mesh),log10(TrialResults(nnn).filter(1).coarse_lam_mesh),TrialResults(nnn).filter(1).coarse_grid.')
hold on
plot(log10(TrialResults(nnn).filter(1).coarse_params(1)),log10(TrialResults(nnn).filter(1).coarse_params(2)),'ko','markersize',10,'markerfacecolor','m')
colorbar
xlabel('log(\epsilon)')
ylabel('log(\lambda)')
set(gca,'fontsize',15)
title('VPT heat map')
colormap parula
title('Mean')

figure(2)
imagesc(log10(epsilon_mesh),log10(lambda_mesh),param_heat_map_mean.')
hold on
plot(log10(epsilon_final),log10(lambda_final),'ko','markersize',10,'markerfacecolor','m')
colorbar
xlabel('log(\epsilon)')
ylabel('log(\lambda)')
set(gca,'fontsize',15)
title('VPT heat map')
colormap parula
title('Mean')

%best param values
epsilon_final
lambda_final


%% form the validated kernel

%first, construct the rbf gaussian kernel
K_hat = (exp(D_sq_const.Value/(4*epsilon_final)));

%convenience of notation
N = length(K_hat);

%do sampling density bias normalization
q = (sum(K_hat,1)/N);
K_hat = (K_hat./q)./q';

%do transition normalization
q_hat = (sum(K_hat,1)/N);
K_hat = (K_hat.*q_hat.^(-1/2)).*(q_hat'.^(-1/2));

%transform coordinates
K_hat = W*K_hat*W';

%solve for alpha
warning('off','all')
alpha = W'*((K_hat + lambda_final*eye(size(K_hat)))\Y_train);


%% test over many trials

%loop over all 32 segments
tic
for nnn = 1:32

    %select a single testing trajectory
    u_chosen = u(:,:,:,nnn);
    v_chosen = v(:,:,:,nnn);

    %select indices over which to forecast
    u_test = u_chosen(:,:,40:end);
    v_test = v_chosen(:,:,40:end);

    %reshape the testing data
    u_testing = [reshape(u_test, [], 31)', reshape(v_test, [], 31)'];

    %normalize the testing data in POD space
    u_testing_pod = (u_testing - X_mean) ./ X_std;
    u_testing_pod = u_testing_pod * V;
    u_testing_pod = (u_testing_pod - A_mu) ./ A_std;

    %extract the trajectory
    u_test_traj = u_testing_pod;
    t_test_traj = (0:30)*dt;

    %initialize estimated kernel solution
    k_sol = zeros(size(u_test_traj));

    %initial condition
    x_current = (u_test_traj(1,:));
    k_sol(1,:) = x_current;

    %generate kernel approximation of the data
    for i=1:size(k_sol,1)-1

        %iterate the model
        k_sol(i+1,:) = kernel_func(x_current,X_train,N,q,q_hat,epsilon_final,alpha,X_sq_const.Value);

        %update for next step
        x_current = k_sol(i+1,:);
    end

    %update VPT
    VPT_vec = mean( mean((k_sol-u_test_traj).^2,2) ./ mean(u_test_traj.^2,2) );

    toc

    %store stats
    TrialResults(nnn).filter(1).test_vpts = VPT_vec;
    TrialResults(nnn).filter(1).mean_vpt  = mean(VPT_vec);
    TrialResults(nnn).filter(1).std_vpt   = std(VPT_vec);
    TrialResults(nnn).filter(1).max_vpt   = max(VPT_vec);
    TrialResults(nnn).filter(1).min_vpt   = min(VPT_vec);
    TrialResults(nnn).filter(1).ker_sol   = k_sol;
    TrialResults(nnn).filter(1).u_true    = u_testing;
    TrialResults(nnn).filter(1).u_pod    = u_testing_pod;

    mean(VPT_vec)

end


%% save the results

%save everything
save('weak_DM_results_r_100.mat', 'TrialResults', '-v7.3');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% auxiliary function

function[u] = kernel_func(x_current,X,N,q,q_hat,epsilon,alpha,B2)

%distance thing (for gpu - it's faster somehow)
A2 = sum(x_current.^2,2);
D = 2*(x_current*X') - A2 - B2;

%first, construct the rbf gaussian kernel
K_hat = exp(D/(4*epsilon));

%do sampling density bias normalization
q_new = sum(K_hat,2)/N;
K_hat = (K_hat ./ q_new) ./ q;

%do transition normalization
q_hat_new = sum(K_hat,2)/N;
K_hat = (K_hat .* (q_hat_new.^(-1/2))) .* (q_hat.^(-1/2));

%output
u = K_hat*alpha;

end

%linear transformation
function[W] = create_W(t_train,dt,p,L,overlap)

%initialize matrices
W = [];

%translation
h = overlap;

%how far can we translate?
Kmax = round((length(t_train)-L/dt)/h*dt);

%loop over appropriate k's
for k = 1:Kmax

    %shift index
    shift = h*(k-1);

    %store in W_psi
    row_temp = spline_poly(t_train,p,0+shift,L+shift);
    W = [W; row_temp];

end

%normalize for trapezoidal integration
W = dt*W;
W(:,1) = W(:,1)/2;
W(:,end) = W(:,end)/2;

end

%polynomial
function[u] = spline_poly(t,p,a,b)

%define constant
C = 1/(p^p*p^p)*(2*p/(b-a))^(2*p);

%output polynomial
u = C*(t-a).^p.*(b-t).^p .* (t>=a & t<=b);

end
