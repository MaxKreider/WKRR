%% setup

%mr clean
clc
clf
close all
clear

%VPT threshold
gamma = 0.5;

%lyapunov time
Lambda = 0.043;

%initialize cell arrays to store results
TrialResults(50).filter(1).test_vpts = [];


%% load the data

%load data
load('KS_filtered_data_05.mat')

%time
t_train = t_train_output;
t_val = t_val_output;
t_test = t_test_output;

%indices
test_indices = master_test_indices;

%polynomial tuning parameters
if nlevel == 0.01
    p = 7;
    L = 55*dt;
    overlap = 0.2;
elseif nlevel == 0.05
    p = 5;
    L = 80*dt;
    overlap = 0.2;
elseif nlevel == 0.1
    p = 5;
    L = 95*dt;
    overlap = 0.2;
elseif nlevel == 0.2
    p = 6;
    L = 115*dt;
    overlap = 0.2;
end


%% create W matrix

%create W
W = create_W(t_train(1:end-1),dt,p,L,overlap);


%% loop over all filters and all 50 trajectories

%choose trajectory of interest
nnn = 16;

%keep track of how far we are
fprintf('\n>>> Trial %d / 50 <<<\n', nnn);

%rng for validation
rng(1)

%training
X_train = u_train_filter_poly_output(1:end-1,:,nnn);
Y_train = u_train_filter_poly_output(2:end,:,nnn);

%transform RHS data
Y_train = W*Y_train;

%validation
u_validation = u_val_filter_poly_output(:,:,nnn);

%testing data
u_test_true = u_test_true_output(:,:,nnn);

%for the parallel loop
D_sq_const = parallel.pool.Constant(-pdist2(X_train,X_train).^2);
X_sq_const = parallel.pool.Constant(sum(X_train.^2,2)');


%% construct initial guess for epsilon

%define range for eta
eta = logspace(-5,5,100);

%define L parameter
LLL = max(max(pdist2(X_train,X_train)));

%compute the S and the d
S = zeros(1,length(eta));
r = zeros(1,length(eta));

%loop
for i=1:length(eta)

    %define rho matrix
    rho = exp(D_sq_const.Value/(LLL^2*eta(i)));

    %define S
    S(i) = 1/(size(rho,1)*size(rho,2))*sum(sum(rho));

    %define r
    r(i) = sum(sum(rho.*pdist2(X_train,X_train).^2));

end

%the d function
d_eta = 2/(size(rho,1)*size(rho,2)*LLL^2).*r./(S.*eta);

%find the max and avoid spurious peaks :(
[pks, locs] = findpeaks(d_eta);
if ~isempty(locs)
    %pick the right-most peak (the larger epsilon/eta scale)
    idx = locs(end);
else
    [~, idx] = max(d_eta);
end
eta_star = eta(idx);

%find epsilon
epsilon = 250*LLL^2*eta_star;


%% construct initial guess for lambda

%first, construct the rbf gaussian kernel
K_hat = exp(-pdist2(X_train,X_train).^2/(4*epsilon));

%diagonlize
lambda = eig(K_hat);

%find lambda
lambda = abs(min(lambda)); 

%scale it
if nlevel ~= 0.01
    lambda = lambda * 1000;
end


%% prepare for validation

%epsilons
DeltaEps = 1e-3/10;
epsilon_mesh = logspace(log10(epsilon*DeltaEps),log10(epsilon/DeltaEps),20);

%lambdas
DeltaLambda = 1e-5;
lambda_mesh = logspace(log10(lambda),log10(lambda/DeltaLambda),20);

%solution matrix for storing computing VPTs for each parameter pair
param_heat_map_mean = zeros(length(epsilon_mesh),length(lambda_mesh));


%% validate

%do a coarse search, then a refined search
N_search_passes = 2;

%break validation into multiple segments
Nv = 200;

%go loop over the passes
for pass = 1:N_search_passes

    %number of validation trajectories
    if pass == 1
        N_segments = 20;
    else
        N_segments = 30;
    end
    val_start_indices = randi([1 length(u_validation)-Nv], [1 N_segments]);

    tic
    %do the loop
    parfor iii = 1:length(epsilon_mesh)

        %temporary row storage for heat plot
        temp_row_mean = zeros(1, length(lambda_mesh));


        %% solve for the normalizations of the kernel q and q_hat

        %first, construct the rbf gaussian kernel
        K_hat = (exp(D_sq_const.Value/(4*epsilon_mesh(iii))));

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
                u_val_segment = u_validation(start_idx : start_idx + Nv, :);
                t_val_segment = t_val(start_idx : start_idx + Nv) - t_val(start_idx);

                %estimated kernel solution
                k_sol = (zeros(size(u_val_segment)));

                %initial condition
                x_current = (u_val_segment(1,:));
                k_sol(1,:) = x_current;

                %break step
                divergence_step = Nv + 1;

                %precompute standard deviation for vpt computations
                sigma = std(u_val_segment);

                %loop it
                for i=1:(size(u_val_segment, 1) - 1)

                    %iterate the model
                    k_sol(i+1,:) = kernel_func(x_current,X_train,epsilon_mesh(iii),alpha,X_sq_const.Value);

                    %do a running error check and break when the model diverges
                    temp = sqrt(sum(((u_val_segment(i+1,:) - k_sol(i+1,:))./sigma).^2,2)/size(u_val_segment,2));
                    if temp >= gamma
                        divergence_step = i;
                        break;
                    end

                    %update for next step
                    x_current = k_sol(i+1,:);
                end


                %% compute the VPT

                % Record the VPT for this trajectory
                vpt_accumulator(kkk) = Lambda * (t_val_segment(divergence_step));

            end

            %update the parameter heat map
            temp_row_mean(jjj) = mean(vpt_accumulator);

        end

        %update heat map
        param_heat_map_mean(iii,:) = temp_row_mean;

    end
    toc

    %find best of this pass
    [Val_VPT, idx_robust] = max(param_heat_map_mean(:));
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
title('Coarse')
colormap parula

figure(2)
imagesc(log10(epsilon_mesh),log10(lambda_mesh),param_heat_map_mean.')
hold on
plot(log10(epsilon_final),log10(lambda_final),'ko','markersize',10,'markerfacecolor','m')
colorbar
xlabel('log(\epsilon)')
ylabel('log(\lambda)')
set(gca,'fontsize',15)
title('Fine')
colormap parula


%% form the validated kernel

%first, construct the rbf gaussian kernel
K_hat = (exp(-pdist2(X_train,X_train).^2/(4*epsilon_final)));

%transform coordinates
K_hat = W*K_hat*W';

%solve for alpha
warning('off','all')
alpha = W'*((K_hat + lambda_final*eye(size(K_hat)))\Y_train);


%% test over many trials

%sample random initial condition from the testing data
idx = 400;

%extract the trajectory
u_test_traj = u_test_true(idx:idx+3000,:);
t_test_traj = t_test(idx:idx+3000) - t_test(idx);

%initialize estimated kernel solution
k_sol = zeros(size(u_test_traj));

%initial condition
x_current = (u_test_traj(1,:));
k_sol(1,:) = x_current;

%break step
divergence_step = length(u_test_traj);

%precompute standard deviation for VPT computations
sigma = std(u_test_traj);
flag = 1;

tic
%generate kernel approximation of the data
for i=1:length(k_sol)-1

    %iterate the model
    k_sol(i+1,:) = kernel_func(x_current,X_train,epsilon_final,alpha,X_sq_const.Value);

    %do a running error check and break when the model diverges
    temp = sqrt(sum(((u_test_traj(i+1,:)- k_sol(i+1,:))./sigma).^2,2)/size(u_test_traj,2));
    if temp >= gamma && flag == 1
        divergence_step = i;
        flag = 0;
    end

    %update for next step
    x_current = k_sol(i+1,:);
end
toc

%update VPT
my_VPT = Lambda * t_test_traj(divergence_step)


%% visualize the truth and the reconstruction and the error

%define a custom colormap because I have no life and I can't think of anything better to do on a Saturday afternoon, isn't that sad :/
colors = [ 0.00, 0.00, 0.00;   %black
    0.20, 0.05, 0.35;          %dark purple
    0.55, 0.25, 0.70;          %light purple
    0.95, 0.50, 0.30;          %light orange
    1,    1,    0.1];          %orange

%define transition points for the colors to be used in a linear interpolation
transitions = [1, 30, 90, 180, 256];

%interpolate the heck out of those rascals
x_interp  = 1:256;
my_custom_cmap = interp1(transitions, colors, x_interp, 'linear');

%time and spatial coordinates
x_spatial = linspace(0, 22, 64);
time_index = 0:1400;

%plot it
figure(10)
imagesc(time_index*Lambda*dt,x_spatial,u_test_traj(time_index+1,:)')
colormap(my_custom_cmap);
colorbar
xlabel('Lyapunov time \Lambdat')
ylabel('x')
set(gca,'fontsize',15)
box on
axis square
title('Truth')
clim([-3.5 3.5]) 
xlim([0 4])

%plot it
figure(11)
imagesc(time_index*Lambda*dt,x_spatial,k_sol(time_index+1,:)')
colormap(my_custom_cmap);
colorbar
xlabel('Lyapunov time \Lambdat')
ylabel('x')
set(gca,'fontsize',15)
box on
axis square
title('WKRR')
clim([-3.5 3.5]) 
xline(my_VPT,'g','linewidth',4)
xlim([0 4])

%plot it
figure(12)
imagesc(time_index*Lambda*dt,x_spatial,abs(u_test_traj(time_index+1,:)'-k_sol(time_index+1,:)')./max(max(abs( u_test_traj(time_index+1,:)'))))
colormap(my_custom_cmap);
colorbar
xlabel('Lyapunov time \Lambdat')
ylabel('x')
set(gca,'fontsize',15)
box on
axis square
title('Error')
xline(my_VPT,'g','linewidth',4)
xlim([0 4])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% auxiliary function

function[u] = kernel_func(x_current,X,epsilon,alpha,B2)

%distance thing (for gpu - it's faster somehow)
A2 = sum(x_current.^2,2);
D = 2*(x_current*X') - A2 - B2;

%first, construct the rbf gaussian kernel
K_hat = exp(D/(4*epsilon));

%output
u = K_hat*alpha;

end

%linear transformation
function[W] = create_W(t_train,dt,p,L,overlap)

%initialize matrices
W = [];

%translation
h = overlap*L;

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