%% setup

%mr clean
clc
clf
close all
clear

%data storage
TrialResults(100).filter(1).test_vpts = [];


%% load data

%load it
load('L63_data_20.mat')

%indices
test_indices = master_test_indices;

%notation
na = 1999; %length of training data segments
D = 3;

%random seed
rng(1)


%% RAFDA setup

%reservoir size
Dr = 300;

%input weights and input bias
fac_w = 0.005;
fac_b = 4;
beta = 0.001;

%size of ensemble M
M = 300;

%projections matrix S
w = ones(M,1)/M;
S = eye(M) - w*ones(1,M);

%inflation
delta_x = 1.0;
delta_p = 1.0;


%% Observations

%observation operator H: observing only every dobs'th oscillator
dobs = 1;
Ds = ceil(D/dobs);
if(dobs==1)
    Ds = D;
end

H = zeros(Ds,D);
for l=1:Ds
    H(l,(l-1)*dobs+1) = 1;
end


%% L2-forecast error

%vpt threshold
gamma = 0.3;

%lyapunov time
Lambda = 0.91;


%% Outer Loop

%number of models
N_models = 100;

%loop over number of models
for nnn=1:N_models

    tic

    %keep track of how far we are
    fprintf('\n>>> Trial %d / 100 <<<\n', nnn);

    %noise 
    eta = nstrength(:,nnn).^2;
    R = eta.*eye(Ds);
    
    %training data
    u = u_train_true(:,:,nnn);
    u = u(250:end-251,:)';
    u0 = u(:,1);
    
    %observations
    y = u_train(:,:,nnn);
    y = y(250:end-251,:)';


    %% Construct reservoir system

    %input weights and input bias
    w_in = fac_w*(-1 + 2*rand(Dr,D));
    b_in =  fac_b*(-1 + 2*rand(Dr,1));


    %% generate a prior P0 for the parameter matrix P
    
    % by doing linear regression on the observations
    % run reservoir
    y_in  = y(:,1:na);
    y_out = y(:,2:na+1);

    for j=1:na
        x = w_in*y_in(:,j) + b_in;
        r(:,j) = tanh(x);
    end

    % Ridge regression to compute output weight
    % minimize ||P r - u_out||
    z = r;
    P0 = y_out*z'*inv(z*z'+beta*eye(Dr));


    %% run reservoir-DA system

    u_a = zeros(na,D);
    P_a = zeros(na,D*Dr);

    fac1 = 1;
    fac2 = 1;

    P1_ensemble = fac1*P0(1,:)'*ones(1,M) + fac2*sqrt(var(var(P0)))*randn(Dr,M);
    P2_ensemble = fac1*P0(2,:)'*ones(1,M) + fac2*sqrt(var(var(P0)))*randn(Dr,M);
    P3_ensemble = fac1*P0(3,:)'*ones(1,M) + fac2*sqrt(var(var(P0)))*randn(Dr,M);
    x_ensemble = u0*ones(1,M) + sqrt(eta).*randn(3,M);

    for j=1:na
        % forecast model
        r_ensemble = tanh(w_in*x_ensemble+b_in*ones(1,M));
        x_ensemble(1,:) = sum(P1_ensemble.*r_ensemble,1);
        x_ensemble(2,:) = sum(P2_ensemble.*r_ensemble,1);
        x_ensemble(3,:) = sum(P3_ensemble.*r_ensemble,1);

        % analysis step
        Pxx = 1/M*x_ensemble*S*x_ensemble';
        I = H*x_ensemble + sqrt(eta).*randn(Ds,M)*S - y(:,j+1)*ones(1,M);

        % update parameters
        P1px = 1/M*P1_ensemble*S*x_ensemble(1,:)';
        P1_ensemble = P1_ensemble - P1px*inv(Pxx(1,1) + eta(1))*I(1,:);
        P2px = 1/M*P2_ensemble*S*x_ensemble(2,:)';
        P2_ensemble = P2_ensemble - P2px*inv(Pxx(2,2) + eta(2))*I(2,:);
        P3px = 1/M*P3_ensemble*S*x_ensemble(3,:)';
        P3_ensemble = P3_ensemble - P3px*inv(Pxx(3,3)+ eta(3))*I(3,:);

        % update states
        x_ensemble = x_ensemble - Pxx*H'*inv(H*Pxx*H' + eta.*eye(Ds))*I;

        % outputs as ensemble means
        u_a(j,:) = x_ensemble*w;
        P_a(j,1:Dr)        = P1_ensemble*w;
        P_a(j,Dr+1:2*Dr)   = P2_ensemble*w;
        P_a(j,2*Dr+1:3*Dr) = P3_ensemble*w;
    end


    %% read in converged regression matrix P to be used for forecasting

    %do it
    u_a = u_a';
    P = [mean(P1_ensemble,2)';mean(P2_ensemble,2)';mean(P3_ensemble,2)'];


    %% predictive reservoir run to forecast validation trajectory u_valid

    %number of trials
    N_trial = length(test_indices);

    %VPT vector
    VPT_vec = zeros(1,length(test_indices));

    %testing data
    u_testing = u_test_true(:,:,nnn);

    %loop over each trial
    parfor ppp = 1:N_trial

        %sample random initial condition from the testing data
        idx = test_indices(ppp);

        %extract the trajectory
        u_test_traj = u_testing(idx:idx+3000,:);
        t_test_traj = t_test(idx:idx+3000) - t_test(idx);

        %initialize reservoir solution
        r_valid = zeros(Dr, size(u_test_traj, 1));

        %initial condition
        u0 = u_test_traj(1,:)';

        %break step
        divergence_step = length(u_test_traj);

        %precompute standard deviation for VPT computations
        sigma = std(u_test_traj);

        %run the reservoir
        r_valid(:,1) = tanh(w_in*u0+b_in);
        for j=2:length(u_test_traj)
            x_pred = w_in*P*r_valid(:,j-1) + b_in;
            r_valid(:,j) = tanh(x_pred);
        end
        v_valid = P*r_valid;

        %generate kernel approximation of the data
        for j=1:length(u_test_traj)

            %do a running error check and break when the model diverges
            temp = sqrt(sum(((u_test_traj(j,:)- v_valid(:,j)')./sigma).^2,2)/size(u_test_traj,2));
            if temp >= gamma
                divergence_step = j;
                break;
            end
        end

        %update VPT
        VPT_vec(ppp) = Lambda * t_test_traj(divergence_step);

    end
    toc

    %store stats
    TrialResults(nnn).filter(1).test_vpts = VPT_vec;
    TrialResults(nnn).filter(1).mean_vpt  = mean(VPT_vec);
    TrialResults(nnn).filter(1).std_vpt   = std(VPT_vec);
    TrialResults(nnn).filter(1).max_vpt   = max(VPT_vec);
    TrialResults(nnn).filter(1).min_vpt   = min(VPT_vec);

    mean(VPT_vec)

end


%% save stuff

%save everything
save('L63_RAFDA_20.mat', 'TrialResults', '-v7.3');

