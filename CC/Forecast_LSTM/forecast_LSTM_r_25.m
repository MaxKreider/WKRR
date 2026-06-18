%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Challenge 2.1: Time-Forecasting of a turbulent cavity flow
%
% Given the initial 70 snapshots of the planar velocity field
% from a turbulent cavity flow, forecast the subsequent 30 snapshots of the
% sequence.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ML method: Long short-term memory (LSTM) networks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script performs five steps:
%
% (1) Reads the training data and test input data
% (2) Obtains POD expansion coefficients for training and test data
% (3) Trains an LSTM network on the normalized POD coefficients
% (4) Performs closed-loop forecasting for all test episodes
% (5) Writes the result file, which serves as an example of the file
%     participants are expected to send to the challenge POC
%
% Datasets, challenge details and submission guidelines are
% maintained on the website:
% https://fluids-challenge.engin.umich.edu/
%
% 1/8/2026, initial version, OTS <oschmidt@ucsd.edu>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%this is my modification of the code as described above


%% clear stuff

%mr clean
clc
clf
close all
clear 


%% load the data

%load data
ux_train        = h5read("Challenge2_1_train.h5",'/ux');
uy_train        = h5read("Challenge2_1_train.h5",'/uy');
ux_test_input   = h5read("Challenge2_1_test_input.h5",'/ux');
uy_test_input   = h5read("Challenge2_1_test_input.h5",'/uy');
x           = h5read("Challenge2_1_grid.h5",'/x');
y           = h5read("Challenge2_1_grid.h5",'/y');
dt          = h5read('Challenge2_1_parameters.h5','/dt');


%% setup

%set rng for reproducibility
rng(1)

%hindcast/forecast lengths (length it sees / length it forecasts)
nt_hind     = 40;
nt_fore     = 30;
nt_episode  = nt_hind+nt_fore;

%how many testing segments (32)
n_sample_test   = size(ux_test_input,4);

%dimensions and 2D grid
[nx,ny,nt_train] = size(ux_train);
ndof        = 2*nx*ny;
[xx,yy]     = meshgrid(x,y);
n_sample_train  = floor(nt_train/nt_episode);
if n_sample_train < 1
    error('Training data does not contain a full hindcast/forecast block.');
end

%initialize forecasting solutions for u and v
ux_forecast = zeros(nx,ny,nt_fore,n_sample_test);
uy_forecast = zeros(nx,ny,nt_fore,n_sample_test);

%model parameters
nt_test     = n_sample_test*nt_episode;
n_modes     = 25;
n_lstmLayer = 128;
n_lstmEpoch = 200;
n_miniBatch = 26; % ~7 iterations/epoch for 182 (= n_sample_train) training samples

%mean subtraction applied to training and test data
ux_mean     = mean(ux_train,3);
uy_mean     = mean(uy_train,3);
ux_std = std(ux_train,[],3); 
uy_std = std(uy_train,[],3);
ux_train    = (ux_train - ux_mean)./ux_std;
uy_train    = (uy_train - uy_mean)./uy_std;
ux_test_input    = (ux_test_input - ux_mean)./ux_std;
uy_test_input    = (uy_test_input - uy_mean)./uy_std;

%training data
Q_train     = [reshape(ux_train(:,:,1:nt_train),[nx*ny nt_train]); reshape(uy_train(:,:,1:nt_train),[nx*ny nt_train])];

%testing data
Q_test_input = [reshape(ux_test_input,[nx*ny nt_test]);reshape(uy_test_input,[nx*ny nt_test])];


%% POD on training data

%take svd of the training data matrix
[Phi,Sigma,V] = svds(Q_train,n_modes,'largest');

%look at time coefficients
A_train       = Sigma*V';    

%compute power of singular values
lambda        = diag(Sigma).^ 2; 

%show spectrum of retained modes
figure
bar(1:n_modes, lambda, 'facecolor', 'b', 'EdgeColor','none'); hold on
set(gca,'YScale','log')
title('POD spectrum')
xlabel('mode'); 
ylabel('$\lambda$','Interpreter','latex')


%% Obtain POD expansion coefficients for test data

%product test data onto POD basis
A_test      = Phi' * Q_test_input;


%% POD reconstruction demo for a single hindcast window

%expectation management: LSTM can only reconstruct the flow field as well as the POD basis allows

%model (out of 32) to display
i_display   = 1;

%select first the ith model (one trajectory of length 70)
Q_hind_disp = [reshape(ux_test_input(:,:,1:nt_hind,i_display),[nx*ny nt_hind]); ...
               reshape(uy_test_input(:,:,1:nt_hind,i_display),[nx*ny nt_hind])];

%project onto POD space
A_hind_disp = Phi' * Q_hind_disp;

%undo the projection
Q_rec_hind  = Phi * A_hind_disp;

%time at which to view the reconstruction
t_i = 1;

%form the explicit u and v reconstruction and truth
ux_rec      = reshape(Q_rec_hind(1:ndof/2,     t_i),nx,ny);
uy_rec      = reshape(Q_rec_hind(ndof/2+1:end, t_i),nx,ny);
ux_true     = reshape(Q_hind_disp(1:ndof/2,    t_i),nx,ny);
uy_true     = reshape(Q_hind_disp(ndof/2+1:end,t_i),nx,ny);

%plot the truth and reconstruction u and v
figure(1)
tl_hind = tiledlayout(2,2,'TileSpacing','compact');
title(tl_hind, sprintf('POD reconstruction for episode %d, step %d', i_display, t_i), 'Interpreter','latex')
nexttile; pcolor(xx,yy,ux_true);  colorbar; shading interp; clim(50*[-1 1])
nexttile; pcolor(xx,yy,uy_true);  colorbar; shading interp; clim(50*[-1 1])
nexttile; pcolor(xx,yy,ux_rec);   colorbar; shading interp; clim(50*[-1 1])
nexttile; pcolor(xx,yy,uy_rec);   colorbar; shading interp; clim(50*[-1 1])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% POD-LSTM closed-loop forecasting %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%build hind windows for training
A_hind_train    = cell(n_sample_train,1);
for i=1:n_sample_train

    %sequential hind/forecast blocks
    offset      = (i-1)*nt_episode; 

    %select indices for the hindcast
    t_idx       = (1:nt_hind) + offset;
    A_hind_train{i}   = A_train(:,t_idx);
end

%build hind windows for testing
A_hind_test    = cell(n_sample_test,1);
for i=1:n_sample_test

    %NEW - modify the offset here so that the testing blocks see only the first 40 (not 70) timesteps 
    offset      = (i-1)*nt_episode;  

    %the index stuff remains the same
    t_idx       = (1:nt_hind) + offset;
    A_hind_test{i}  = A_test(:,t_idx);
end

%normalize modal coefficients using training statistics
A_mu   = mean(A_train,2);
A_std  = std(A_train,0,2) + eps;

%build one-step input/target sequences from hind segments (normalized)
A1 = cell(n_sample_train,1);
A2 = cell(n_sample_train,1);
for n = 1:n_sample_train
    A_this  = (A_hind_train{n} - A_mu) ./ A_std; % [n_modes x nt_hind]
    A1{n}   = A_this(:,1:end-1).';               % [T x n_modes]
    A2{n}   = A_this(:,2:end  ).';
end

%define and train LSTM (using trainNetwork)
layers = [
    sequenceInputLayer(n_modes)
    lstmLayer(n_lstmLayer,'OutputMode','sequence')
    fullyConnectedLayer(n_modes)
    regressionLayer
];
options = trainingOptions('adam', ...
    'MaxEpochs', n_lstmEpoch, ...
    'Shuffle', 'every-epoch', ...
    'MiniBatchSize', n_miniBatch, ...
    'Verbose', false);

%convert to features-first for trainNetwork
XTrain = cellfun(@transpose,A1,'UniformOutput',false); % [n_modes x T]
YTrain = cellfun(@transpose,A2,'UniformOutput',false);
net = trainNetwork(XTrain,YTrain,layers,options);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Training forecast error analysis       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%initialize
f_error_train = figure(2);
e_nmse  = zeros(nt_fore,n_sample_train);
A_fore_train  = zeros(n_sample_train,nt_fore,n_modes);

%over each of the training segments
for n = 1:n_sample_train

    %start and forecasting indices for each training segment forecast
    idx_start = (n-1)*nt_episode + 1;
    idx_fore  = (idx_start+nt_hind):(idx_start+nt_hind+nt_fore-1);

    %truth over forecast time indices (last 30 timesteps)
    Q_fore_true = [reshape(ux_train(:,:,idx_fore),[nx*ny nt_fore]);reshape(uy_train(:,:,idx_fore),[nx*ny nt_fore])];

    %training input data (first 70 timesteps)
    A_in = (A_hind_train{n} - A_mu) ./ A_std; % [n_modes x nt_hind]

    %do the forecast
    [~, A_fore] = forecastClosedLoop(net, A_in, nt_fore);

    %undo the mode normalization
    A_fore = A_fore .* A_std.' + A_mu.'; % [nt_fore x n_modes]

    %store the forecast
    A_fore_train(n,:,:) = A_fore;

    %recover the original coordinates
    Q_rec = Phi * A_fore.'; % [ndof x nt_fore]

    %compute the nmse error
    for ti = 1:nt_fore
        q_true = Q_fore_true(:,ti);
        q_hat  = Q_rec(:,ti);
        denom = mean(q_true.^2) + eps;           % normalize by true energy
        e_nmse(ti,n) = mean((q_true - q_hat).^2) / denom;
    end

    % %plot it
    % figure(f_error_train)
    % set(f_error_train,'Name',sprintf('%d/%d (training)',n,n_sample_train))
    % plot(e_nmse_train(:,n),'LineWidth',0.5,'Color',0.8*[1 1 1]); hold on
    % %drawnow
end

% %do figure stuff
% figure(f_error_train);
% hold on;
% 
% rgb = lines;
% e_nmse_mean = mean(e_nmse_train,2);
% e_nmse_std  = std(e_nmse_train,0,2);
% h_band = fill([1:nt_fore nt_fore:-1:1],[e_nmse_mean+e_nmse_std; flipud(e_nmse_mean-e_nmse_std)], ...
%              rgb(2,:),'EdgeColor','none','FaceAlpha',0.1);
% h_mean = plot(e_nmse_mean,'r-','LineWidth',2,'Color',rgb(2,:));
% xlabel('$\Delta t$','Interpreter','latex'); ylabel('$e_\mathrm{NMSE}$','Interpreter','latex')
% title('LSTM forecast NMSE across forecast horizon (training data)','Interpreter','latex')
% if ~isempty(h_band) && ~isempty(h_mean)
%     leg = legend([h_band h_mean], 'std','mean','Location','best');
%     set(leg,'Interpreter','latex');
% end
% train_nmse_mean = mean(e_nmse_train(:));
% train_nmse_std  = std(e_nmse_train(:));
% fprintf('Training NMSE (mean +/- std over all blocks/horizons): %.3e +/- %.3e\n', ...
%     train_nmse_mean, train_nmse_std);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% POD-LSTM closed-loop forecasting %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Forecast test episodes (but now with ground truth available!)

%initialize
A_fore_test = zeros(n_sample_test,nt_fore,n_modes);

%over each of the 32 segments
for n = 1:n_sample_test

    %form the input (over the testing data) and normalize it
    A_in = (A_hind_test{n} - A_mu) ./ A_std; % [n_modes x nt_hind]

    %do the forecast
    [~, A_fore] = forecastClosedLoop(net, A_in, nt_fore);

    %unnormalize the forecast
    A_fore = A_fore .* A_std.' + A_mu.'; % [nt_fore x n_modes]

    %store it
    A_fore_test(n,:,:) = A_fore;

    %reconstruct in original coordinates
    Q_rec = Phi * A_fore.'; % [ndof x nt_fore]

    %save the forecasts in original coordinates as grids for plotting
    for ti = 1:nt_fore
        q_hat = Q_rec(:,ti);
        ux_forecast(:,:,ti,n) = reshape(q_hat(1:nx*ny),[nx ny]);
        uy_forecast(:,:,ti,n) = reshape(q_hat(nx*ny+1:end),[nx ny]);
    end
end

%quick sanity plot of coefficients for first test episode
figure(3)
plot(1:nt_hind, A_hind_test{1}.')
hold on
plot(nt_hind+1:nt_hind+nt_fore, squeeze(A_fore_test(1,:,:)))
title('Hind- and forecast POD coefficients for episode 1','Interpreter','latex')
xlabel('time step')
ylabel('coefficient value')


%% extract corresponding truth and true POD reconstruction

%initialize
A_truth_test = zeros(n_sample_test,ndof,nt_fore);
A_POD_truth_test = zeros(n_sample_test,ndof,nt_fore);

%re-add the means
test_u = ux_test_input.*ux_std + ux_mean;
test_v = uy_test_input.*uy_std + uy_mean;

%over each of the 32 segments
for n = 1:n_sample_test

    %indices
    idx_start_d = (n-1)*nt_episode + 1;
    idx_fore_d  = (idx_start_d+nt_hind):(idx_start_d+nt_hind+nt_fore-1);

    %extract truth
    Q_true_disp = [reshape(test_u(:,:,idx_fore_d),[nx*ny nt_fore]); reshape(test_v(:,:,idx_fore_d),[nx*ny nt_fore])];
    
    %save
    A_truth_test(n,:,:) = Q_true_disp;

    %stuff to do POD
    Q_POD_disp = [reshape(ux_test_input(:,:,idx_fore_d),[nx*ny nt_fore]); reshape(uy_test_input(:,:,idx_fore_d),[nx*ny nt_fore])];

    %convert to POD
    temp = Phi' * Q_POD_disp;
    Q_POD_true_disp = Phi * temp;

    %save
    A_POD_truth_test(n,:,:) = Q_POD_true_disp;
end


%% visualize

%timestep 
timeslice = 10;

%model
model_idx = 5;

%extract truth
truth_u = reshape(A_truth_test(model_idx,1:nx*ny,timeslice),[nx ny]);
truth_v = reshape(A_truth_test(model_idx,nx*ny+1:end,timeslice),[nx ny]);

%extract POD
POD_truth_u = reshape(A_POD_truth_test(model_idx,1:nx*ny,timeslice),[nx ny]).*ux_std + ux_mean;
POD_truth_v = reshape(A_POD_truth_test(model_idx,nx*ny+1:end,timeslice),[nx ny]).*uy_std + uy_mean;

%plot the truth and POD
figure(1)

subplot(3,2,1)
imagesc(x,y,truth_u)
title('true u')
clim([-100 250])
colorbar
axis image off
colormap(turbo)
set(gca, 'YDir', 'normal')

subplot(3,2,2)
imagesc(x,y,truth_v)
title('true v')
clim([-25 25])
colorbar
axis image off
colormap(turbo)
set(gca, 'YDir', 'normal')

subplot(3,2,3)
imagesc(x,y,POD_truth_u)
title('POD true u')
clim([-100 250])
colorbar
axis image off
colormap(turbo)
set(gca, 'YDir', 'normal')

subplot(3,2,4)
imagesc(x,y,POD_truth_v)
title('POD true v')
clim([-25 25])
colorbar
axis image off
colormap(turbo)
set(gca, 'YDir', 'normal')

%extract the recon
LSTM_u = reshape(ux_forecast(:,:,timeslice,model_idx),[nx ny]).*ux_std + ux_mean;
LSTM_v = reshape(uy_forecast(:,:,timeslice,model_idx),[nx ny]).*uy_std + uy_mean;

%plot the truth and POD
subplot(3,2,5)
imagesc(x,y,LSTM_u)
title('LSTM u')
clim([-100 250])
colorbar
axis image off
colormap(turbo)
set(gca, 'YDir', 'normal')

subplot(3,2,6)
imagesc(x,y,LSTM_v)
title('LSTM v')
clim([-25 25])
colorbar
axis image off
colormap(turbo)
set(gca, 'YDir', 'normal')


%% output LSTM plotting data at appropriate timeslices

%define timeslices
my_time = [1, 2, 5, 15, 30];

%model choice
my_model = 5;

%define LSTM solution
LSTM_sol_u = zeros(length(my_time),nx,ny);
LSTM_sol_v = zeros(length(my_time),nx,ny);

LSTM_POD_sol_u = zeros(length(my_time),nx,ny);
LSTM_POD_sol_v = zeros(length(my_time),nx,ny);

LSTM_truth_sol_u = zeros(length(my_time),nx,ny);
LSTM_truth_sol_v = zeros(length(my_time),nx,ny);

%loop over all timeslices
for count = 1:length(my_time)

    %extract the recon
    LSTM_u = reshape(ux_forecast(:,:,my_time(count),model_idx),[nx ny]).*ux_std + ux_mean;
    LSTM_v = reshape(uy_forecast(:,:,my_time(count),model_idx),[nx ny]).*uy_std + uy_mean;

    %save it
    LSTM_sol_u(count,:,:) = LSTM_u;
    LSTM_sol_v(count,:,:) = LSTM_v;

    %extract POD
    POD_u = reshape(A_POD_truth_test(model_idx,1:nx*ny,my_time(count)),[nx ny]).*ux_std + ux_mean;
    POD_v = reshape(A_POD_truth_test(model_idx,nx*ny+1:end,my_time(count)),[nx ny]).*uy_std + uy_mean;

    %save it
    LSTM_POD_sol_u(count,:,:) = POD_u;
    LSTM_POD_sol_v(count,:,:) = POD_v;

    %extract truth
    truth_u = reshape(A_truth_test(model_idx,1:nx*ny,my_time(count)),[nx ny]);
    truth_v = reshape(A_truth_test(model_idx,nx*ny+1:end,my_time(count)),[nx ny]);

    %save it
    LSTM_truth_sol_u(count,:,:) = truth_u;
    LSTM_truth_sol_v(count,:,:) = truth_v;

end


%% produce error plot

%error
e_nmse = zeros(30,32);

%loop over all timeslices
for n=1:32

    %extract forecast (with mean subtraction!)
    u_temp_pod = ux_forecast(:,:,:,n).*ux_std;
    v_temp_pod = uy_forecast(:,:,:,n).*uy_std ;
    QQQ = [reshape(u_temp_pod,[nx*ny,30]); reshape(v_temp_pod,[nx*ny,30])];

    %indices
    idx_start_d = (n-1)*nt_episode + 1;
    idx_fore_d  = (idx_start_d+nt_hind):(idx_start_d+nt_hind+nt_fore-1);

    %extract truth (with mean subtraction)
    u_temp_true = ux_test_input(:,:,41:70,n).*ux_std;
    v_temp_true = uy_test_input(:,:,41:70,n).*uy_std;
    QQQ_true = [reshape(u_temp_true,[nx*ny,30]); reshape(v_temp_true,[nx*ny,30])];

    %compute the nmse error
    for ti = 1:nt_fore
        q_true = QQQ_true(:,ti);
        q_hat  = QQQ(:,ti);
        denom = mean(q_true.^2) + eps;           % normalize by true energy
        e_nmse(ti,n) = mean((q_true - q_hat).^2) / denom;
    end

    %plot
    figure(8)
    hold on
    plot(1:30,e_nmse(:,n),'color',[0.6 0.6 0.6])
    
end

%highlight chosen trajectory
plot(1:30,e_nmse(:,my_model),'--','color',[.72 .45 .2],'linewidth',2)

%compute mean
my_mean = mean(e_nmse,2);
plot(1:30,my_mean,'color',[.72 .45 .2],'linewidth',3)
xlim([0 30])
ylim([0 2])
box on
set(gca,'fontsize',14)
xlabel('\Deltat')
ylabel('Error')
title('LSTM')

%compute std
my_std  = std(e_nmse,0,2);
h_band = fill([1:nt_fore nt_fore:-1:1],[my_mean + my_std; flipud(my_mean - my_std)],[.72 .45 .2],'EdgeColor','none','FaceAlpha',0.2);

%save it
save('lstm_results_r_25.mat','LSTM_sol_u','LSTM_sol_v','LSTM_POD_sol_u','LSTM_POD_sol_v','LSTM_truth_sol_v','LSTM_truth_sol_u','e_nmse')


%% auxiliary function

function [net_out, A_fore] = forecastClosedLoop(net_in, A_norm, nt_fore)
% Helper: advance LSTM in closed loop for nt_fore steps
    net_out = resetState(net_in);
    X0 = A_norm(:,1:end-1).';           % [T x n_modes]
    X0_t = X0.';                        % [n_modes x T]
    [net_out, ~]        = predictAndUpdateState(net_out, X0_t);
    [net_out, y_prev]   = predictAndUpdateState(net_out, X0_t(:,end));

    n_modes = size(A_norm,1);
    A_fore = zeros(nt_fore,n_modes);
    A_fore(1,:) = y_prev(:).';
    for t = 2:nt_fore
        [net_out, y_t]  = predictAndUpdateState(net_out, A_fore(t-1,:).');
        A_fore(t,:) = y_t(:).';
    end
end