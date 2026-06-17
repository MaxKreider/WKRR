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

%select a model
nnn = 16;

tic

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

v = P0*z;

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

u_a = u_a';
P = [mean(P1_ensemble,2)';mean(P2_ensemble,2)';mean(P3_ensemble,2)'];


%% predictive reservoir run to forecast validation trajectory u_valid

%testing data
u_testing = u_test_true(:,:,nnn);

%sample random initial condition from the testing data
idx = 9000;

%extract the trajectory
u_test_traj = u_testing(idx:idx+40000,:);
t_test_traj = t_test(idx:idx+40000) - t_test(idx);

%initialize reservoir solution
r_valid = zeros(Dr, size(u_test_traj, 1));

%initial condition
u0 = u_test_traj(1,:)';

%precompute standard deviation for VPT computations
sigma = std(u_test_traj);

%run the reservoir
r_valid(:,1) = tanh(w_in*u0+b_in);
for j=2:length(u_test_traj)
    x_pred = w_in*P*r_valid(:,j-1) + b_in;
    r_valid(:,j) = tanh(x_pred);
end
v_valid = P*r_valid;
v_valid = v_valid';
k_sol = v_valid;

toc


%% Plot Time-Series (First 10 Seconds)

%variables for components
comp_names = {'x', 'y', 'z'};

%create the figure
figure(10);
set(gcf, 'Position', [150, 150, 1200, 600]); 
tiledlayout(3, 1, 'TileSpacing', 'Compact');

%plot each component
for d = 1:3
  
    %formatting stuff
    nexttile;
    hold on;
    grid on;
    set(gca, 'FontSize', 14);
    box on
    xlim([0 9])

    %axis labels
    ylabel(comp_names{d});
    if d == 3
        xlabel('Lyapunov time \Lambdat'); 
    end
    
    %plot
    plot(t_test_traj(1:1000)*Lambda, u_test_traj(1:1000, d), 'k-', 'LineWidth', 3);
    plot(t_test_traj(1:1000)*Lambda, k_sol(1:1000, d), '--', 'Color', [0.4, 0.75, 0.4], 'LineWidth', 2.5);
end


%% Plot 3D Phase Portrait

%create figure
figure(11);
set(gcf, 'Position', [410.6,35.4,799.9999999999999,677.6]);

%how many timepoints from the end of the trajectory to plot
ii = 5000;

%plot
hold on;
plot3(u_test_traj(end-ii:end,1), u_test_traj(end-ii:end,2), u_test_traj(end-ii:end,3), 'k-', 'LineWidth', 4, 'Color', [0 0 0 0.4]);
plot3(k_sol(end-ii:end,1), k_sol(end-ii:end,2), k_sol(end-ii:end,3), '--', 'Color', [0.4, 0.75, 0.4], 'LineWidth', 3);

%formatting stuff
grid on;
view(45, 20);
xlabel('x'); 
ylabel('y'); 
zlabel('z');
set(gca, 'FontSize', 14);
axis tight;
box on
set(gca,'fontsize',15)


%% create histograms

%make the figure
figure(12);
set(gcf, 'Position', [100, 100, 1200, 400]);
tiledlayout(1, 3, 'TileSpacing', 'Compact');

%loop over each component
for d = 1:3

    %go to next tile
    nexttile;
    hold on;
    
    %define consistent bins for this dimension
    [~, edges] = histcounts(u_test_traj(:,d), 50);
    
    %compute densities
    p_true = histcounts(u_test_traj(:,d), edges, 'Normalization', 'pdf');
    q_model = histcounts(k_sol(:,d), edges, 'Normalization', 'pdf');
    
    %calculate center of bins for plotting
    bin_centers = edges(1:end-1) + diff(edges)/2;
    
    %plot WDMKRR as a filled area
    fill_color = [0.4, 0.75, 0.4];
    area(bin_centers, q_model, 'FaceColor', fill_color, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    
    %plot Truth as a solid thick outline
    plot(bin_centers, p_true, 'k', 'LineWidth', 2);
        
    %formatting
    if d == 1
        xlabel('x');
        ylabel('Probability Density'); 
    elseif d == 2
        xlabel('y')
    else
        xlabel('z')
    end
    grid on;
    box on;
    set(gca,'fontsize',14)
    ylim([0 0.08])
    
    if d == 3
        legend('Prediction', 'Truth', 'Location', 'best');
    end

end





