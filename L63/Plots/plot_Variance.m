%% setup

%mr clean
clc
clf
close all
clear

%loop parameters
noise_levels = 0.10;
num_iterations = 100;

%timestep
dt = 0.01;

%system parameters
sigma = 10;
rho = 28;
beta = 8/3;

%total length of trajectory and number of samples
N_burn = 10000;
N_train = 1000;
N_tot = N_burn + N_train;

%time vectors
t_full = 0:dt:(N_tot-1)*dt;
t_train = 0:dt:(N_train-1)*dt;
t_long = 0:dt:(2*N_train-1)*dt;

%vector field and ODE options
F = @(t,u) [sigma*(u(2)-u(1)); u(1)*(rho-u(3))-u(2); u(1)*u(2)-beta*u(3)];
opts = odeset('RelTol',1e-10,'AbsTol',1e-12);


%% choose polynomial parameters (p,L,...)

%panel (c)
p_range = 10;
L_range = 70 * dt;

% %panel (b)
% p_range = 4;
% L_range = 120 * dt;

% %panel (a)
% p_range = 12;
% L_range = 150 * dt;


%% loop over overlap

%overlap (h) range
overlap_range = 0.1:dt:0.35;


%% initialize stuff

%store the noise levels
nlevels = zeros(num_iterations,3);

%initialize error
total_error = zeros(num_iterations,length(overlap_range));
my_bias_est = zeros(num_iterations,length(overlap_range));
my_bias_fourier_est = zeros(num_iterations,length(overlap_range));
my_var_est = zeros(num_iterations,length(overlap_range));


%% generate data

%count variable
count = 0;

%initial condition and solve
x0 = [1 1 1];
[~, u_clean] = ode113(F, t_full, x0, opts);

%define noise
nlevels = rms(u_clean) * noise_levels;

%loop over overlap
for ov_idx = overlap_range

    %update count
    count = count + 1;

    %test function info
    p = p_range
    L = L_range
    overlap = ov_idx

    %loop over all trials
    for i = 1:num_iterations

        %set random seed
        rng(i*10)

        %corrupt the data
        u_noisy = u_clean + randn(size(u_clean)) .* nlevels;

        %prune off the burn-in
        u_c_pruned = u_clean(N_burn+1:end, :);
        u_n_pruned = u_noisy(N_burn+1:end, :);

        %form the training data
        u_train = u_n_pruned(1:N_train, :);
        u_train_true = u_c_pruned(1:N_train, :);


        %% filter

        %filter the data
        u_train_filter_poly = my_poly_filter(u_train,t_train,t_long,dt,p,L,overlap);


        %% total observed error (bias + variance)

        %observed total error
        total_error(i,count) = norm(u_train_true - u_train_filter_poly, 'fro')^2;


        %% observed bias

        %ppply filter to the truth
        u_true_filter_poly = my_poly_filter(u_train_true,t_train,t_long,dt,p,L,overlap);

        %calculate bias (true - reconstructed true)
        my_bias_est(i,count) = norm(u_train_true - u_true_filter_poly, 'fro')^2;


        %% variance

        %observed variance
        my_var_est(i,count) = total_error(i,count) - my_bias_est(i,count);


        %% fourier tail for the bias

        %compute coefficients
        my_fft = fftshift(fft(u_train_true))/sqrt(N_train);

        %sum them
        my_total_fft = zeros(length(my_fft),1);
        for j=1:3
            my_total_fft = my_total_fft + abs(my_fft(:,j)).^2;
        end

        %compute Kmax
        Kmax = (N_train)/(overlap/dt);

        %prune the fourier signal
        my_total_fft(500-round(Kmax/2):500) = 0;
        my_total_fft(500:500+round(Kmax/2)) = 0;

        %store it
        my_bias_fourier_est(i,count) = sum(my_total_fft);

    end
end


%% combine the errors

%take the averages
my_bias_est = mean(my_bias_est, 1);
my_bias_fourier_est = mean(my_bias_fourier_est, 1);
my_var_est = mean(my_var_est, 1);
total_error = mean(total_error, 1);


%% theoretical fit for variance, theoretical bound for bias

%define variance fit
total_sigma_sq = sum(nlevels.^2);
my_var_fit = (N_train)*dt*total_sigma_sq.*overlap_range.^(-1);

%define bias fit
my_bias_fit = overlap_range.^(2*(p+1));
my_bias_fit = my_bias_fit/my_bias_fit(1)*my_bias_est(1);


%% variance plot

%plot the variance
figure(1)
hold on
plot(overlap_range,my_var_est,'k-','linewidth',3)
plot(overlap_range,my_var_est,'k.','markersize',40)
plot(overlap_range,my_var_fit,'c-','linewidth',3)
set(gca,'fontsize',15)
box on
axis square
grid on
xlabel('h')
ylabel('Error')
legend('Observed','','Theory','Location','northeast')
title('Variance')


%% bias plot

%plot the bias
figure(2)
hold on
plot(overlap_range,my_bias_est,'k-','linewidth',3)
plot(overlap_range,my_bias_est,'k.','markersize',40)
plot(overlap_range,my_bias_fourier_est,'r-','linewidth',3)
set(gca,'fontsize',15)
box on
axis square
grid on
xlabel('h')
ylabel('Error')
legend('Observed','','Theory','Location','northwest')
title('Bias')



%% auxiliary functions

%%%%%%%%%%%%%%%%%%%
% filter the data
%%%%%%%%%%%%%%%%%%%
function[u] = my_poly_filter(u_train,t_train,t_long,dt,p,L,h)

%initialize matrices
W = [];

%how far can we translate?
Kmax = round((length(t_train))/h*dt);

%loop over appropriate k's
for k = 1:Kmax

    %shift index
    shift = h*(k-1);

    %store in W_psi
    row_temp = spline_poly(t_train,t_long,dt,L,p,0+shift,L+shift);
    W = [W; row_temp];

end

%for the reconstruction
Phi = W;

%normalize for trapezoidal integration
W = dt*W;
W(:,1) = W(:,1)/2;
W(:,end) = W(:,end)/2;

%gram matrix
G = W*Phi';

%do the reconstruction
temp = W*u_train;
temp = G\temp;
u = Phi'*temp;

end

%%%%%%%%%%%%%%%%%%%
%   define poly
%%%%%%%%%%%%%%%%%%%
function[u_output] = spline_poly(t,t_long,dt,L,p,a,b)

%define constant
C = 1/(p^p*p^p)*(2*p/(b-a))^(2*p);

%output polynomial
u = C*(t_long-a).^p.*(b-t_long).^p .* (t_long>=a & t_long<=b);

%determine how big the test function can be
shift_length = L/dt;

%compress
u_output = u(1:length(t));

%wrap only when needed
if b >= length(t)*dt
    u_output(1:shift_length) = u(length(t)+1:length(t)+shift_length);
end

end