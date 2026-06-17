%% setup

%mr clean
clc
clf
close all
clear

%loop parameters
noise_levels = 0.10;
num_iterations = 1;

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

%vector field and ODE options
F = @(t,u) [sigma*(u(2)-u(1)); u(1)*(rho-u(3))-u(2); u(1)*u(2)-beta*u(3)];
opts = odeset('RelTol',1e-10,'AbsTol',1e-12);


%% choose polynomial parameters (p,L,...)

%panel (a)
% p_range = 3;
% L_range = 10 * dt;

%panel (b)
% p_range = 5;
% L_range = 50 * dt;

%panel (c)
p_range = 8;
L_range = 60 * dt;


%% Preallocation

%solution vectors: [time_steps, 3_dimensions, iteration_number]
data = cell(4,1);
for n = 1:4
    data{n}.u_train = zeros(N_train, 3, num_iterations);
    data{n}.u_train_true = zeros(N_train, 3, num_iterations);
    data{n}.nstrength = zeros(3,num_iterations);
    data{n}.L2_error = zeros(3,num_iterations);
end


%% generate some data

%loop over all trials
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

        %save the noise strength
        data{n}.nstrength(:,i) = (rms(u_clean) * nlevel)';

    end
end


%% filter the data

%loop over all noise levels
for nnn = 1:length(noise_levels)

    %count variable
    count = 0;

    %overlap (h) range
    overlap_range = 0.2*(25:5:150)*dt;

    %initialize error
    total_error = zeros(num_iterations,length(overlap_range));
    my_err_vec_pred_bias = zeros(num_iterations,length(overlap_range));
    my_var_est = zeros(num_iterations,length(overlap_range));

    %do a brute force search over poly parameters
    for p_idx = p_range
        for L_idx = L_range
            for ov_idx = overlap_range

                %test function info
                p = p_idx
                L = L_idx
                overlap = ov_idx

                %update count
                count = count + 1;

                %loop over number of trials
                for qqq = 1:num_iterations

                    %% total error (bias + variance)

                    %define the training data appropriately
                    u_train_current = data{nnn}.u_train(:,:,qqq);
                    u_train_true_current = data{nnn}.u_train_true(:,:,qqq);
                    t_train_current = t_train;

                    %filter the data
                    u_train_filter_poly = my_poly_filter(u_train_current,t_train_current,dt,p,L,overlap);

                    %direct data
                    X_poly = u_train_filter_poly(1:end,:);
                    X_true = u_train_true_current(1:end,:);

                    %observed total error
                    L2_poly = norm(X_poly - X_true, 'fro')^2;
                    total_error(qqq,count) = L2_poly;


                    %% bias

                    %full unpruned truth
                    u_true_full = data{nnn}.u_train_true(:,:,qqq);

                    %ppply filter to the truth
                    u_true_filt_full = my_poly_filter(u_true_full,t_train,dt,p,L,overlap);

                    %prune it
                    X_true_filt = u_true_filt_full(1:end, :);

                    %calculate bias (true - reconstructed true)
                    Bias_sq = norm(X_true_filt - X_true, 'fro')^2;

                    %observed bias
                    my_err_vec_pred_bias(qqq,count) = Bias_sq;


                    %% variance

                    %observed variance
                    my_var_est(qqq,count) = total_error(qqq,count) - my_err_vec_pred_bias(qqq,count);

                end
            end
        end
    end
end


%% combine the errors

%take the averages
my_err_vec_pred_bias = mean(my_err_vec_pred_bias, 1);
my_var_est = mean(my_var_est, 1);
total_error = mean(total_error, 1);


%% theoretical fit for variance, theoretical bound for bias

%define variance fit
total_sigma_sq = sum(mean(data{1}.nstrength.^2,2));
my_var_fit = (N_train - L_range/dt)*dt*total_sigma_sq.*overlap_range.^(-1);

%define bias fit
my_bias_fit = overlap_range.^(2*(p+1));
my_bias_fit = my_bias_fit/my_bias_fit(1)*my_err_vec_pred_bias(1);


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
loglog(overlap_range,my_err_vec_pred_bias,'k-','linewidth',3)
loglog(overlap_range,my_err_vec_pred_bias,'k.','markersize',40)
loglog(overlap_range,my_bias_fit,'r-','linewidth',3)
set(gca,'fontsize',15)
box on
axis square
grid on
xlabel('h')
ylabel('Log Error')
set(gca,'Xscale','log','Yscale','log')
legend('Observed','','Theory','Location','northeast')
title('Bias')
ylim([10^4 10^6])


%% auxiliary functions

%linear transformation
function[u] = my_poly_filter(u_train,t_train,dt,p,L,overlap)

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

%polynomial
function[u] = spline_poly(t,p,a,b)

%define constant
C = 1/(p^p*p^p)*(2*p/(b-a))^(2*p);

%output polynomial
u = C*(t-a).^p.*(b-t).^p .* (t>=a & t<=b);

end