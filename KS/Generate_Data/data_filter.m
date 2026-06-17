%% setup

%mr clean
clc
clf
close all
clear

%num dimensions
num_dim = 64;

%num filters (na,wdenoise,poly)
num_filt = 3;

%solution vectors: [time_steps, 64_dimensions, iteration_number]
data = cell(4,1);
N_train_new = 4000;
N_val_new = 5000;
N_test_new = 50000;
num_iterations = 50;
for n = 1:4

    %training
    data{n}.u_train = zeros(N_train_new, num_dim, num_iterations);
    data{n}.u_train_filter_wdenoise = zeros(N_train_new, num_dim, num_iterations);
    data{n}.u_train_filter_poly = zeros(N_train_new, num_dim, num_iterations);
    data{n}.u_train_true = zeros(N_train_new, num_dim, num_iterations);

    %validation
    data{n}.u_val = zeros(N_val_new, num_dim, num_iterations);
    data{n}.u_val_filter_wdenoise = zeros(N_val_new, num_dim, num_iterations);
    data{n}.u_val_filter_poly = zeros(N_val_new, num_dim, num_iterations);
    data{n}.u_val_true = zeros(N_val_new, num_dim, num_iterations);

    %testing
    data{n}.u_test = zeros(N_test_new, num_dim, num_iterations);
    data{n}.u_test_true = zeros(N_test_new, num_dim, num_iterations);

    %error metrics
    data{n}.error_angle = zeros(num_filt,num_iterations);
    data{n}.error_rmse= zeros(num_filt,num_iterations);
    data{n}.error_LSD = zeros(num_filt,num_iterations);

end

%loop over all noise levels
for nnn = 1:4
    
    %load the data and polynomial parameters
    if nnn == 1

        %data
        load('KS_data_01.mat')

        %parameters
        p = 7; 
        L = 55*dt; 
        overlap = 0.2;

    elseif nnn == 2

        %data
        load('KS_data_05.mat')

        %parameters
        p = 5;  
        L = 80*dt; 
        overlap = 0.2;

    elseif nnn == 3

        %data
        load('KS_data_10.mat')

        %parameters
        p = 5;  
        L = 95*dt; 
        overlap = 0.2;

    elseif nnn == 4

        %data
        load('KS_data_20.mat')

        %parameters
        p = 6;  
        L = 115*dt; 
        overlap = 0.2;
    end
    
    %output progress
    fprintf('Processing noise level: %.2f...\n', nlevel);


    %% over each 50 trials

    %loop over number of trials
    for qqq = 1:num_iterations

        %output progress
        fprintf('Trial: %.0f\n', qqq);

        %define the training data appropriately
        u_train_current = u_train(1:4500,:,qqq);
        u_train_true_current = u_train_true(1:4500,:,qqq);
        t_train_current = t_train(1:4500);

        %define the validation data appropriately
        u_val_current = u_val(:,:,qqq);
        u_val_true_current = u_val_true(:,:,qqq);
        t_val_current = t_val;

        %define the testing data appropriately
        u_test_current = u_test(:,:,qqq);
        u_test_true_current = u_test_true(:,:,qqq);
        t_test_current = t_test;


        %% wdenoise (in time) filtering

        %define filtered training data
        u_train_filter_wdenoise = wdenoise(u_train_current,'Wavelet', 'sym12','DenoisingMethod', 'Bayes', 'ThresholdRule', 'Median');
        u_train_filter_wdenoise = u_train_filter_wdenoise(250:end-251,:);

        %define filtered validation data
        u_val_filter_wdenoise = wdenoise(u_val_current,'Wavelet', 'sym12','DenoisingMethod', 'Bayes', 'ThresholdRule', 'Median');
        u_val_filter_wdenoise = u_val_filter_wdenoise(250:end-251,:);


        %% polynomial filter

        %define filtered training data
        u_train_filter_poly = my_poly_filter(u_train_current,t_train_current,dt,p,L,overlap);
        u_train_filter_poly = u_train_filter_poly(250:end-251,:);

        %define filtered validation data
        u_val_filter_poly = my_poly_filter(u_val_current,t_val_current,dt,p,L,overlap);
        u_val_filter_poly = u_val_filter_poly(250:end-251,:);


        %% prune the rest of the training data

        %prune the training time
        t_train_current = t_train_current(250:end-251) - t_train_current(250);
        t_val_current = t_val_current(250:end-251) - t_val_current(250);

        %prune the signal
        u_train_current = u_train_current(250:end-251,:);
        u_train_true_current = u_train_true_current(250:end-251,:);
        u_val_current = u_val_current(250:end-251,:);
        u_val_true_current = u_val_true_current(250:end-251,:);


        %% ease of notation for error metrics

        %direct data
        X_na = u_train_current(1:end-1,:);
        X_wdenoise = u_train_filter_wdenoise(1:end-1,:);
        X_poly = u_train_filter_poly(1:end-1,:);
        X_true = u_train_true_current(1:end-1,:);


        %% angle

        %direct
        angle_na = mean(acos(dot(X_na, X_true, 2) ./ (sqrt(sum(X_na.^2, 2)) .* sqrt(sum(X_true.^2, 2)))));
        angle_wdenoise = mean(acos(dot(X_wdenoise, X_true, 2) ./ (sqrt(sum(X_wdenoise.^2, 2)) .* sqrt(sum(X_true.^2, 2)))));
        angle_poly = mean(acos(dot(X_poly, X_true, 2) ./ (sqrt(sum(X_poly.^2, 2)) .* sqrt(sum(X_true.^2, 2)))));
        angle_errors = [angle_na; angle_wdenoise; angle_poly];


        %% RMSE

        %direct
        rmse_na = mean(rms(X_na - X_true));
        rmse_wdenoise = mean(rms(X_wdenoise - X_true));
        rmse_poly = mean(rms(X_poly - X_true));
        rmse_errors = [rmse_na; rmse_wdenoise; rmse_poly];


        %% power spectral density of filtered signals

        %initialize
        LSD_na = 0;
        LSD_wdenoise = 0;
        LSD_poly = 0;

        %compute PSD and average over all coordinates
        fs = 1/dt;
        for iii=1:64
            [pxx_na, f_na] = pwelch(X_na(:,iii), [], [], [], fs);
            [pxx_wdenoise, f_wdenoise] = pwelch(X_wdenoise(:,iii), [], [], [], fs);
            [pxx_poly, f_poly] = pwelch(X_poly(:,iii), [], [], [], fs);
            [pxx_true, f_true] = pwelch(X_true(:,iii), [], [], [], fs);

            %quantitative result for PSD
            LSD_na = LSD_na +  sqrt(mean((10*log10(pxx_true) - 10*log10(pxx_na)).^2));
            LSD_wdenoise = LSD_wdenoise + sqrt(mean((10*log10(pxx_true) - 10*log10(pxx_wdenoise)).^2));
            LSD_poly     = LSD_poly +  sqrt(mean((10*log10(pxx_true) - 10*log10(pxx_poly)).^2));
        end
        LSD_errors = [LSD_na/64; LSD_wdenoise/64; LSD_poly/64];

  
        %% save the results over each trial

        %training
        data{nnn}.u_train(:,:,qqq) = u_train_current;
        data{nnn}.u_train_filter_wdenoise(:,:,qqq) = u_train_filter_wdenoise;
        data{nnn}.u_train_filter_poly(:,:,qqq) = u_train_filter_poly;
        data{nnn}.u_train_true(:,:,qqq) = u_train_true_current;

        %validation
        data{nnn}.u_val(:,:,qqq) = u_val_current;
        data{nnn}.u_val_filter_wdenoise(:,:,qqq) = u_val_filter_wdenoise;
        data{nnn}.u_val_filter_poly(:,:,qqq) = u_val_filter_poly;
        data{nnn}.u_val_true(:,:,qqq) = u_val_true_current;

        %testing
        data{nnn}.u_test(:,:,qqq) = u_test_current;
        data{nnn}.u_test_true(:,:,qqq) = u_test_true_current;

        %error
        data{nnn}.error_angle(:,qqq) = angle_errors;
        data{nnn}.error_rmse(:,qqq) = rmse_errors;
        data{nnn}.error_LSD(:,qqq) = LSD_errors;

    end

    %% export the all the results

    %name the files correctly
    file_suffixes = {'01', '05', '10', '20'};
    filename_data = sprintf('KS_filtered_data_%s.mat', file_suffixes{nnn});
    filename_error = sprintf('KS_filtered_data_error_%s.mat', file_suffixes{nnn});

    %rename stuff again 
    u_train_output = data{nnn}.u_train;
    u_train_filter_wdenoise_output = data{nnn}.u_train_filter_wdenoise;
    u_train_filter_poly_output = data{nnn}.u_train_filter_poly;
    u_train_true_output = data{nnn}.u_train_true;
    t_train_output = t_train_current;

    u_val_output = data{nnn}.u_val;
    u_val_filter_wdenoise_output = data{nnn}.u_val_filter_wdenoise;
    u_val_filter_poly_output = data{nnn}.u_val_filter_poly;
    u_val_true_output = data{nnn}.u_val_true;
    t_val_output = t_val_current;

    u_test_output = data{nnn}.u_test;
    u_test_true_output = data{nnn}.u_test_true;
    t_test_output = t_test_current;

    %and for the error
    angle_error_output = data{nnn}.error_angle;
    rmse_error_output = data{nnn}.error_rmse;
    LSD_error_output = data{nnn}.error_LSD;

    %save those rascals up real good
    save(filename_data, ...
        'u_train_true_output','u_train_output', 'u_train_filter_wdenoise_output', 'u_train_filter_poly_output', 't_train_output', ...
        'u_val_true_output','u_val_output','u_val_filter_wdenoise_output','u_val_filter_poly_output','t_val_output', ...
        'u_test_true_output','u_test_output','t_test_output', ...
        'dt','master_test_indices','nlevel')

    save(filename_error,'angle_error_output','rmse_error_output','LSD_error_output')
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% auxiliary functions

%linear transformation
function[u] = my_poly_filter(u_train,t_train,dt,p,L,overlap)

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
