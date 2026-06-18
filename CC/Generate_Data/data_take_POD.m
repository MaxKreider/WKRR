%% setup

%mr clean
clc
clf
close all
clear


%% grid 

%grid info
grid_info = h5info('Challenge2_1_grid.h5');
grid_names = {grid_info.Datasets.Name};

%load the data
grid_x = h5read('Challenge2_1_grid.h5', '/x');
grid_y = h5read('Challenge2_1_grid.h5', '/y');


%% test input

%test info
test_info = h5info('Challenge2_1_test_input.h5');
test_names = {test_info.Datasets.Name};

%load the data
u = h5read('Challenge2_1_test_input.h5', '/ux');
v = h5read('Challenge2_1_test_input.h5', '/uy');


%% visualize validation data

%which model to visualize
model_idx = 5; 


%% load training data for POD

%use the training set to build the basis
train_info = h5info('Challenge2_1_train.h5');
ux_train = h5read('Challenge2_1_train.h5', '/ux');
uy_train = h5read('Challenge2_1_train.h5', '/uy');

%get size
[nx, ny, nt_train] = size(ux_train);

%concatenate into a data matrix (time, spatial)
X_train = [reshape(ux_train, [], nt_train)', reshape(uy_train, [], nt_train)'];


%% perform POD 

%reshape training data so its easier to work with
u_train_flat = reshape(ux_train, [], nt_train)';
v_train_flat = reshape(uy_train, [], nt_train)';

%compute means and standard deviation
u_mean = mean(u_train_flat, 1);
v_mean = mean(v_train_flat, 1);
u_std = std(u_train_flat, [], 1);
v_std = std(v_train_flat, [], 1);

%normalize u and v
B_u = (u_train_flat - u_mean) ./ u_std;
B_v = (v_train_flat - v_mean) ./ v_std;

%combine the data back 
B = [B_u, B_v];

%truncation index
r = 100;

%svd
[~, S, V] = svds(B, r);
sv_vec = diag(S);

%project into POD space and compute statistics of the modes
A_train = (B * V); 
A_mu  = mean(A_train, 1);
A_std = std(A_train, 0, 1);


%% Project a Test Snapshot

%load one of the test data segements
u_test_flat = reshape(u(:,:,:,model_idx), [], 70)';
v_test_flat = reshape(v(:,:,:,model_idx), [], 70)';

%normalize and concatenate
B_test = [(u_test_flat - u_mean)./u_std, (v_test_flat - v_mean)./v_std];

%project it to a reduced state
x_tilde = (B_test * V);

%normalize per mode
x_tilde_norm = (x_tilde - A_mu) ./ A_std;


%% Reconstruct the test snapshot

%undo mode normalization
X_rec = x_tilde_norm.*A_std + A_mu;

%reconstruction
X_rec = X_rec * V';

%undo normalization
u_rec_flat = X_rec(:, 1:nx*ny) .* u_std + u_mean;
v_rec_flat = X_rec(:, nx*ny+1:end) .* v_std + v_mean;

%reshape for plotting: [nx, ny, time]
u_rec = reshape(u_rec_flat', [nx, ny, 70]);
v_rec = reshape(v_rec_flat', [nx, ny, 70]);


%% Visualize POD Reconstruction

%timeslices to plot
time_indices = [1, 2, 5, 10, 15]+40; 

%build figure to view the original data and the reconstruction
figure(2)
set(gcf, 'Position', [83.4, 50, 1404, 450])
tiledlayout(4, length(time_indices), 'TileSpacing', 'compact', 'Padding', 'normal');
colormap turbo

%original u
for i = 1:length(time_indices)
    nexttile(i);
    imagesc(grid_x, grid_y, u(:,:,time_indices(i), model_idx))
    axis image off
    clim([-100, 250])
    colorbar
    set(gca, 'YDir', 'normal')
    if i == 1
        ylabel('u', 'Visible', 'on', 'FontWeight', 'bold')
    end
end

%POD u
for i = 1:length(time_indices)
    nexttile(i + length(time_indices));
    imagesc(grid_x, grid_y, u_rec(:,:,time_indices(i)))
    axis image off
    clim([-100, 250])
    colorbar
    set(gca, 'YDir', 'normal')
    if i == 1
        ylabel('POD', 'Visible', 'on', 'FontWeight', 'bold')
    end
end

%original v
for i = 1:length(time_indices)
    nexttile(i + 2*length(time_indices));
    imagesc(grid_x, grid_y, v(:,:,time_indices(i), model_idx))
    axis image off
    clim([-25, 25])
    colorbar
    set(gca, 'YDir', 'normal')
    if i == 1
        ylabel('v', 'Visible', 'on', 'FontWeight', 'bold')
    end
end

%POD v
for i = 1:length(time_indices)
    nexttile(i + 3*length(time_indices));
    imagesc(grid_x, grid_y, v_rec(:,:,time_indices(i)))
    axis image off
    clim([-25, 25])
    colorbar
    set(gca, 'YDir', 'normal')
    if i == 1
        ylabel('POD', 'Visible', 'on', 'FontWeight', 'bold')
    end
end


%% Plot Singular Value Spectrum (Energy)

%plot the singular values   
figure(3)
loglog(sv_vec, 'linewidth', 2)
ylabel('Singular Value Magnitude')
xlabel('Singular Value Index')
grid on


%% Compute and Plot NMSE for the single trajectory (in original space)

%take truth and reconstruction in original coordinates, and subtract the means
q_true = [u_test_flat - u_mean, v_test_flat - v_mean];
q_recon = [u_rec_flat - u_mean, v_rec_flat - v_mean];

%compute the error for each timestep
nmse_time = mean((q_true - q_recon).^2, 2) ./ mean(q_true.^2, 2);

%plot it
figure(6)
plot(1:70, nmse_time, 'b-', 'LineWidth', 2)
grid on
xlabel('Time step (\Deltat)')
ylabel('NMSE Error')
ylim([0, 2])
set(gca,'fontsize',15)


%% save the POD parameters

%save it
save('POD_param_r_100.mat', 'V', 'u_mean', 'v_mean', 'u_std', 'v_std', 'A_std', 'A_mu');