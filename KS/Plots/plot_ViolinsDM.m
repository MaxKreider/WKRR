%% setup

%mr clean
clc
clf
close all
clear


%% load the data

%Strong
S = load('KS_DM_strong_05.mat');
val_strong_results = [S.TrialResults.filter];

%Weak train noisy
S = load('KS_DM_weak_train_noisy_05.mat');
val_weak_train_noisy_results = [S.TrialResults.filter];

%Weak train filter
S = load('KS_DM_weak_train_filter_05.mat');
val_weak_train_filter_results = [S.TrialResults.filter];


%% extract mean VPTs and make a table (filtered validation)

%extract the data
mean_vpts_strong = reshape([val_strong_results.mean_vpt], 3, 50);
mean_vpts_weak_noisy = reshape([val_weak_train_noisy_results.mean_vpt], 1, 50);
mean_vpts_weak_filter = reshape([val_weak_train_filter_results.mean_vpt], 1, 50);

%list of 100 mean VPTs (each over 500 trials)
vpt_list_strong_na = mean_vpts_strong(1,:);
vpt_list_strong_wavelet = mean_vpts_strong(2,:);
vpt_list_strong_poly = mean_vpts_strong(3,:);
vpt_list_weak_noisy = mean_vpts_weak_noisy;
vpt_list_weak_filter = mean_vpts_weak_filter;

%compute statistics
stats_matrix = [mean(vpt_list_strong_na),     std(vpt_list_strong_na),     max(vpt_list_strong_na),     min(vpt_list_strong_na);
                mean(vpt_list_strong_wavelet),     std(vpt_list_strong_wavelet),     max(vpt_list_strong_wavelet),     min(vpt_list_strong_wavelet);
                mean(vpt_list_strong_poly),     std(vpt_list_strong_poly),     max(vpt_list_strong_poly),     min(vpt_list_strong_poly);
                mean(vpt_list_weak_noisy),     std(vpt_list_weak_noisy),     max(vpt_list_weak_noisy),     min(vpt_list_weak_noisy);
                mean(vpt_list_weak_filter),     std(vpt_list_weak_filter),     max(vpt_list_weak_filter),     min(vpt_list_weak_filter)];

%create table
RowNames = {'Mean_VPT', 'Std_Dev', 'Max_VPT', 'Min_VPT'};
VPT_Comparison_Table_T = array2table(stats_matrix, ...
    'VariableNames', {'Mean VPT', 'Std', 'Max', 'Min'}, ...
    'RowNames', {'Strong DM (n/a)', 'Strong DM (Wavelet)', 'Strong DM (Poly)', 'Weak DM (n/a)', 'Weak DM (Poly)'});

%display table
fprintf('\n\n')
disp(VPT_Comparison_Table_T);


%% violin plots

%create the figure
figure(10);
set(gcf, 'Position', [366.6,93,685.6,600]);
tiledlayout(1, 1, 'TileSpacing', 'Loose');

%violin plot
nexttile;
hold on;
vpt_data_to_plot = [vpt_list_strong_na', vpt_list_strong_wavelet', vpt_list_strong_poly', vpt_list_weak_noisy', vpt_list_weak_filter'];
v = violinplot(vpt_data_to_plot);

%colors
v(1).FaceColor = [0.5, 0.5, 0.5]; 
v(1).FaceAlpha = 0.5;
v(2).FaceColor = [0.25, 0.60, 0.90]; 
v(2).FaceAlpha = 0.5;
v(3).FaceColor = [0.85 0.33 0.10]; 
v(3).FaceAlpha = 0.5;
v(4).FaceColor = [1, 0.45, 0.45]; 
v(4).FaceAlpha = 0.5;
v(5).FaceColor = [1, 0, 0]; 
v(5).FaceAlpha = 0.5;

%plot the median
line_w = 0.15; 

med_strong_na = median(vpt_list_strong_na);
med_strong_wavelet = median(vpt_list_strong_wavelet);
med_strong_poly = median(vpt_list_strong_poly);
med_weak_na = median(vpt_list_weak_noisy);
med_weak_poly = median(vpt_list_weak_filter);

line([1-line_w, 1+line_w], [med_strong_na, med_strong_na], 'Color', 'k', 'LineWidth', 3);
line([2-line_w, 2+line_w], [med_strong_wavelet, med_strong_wavelet], 'Color', 'k', 'LineWidth', 3);
line([3-line_w, 3+line_w], [med_strong_poly, med_strong_poly], 'Color', 'k', 'LineWidth', 3);
line([4-line_w, 4+line_w], [med_weak_na, med_weak_na], 'Color', 'k', 'LineWidth', 3);
line([5-line_w, 5+line_w], [med_weak_poly, med_weak_poly], 'Color', 'k', 'LineWidth', 3);

%formatting
xticklabels({'Strong DM (n/a)', 'Strong DM (Wavelet)','Strong DM (Poly)','Weak DM (n/a)', 'Weak DM (Poly)'});
ylabel('VPT');
grid on;
set(gca, 'FontSize', 14);
box on
