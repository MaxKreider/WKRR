%% setup

%mr clean
clc
clf
close all
clear


%% load the data

%Strong 
S = load('L63_strong_00.mat');
val_strong_results = [S.TrialResults.filter];

%Weak
S = load('L63_weak_00.mat'); 
val_weak_results = [S.TrialResults.filter];

%Weak DM
S = load('L63_DM_weak_00.mat'); 
val_weakDM_results = [S.TrialResults.filter];


%% extract mean VPTs and make a table

%extract the data
mean_vpts_strong = reshape([val_strong_results.mean_vpt], 1, 100);
mean_vpts_weak = reshape([val_weak_results.mean_vpt], 1, 100);
mean_vpts_weakDM = reshape([val_weakDM_results.mean_vpt], 1, 100);

%list of 100 mean VPTs (each over 500 trials)
vpt_list_strong = mean_vpts_strong;
vpt_list_weak = mean_vpts_weak;
vpt_list_weakDM = mean_vpts_weakDM;

%concatenate statistics
stats_matrix = [mean(vpt_list_strong), std(vpt_list_strong), max(vpt_list_strong), min(vpt_list_strong);
                mean(vpt_list_weak), std(vpt_list_weak), max(vpt_list_weak), min(vpt_list_weak);
                mean(vpt_list_weakDM), std(vpt_list_weakDM), max(vpt_list_weakDM), min(vpt_list_weakDM)];

%create table
RowNames = {'Mean_VPT', 'Std_Dev', 'Max_VPT', 'Min_VPT'};
VPT_Comparison_Table_T = array2table(stats_matrix, ...
    'VariableNames', {'Mean VPT', 'Std', 'Max', 'Min'}, ...
    'RowNames', {'Strong','Weak','Weak DM'});

%display table
fprintf('\n\n');
disp(VPT_Comparison_Table_T);


%% violin plots and cumulative distribution

%create the figure
figure(10);
tiledlayout(1, 1, 'TileSpacing', 'Loose');

%violin plot
nexttile;
hold on;
vpt_data_to_plot = [vpt_list_strong', vpt_list_weak', vpt_list_weakDM'];
v = violinplot(vpt_data_to_plot);

%colors
v(1).FaceColor = [0.5, 0.5, 0.5]; 
v(1).FaceAlpha = 0.5;
v(2).FaceColor = [0.45, 0.20, 0.60]; 
v(2).FaceAlpha = 0.5;
v(3).FaceColor = [1, 0, 0]; 
v(3).FaceAlpha = 0.5;

%plot the median
line_w = 0.15; 
med_s = median(vpt_list_strong);
med_w = median(vpt_list_weak);
med_wDM = median(vpt_list_weakDM);
line([1-line_w, 1+line_w], [med_s, med_s], 'Color', 'k', 'LineWidth', 3);
line([2-line_w, 2+line_w], [med_w, med_w], 'Color', 'k', 'LineWidth', 3);
line([3-line_w, 3+line_w], [med_wDM, med_wDM], 'Color', 'k', 'LineWidth', 3);

%formatting
xticklabels({'Strong','Weak','Weak DM'});
ylabel('VPT');
grid on;
set(gca, 'FontSize', 14);
box on
axis square
