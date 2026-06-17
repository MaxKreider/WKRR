%% setup

%mr clean
clc
clf
close all
clear


%% load data

%load the four cases
S = load('L63_weak_val_A_20.mat');
results_50 = [S.TrialResults.filter];

S = load('L63_weak_val_B_20.mat');
results_100 = [S.TrialResults.filter];

S = load('L63_weak_val_C_20.mat');
results_200 = [S.TrialResults.filter];

S = load('L63_weak_val_D_20.mat');
results_400 = [S.TrialResults.filter];

S = load('L63_weak_val_E_20.mat');
results_500 = [S.TrialResults.filter];


%% extract the data

%extract the VPTs
mean_vpts_50 = reshape([results_50.mean_vpt], 1, 100);
mean_vpts_100 = reshape([results_100.mean_vpt], 1, 100);
mean_vpts_200 = reshape([results_200.mean_vpt], 1, 100);
mean_vpts_400 = reshape([results_400.mean_vpt], 1, 100);
mean_vpts_500 = reshape([results_500.mean_vpt], 1, 100);


%% visualize

%organize data into arrays
L_values = [50, 100, 200, 400, 500] * 0.01 * 0.91
vpt_data = {mean_vpts_50, mean_vpts_100, mean_vpts_200, mean_vpts_400, mean_vpts_500};

%calculate statistics
final_means = cellfun(@mean, vpt_data);
final_stds  = cellfun(@std, vpt_data);

%create the Figure
figure(1);
set(gcf, 'Position', [200, 200, 800, 500]);

%plot the main error bar line
e = errorbar(L_values, final_means, final_stds,'-ko','markersize',10,'MarkerFaceColor','k','CapSize',20,'LineWidth',3);

%formatting
grid on;
xlabel('Validation Segment Length (VPT units)')
ylabel('Mean VPT')
set(gca,'fontsize',16)
axis square
box on

%adjust y-lim if needed
ylim([0 2]) % noise 20%
%ylim([1 6]) % noise 1%


%% table

%collect data
vpt_data = {mean_vpts_50, mean_vpts_100, mean_vpts_200, mean_vpts_400, mean_vpts_500};
L_labels = {'L = 50', 'L = 100', 'L = 200', 'L = 400', 'L = 500'};

%initialize stats matrix
stats_matrix = zeros(5, 4);

%population the stats matrix
for i = 1:5
    current_data = vpt_data{i};
    stats_matrix(i, :) = [mean(current_data), std(current_data), max(current_data), min(current_data)];
end

%make the matrix a table
VPT_Sensitivity_Table = array2table(stats_matrix,'VariableNames', {'Mean_VPT', 'Std_Dev', 'Max_VPT', 'Min_VPT'},'RowNames', L_labels);

%display the table
fprintf('\n\n')
disp(VPT_Sensitivity_Table);