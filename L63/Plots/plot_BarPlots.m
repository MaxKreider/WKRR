%% setup

%mr clean
clc
clf
close all
clear

%colors for the bar plots
colors = [0.50, 0.50, 0.50; 0.12 0.47 0.71; 0.93 0.50 0.15];

%for the axis labels
methods = {'n/a', 'Wavelet', 'Poly'};


%% load data

%extract data
load('L63_filtered_data_error_01.mat')


%% just 10% noise RMSE

%extract the error of interest
plot_data = rmse_output;

%compute mean and std
means = mean(plot_data, 2);
stds  = std(plot_data, 0, 2);

%plot it
figure(1000)
hold on
b = bar(1:3, means, 'FaceColor', 'flat', 'EdgeColor', 'none', 'FaceAlpha', 0.7);
for m = 1:3
    b.CData(m,:) = colors(m,:);
end
errorbar(1:3, means, stds, 'k', 'linestyle', 'none', 'linewidth', 1.2);
title('RMSE', 'FontSize', 12);
set(gca, 'XTick', 1:3, 'XTickLabel', methods, 'FontWeight', 'bold');
grid on;
box on;
ylabel('Error')
set(gcf,'Position',[150,242,409.8,420])
set(gca,'fontsize',14)


%% same for angle

%extract error of interest
plot_data = angle_output;

%compute mean and std
means = mean(plot_data, 2);
stds  = std(plot_data, 0, 2);

%plot it
figure(1001)
hold on
b = bar(1:3, means, 'FaceColor', 'flat', 'EdgeColor', 'none', 'FaceAlpha', 0.7);
for m = 1:3
    b.CData(m,:) = colors(m,:);
end
errorbar(1:3, means, stds, 'k', 'linestyle', 'none', 'linewidth', 1.2);
title('Angle', 'FontSize', 12);
set(gca, 'XTick', 1:3, 'XTickLabel', methods, 'FontWeight', 'bold');
grid on;
box on;
ylabel('Error')
set(gcf,'Position',[600,242,409.8,420])
set(gca,'fontsize',14)


%% same for LSD

%extract error of interest
plot_data = LSD_output;

%compute mean and std
means = mean(plot_data, 2);
stds  = std(plot_data, 0, 2);

%plot it
figure(1002)
hold on
b = bar(1:3, means, 'FaceColor', 'flat', 'EdgeColor', 'none', 'FaceAlpha', 0.7);
for m = 1:3
    b.CData(m,:) = colors(m,:);
end
errorbar(1:3, means, stds, 'k', 'linestyle', 'none', 'linewidth', 1.2);
title('LSD', 'FontSize', 12);
set(gca, 'XTick', 1:3, 'XTickLabel', methods, 'FontWeight', 'bold');
grid on;
box on;
ylabel('Error')
set(gcf,'Position',[1050,242,409.8,420])
set(gca,'fontsize',14)