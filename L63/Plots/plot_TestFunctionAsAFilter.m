%% setup

%mr clean
clc
clf
close all
clear


%% load the data

%load data
load('L63_filtered_data_10.mat')

%time
t_train = t_train_output;

%Lyapunov exponent
Lambda = 0.91;


%% loop over all filters and all 100 trajectories

%choose trajectory of interest to plot
nnn = 1;

%plot it
figure(1)
hold on
plot(t_train(1:1500)*Lambda,u_train_output(1:1500,1,nnn),'.','color',[0.6 0.6 1],'markersize',30)
plot(t_train(1:1500)*Lambda,u_train_true_output(1:1500,1,nnn),'k-','LineWidth',4)
plot(t_train(1:1500)*Lambda,u_train_filter_poly_output(1:1500,1,nnn),'r--','LineWidth',4)
xlabel('Time t')
ylabel('State')
set(gca,'fontsize',14)
box on
xlim([0 10])
ylim([-40 25])
set(gcf,'Position',[533,428,1145,420])
grid on
legend('Noisy Observations','Ground Truth','Reconstruction','location','southeast')
