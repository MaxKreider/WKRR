%% setup

%mr clean
clc
clf
close all
clear


%% load the data

%load WKRR data with DM kernel
load("weak_results_DM_r_25.mat")
my_DM_results = TrialResults;

%load WKRR data
load("weak_results_r_25.mat")

%load LSTM data
load("lstm_results_r_25.mat")

%load the POD decomp
load("POD_param_r_25.mat")

%other stuff
x = h5read("Challenge2_1_grid.h5",'/x');
y = h5read("Challenge2_1_grid.h5",'/y');


%% look at WiKRR validation

%look at the mean VPT heat map
figure(100)
heat_map = TrialResults(1).filter(1).coarse_grid.';
heat_map(heat_map > 10) = 10;
imagesc(log10(TrialResults(1).filter(1).coarse_eps_mesh),log10(TrialResults(1).filter(1).coarse_lam_mesh),heat_map)
hold on
plot(log10(TrialResults(1).filter(1).coarse_params(1)),log10(TrialResults(1).filter(1).coarse_params(2)),'ko','markersize',10,'markerfacecolor','m')
colorbar
xlabel('log(\epsilon)')
ylabel('log(\lambda)')
set(gca,'fontsize',15)
title('VPT heat map')
colormap parula
title('Coarse')

figure(101)
heat_map = TrialResults(1).filter(1).fine_grid.';
heat_map(heat_map > 10) = 10;
imagesc(log10(TrialResults(1).filter(1).fine_eps_mesh),log10(TrialResults(1).filter(1).fine_lam_mesh),heat_map)
hold on
plot(log10(TrialResults(1).filter(1).fine_params(1)),log10(TrialResults(1).filter(1).fine_params(2)),'ko','markersize',10,'markerfacecolor','m')
colorbar
xlabel('log(\epsilon)')
ylabel('log(\lambda)')
set(gca,'fontsize',15)
title('VPT heat map')
colormap parula
title('Fine')


%% extract the reconstructions

%initialize
DM_truth_collection = zeros(31,51*154*2,32);
DM_POD_collection = zeros(31,51*154*2,32);
DM_forecast_collection = zeros(31,51*154*2,32);

%WDMKRR
for nnn = 1:32

    %extract forecast
    DM_sol = TrialResults(nnn).filter(1).ker_sol;

    %extract POD truth
    DM_POD = TrialResults(nnn).filter(1).u_pod;

    %extract ground truth
    DM_truth = TrialResults(nnn).filter(1).u_true;

    %undo per mode normalization
    DM_sol = DM_sol.*A_std + A_mu;
    DM_POD = DM_POD.*A_std + A_mu;

    %undo POD
    DM_sol = DM_sol * V';
    DM_POD = DM_POD * V';

    %undo physical normalization
    DM_sol = DM_sol.*[u_std,v_std] + [u_mean,v_mean];
    DM_POD = DM_POD.*[u_std,v_std] + [u_mean,v_mean];
    
    %save it
    DM_truth_collection(:,:,nnn) = DM_truth;
    DM_POD_collection(:,:,nnn) = DM_POD;
    DM_forecast_collection(:,:,nnn) = DM_sol;
    
end

%prune first timestep for fairness
DM_truth_collection = DM_truth_collection(2:end,:,:);
DM_POD_collection = DM_POD_collection(2:end,:,:);
DM_forecast_collection = DM_forecast_collection(2:end,:,:);


%% extract the reconstructions for DM kernel

%initialize
DM_truth_collection_DMDM = zeros(31,51*154*2,32);
DM_POD_collection_DMDM = zeros(31,51*154*2,32);
DM_forecast_collection_DMDM = zeros(31,51*154*2,32);

%WDMKRR
for nnn = 1:32

    %extract forecast
    DM_sol = my_DM_results(nnn).filter(1).ker_sol;

    %extract POD truth
    DM_POD = my_DM_results(nnn).filter(1).u_pod;

    %extract ground truth
    DM_truth = my_DM_results(nnn).filter(1).u_true;

    %undo per mode normalization
    DM_sol = DM_sol.*A_std + A_mu;
    DM_POD = DM_POD.*A_std + A_mu;

    %undo POD
    DM_sol = DM_sol * V';
    DM_POD = DM_POD * V';

    %undo physical normalization
    DM_sol = DM_sol.*[u_std,v_std] + [u_mean,v_mean];
    DM_POD = DM_POD.*[u_std,v_std] + [u_mean,v_mean];
    
    %save it
    DM_truth_collection_DMDM(:,:,nnn) = DM_truth;
    DM_POD_collection_DMDM(:,:,nnn) = DM_POD;
    DM_forecast_collection_DMDM(:,:,nnn) = DM_sol;
    
end

%prune first timestep for fairness
DM_truth_collection_DMDM = DM_truth_collection_DMDM(2:end,:,:);
DM_POD_collection_DMDM = DM_POD_collection_DMDM(2:end,:,:);
DM_forecast_collection_DMDM = DM_forecast_collection_DMDM(2:end,:,:);


%% visualize WiKRR 

%plotting stuff
my_time = [1, 2, 5, 15, 30];
my_model = 5;

%create a cell array to hold the grids
MY_WDMKRR_PLOTS = cell(6,5);

%select the truth
WDMKRR_truth = DM_truth_collection(:,:,my_model);
WDMKRR_POD = DM_POD_collection(:,:,my_model);
WDMKRR_forecast = DM_forecast_collection(:,:,my_model);

%populate the plot cell structure
for ttt = 1:length(my_time)

    %extract the plots
    true = WDMKRR_truth(my_time(ttt),:);
    u_true = reshape(true(1:51*154),[51 154]);
    v_true = reshape(true(51*154+1:2*51*154),[51 154]);

    POD = WDMKRR_POD(my_time(ttt),:);
    u_POD = reshape(POD(1:51*154),[51 154]);
    v_POD = reshape(POD(51*154+1:2*51*154),[51 154]);

    forecast = WDMKRR_forecast(my_time(ttt),:);
    u_forecast = reshape(forecast(1:51*154),[51 154]);
    v_forecast = reshape(forecast(51*154+1:2*51*154),[51 154]);

    %populate the cell plot structure
    MY_WDMKRR_PLOTS{1,ttt} = u_true;
    MY_WDMKRR_PLOTS{2,ttt} = u_POD;
    MY_WDMKRR_PLOTS{3,ttt} = u_forecast;
    MY_WDMKRR_PLOTS{4,ttt} = v_true;
    MY_WDMKRR_PLOTS{5,ttt} = v_POD;
    MY_WDMKRR_PLOTS{6,ttt} = v_forecast;
end

%WiKRR plot
figure(1)
set(gcf,'position',[101,58.6,1337.6,645.6])
sgtitle('WKRR', 'FontWeight', 'bold')

count = 0;
for vertical = 1:6
    for horizontal = 1:5

        %update count
        count = count + 1;

        %plot it
        h = subplot(6,5,count);
        imagesc(x,y,MY_WDMKRR_PLOTS{vertical,horizontal})

        %scale
        if vertical < 4
            clim([-100 250])
        else
            clim([-25 25])
        end
        
        %formatting
        axis image off
        colormap(turbo)
        set(gca, 'YDir', 'normal')
        set(gca,'fontsize',12)

        %top labels
        if vertical == 1
            title(sprintf('\\Deltat = %d', my_time(horizontal)), 'FontWeight', 'bold')
        end

        %side labels
        if horizontal == 1
            if vertical == 1 || vertical == 4
                ax = ylabel('True', 'FontWeight', 'bold');
                set(ax, 'Rotation', 0, 'HorizontalAlignment', 'center');
            elseif vertical == 2 || vertical == 5
                ax = ylabel('POD', 'FontWeight', 'bold');
                set(ax, 'Rotation', 0, 'HorizontalAlignment', 'center');
            elseif vertical == 3 || vertical == 6
                ax = ylabel('WiKRR', 'FontWeight', 'bold');
                set(ax, 'Rotation', 0, 'HorizontalAlignment', 'center');
            end
        end
        ax.Visible = 'on';

        %colorbars
        if count == 5 || count == 20
            original_pos = h.Position;
            colorbar
            h.Position = original_pos;
        end

    end
end


%% visualize WiKRR with DM kernel

%plotting stuff
my_time = [1, 2, 5, 15, 30];
my_model = 5;

%create a cell array to hold the grids
MY_WDMKRR_PLOTS_DMDM = cell(6,5);

%select the truth
WDMKRR_truth_DMDM = DM_truth_collection_DMDM(:,:,my_model);
WDMKRR_POD_DMDM = DM_POD_collection_DMDM(:,:,my_model);
WDMKRR_forecast_DMDM = DM_forecast_collection_DMDM(:,:,my_model);

%populate the plot cell structure
for ttt = 1:length(my_time)

    %extract the plots
    true = WDMKRR_truth_DMDM(my_time(ttt),:);
    u_true = reshape(true(1:51*154),[51 154]);
    v_true = reshape(true(51*154+1:2*51*154),[51 154]);

    POD = WDMKRR_POD_DMDM(my_time(ttt),:);
    u_POD = reshape(POD(1:51*154),[51 154]);
    v_POD = reshape(POD(51*154+1:2*51*154),[51 154]);

    forecast = WDMKRR_forecast_DMDM(my_time(ttt),:);
    u_forecast = reshape(forecast(1:51*154),[51 154]);
    v_forecast = reshape(forecast(51*154+1:2*51*154),[51 154]);

    %populate the cell plot structure
    MY_WDMKRR_PLOTS_DMDM{1,ttt} = u_true;
    MY_WDMKRR_PLOTS_DMDM{2,ttt} = u_POD;
    MY_WDMKRR_PLOTS_DMDM{3,ttt} = u_forecast;
    MY_WDMKRR_PLOTS_DMDM{4,ttt} = v_true;
    MY_WDMKRR_PLOTS_DMDM{5,ttt} = v_POD;
    MY_WDMKRR_PLOTS_DMDM{6,ttt} = v_forecast;
end

%WiKRR plot
figure(2)
set(gcf,'position',[101,58.6,1337.6,645.6])
sgtitle('WKRR with DM Kernel', 'FontWeight', 'bold')

count = 0;
for vertical = 1:6
    for horizontal = 1:5

        %update count
        count = count + 1;

        %plot it
        h = subplot(6,5,count);
        imagesc(x,y,MY_WDMKRR_PLOTS_DMDM{vertical,horizontal})

        %scale
        if vertical < 4
            clim([-100 250])
        else
            clim([-25 25])
        end
        
        %formatting
        axis image off
        colormap(turbo)
        set(gca, 'YDir', 'normal')
        set(gca,'fontsize',12)

        %top labels
        if vertical == 1
            title(sprintf('\\Deltat = %d', my_time(horizontal)), 'FontWeight', 'bold')
        end

        %side labels
        if horizontal == 1
            if vertical == 1 || vertical == 4
                ax = ylabel('True', 'FontWeight', 'bold');
                set(ax, 'Rotation', 0, 'HorizontalAlignment', 'center');
            elseif vertical == 2 || vertical == 5
                ax = ylabel('POD', 'FontWeight', 'bold');
                set(ax, 'Rotation', 0, 'HorizontalAlignment', 'center');
            elseif vertical == 3 || vertical == 6
                ax = ylabel('WiKRR', 'FontWeight', 'bold');
                set(ax, 'Rotation', 0, 'HorizontalAlignment', 'center');
            end
        end
        ax.Visible = 'on';

        %colorbars
        if count == 5 || count == 20
            original_pos = h.Position;
            colorbar
            h.Position = original_pos;
        end

    end
end


%% visualize LSTM 

%create a cell array to hold the grids
MY_LSTM_PLOTS = cell(6,5);

%populate the plot cell structure
for ttt = 1:length(my_time)
    
    true_u = reshape(LSTM_truth_sol_u(ttt,:),[51 154]);
    true_v = reshape(LSTM_truth_sol_v(ttt,:),[51 154]);

    pod_u = reshape(LSTM_POD_sol_u(ttt,:),[51 154]);
    pod_v = reshape(LSTM_POD_sol_v(ttt,:),[51 154]);

    forecast_u = reshape(LSTM_sol_u(ttt,:),[51 154]);
    forecast_v = reshape(LSTM_sol_v(ttt,:),[51 154]);

    %populate the cell plot structure
    MY_LSTM_PLOTS{1,ttt} = true_u;
    MY_LSTM_PLOTS{2,ttt} = pod_u;
    MY_LSTM_PLOTS{3,ttt} = forecast_u;
    MY_LSTM_PLOTS{4,ttt} = true_v;
    MY_LSTM_PLOTS{5,ttt} = pod_v;
    MY_LSTM_PLOTS{6,ttt} = forecast_v;
end

%WDMKRR plot
figure(3)
set(gcf,'position',[101,58.6,1337.6,645.6])
sgtitle('LSTM', 'FontWeight', 'bold')

count = 0;
for vertical = 1:6
    for horizontal = 1:5

        %update count
        count = count + 1;

        %plot it
        h = subplot(6,5,count);
        imagesc(x,y,MY_LSTM_PLOTS{vertical,horizontal})

        %scale
        if vertical < 4
            clim([-100 250])
        else
            clim([-25 25])
        end
        
        %formatting
        axis image off
        colormap(turbo)
        set(gca, 'YDir', 'normal')
        set(gca,'fontsize',12)

        %top labels
        if vertical == 1
            title(sprintf('\\Deltat = %d', my_time(horizontal)), 'FontWeight', 'bold')
        end

        %side labels
        if horizontal == 1
            if vertical == 1 || vertical == 4
                ax = ylabel('True', 'FontWeight', 'bold');
                set(ax, 'Rotation', 0, 'HorizontalAlignment', 'center');
            elseif vertical == 2 || vertical == 5
                ax = ylabel('POD', 'FontWeight', 'bold');
                set(ax, 'Rotation', 0, 'HorizontalAlignment', 'center');
            elseif vertical == 3 || vertical == 6
                ax = ylabel('LSTM', 'FontWeight', 'bold');
                set(ax, 'Rotation', 0, 'HorizontalAlignment', 'center');
            end
        end
        ax.Visible = 'on';

        %colorbars
        if count == 5 || count == 20
            original_pos = h.Position;
            colorbar
            h.Position = original_pos;
        end

    end
end


%% visualize both together


%plot cell
MY_PLOTS = cell(10,5);

%populate it
for ttt = 1:length(my_time)

    %extract the plots for WKRR
    true = WDMKRR_truth(my_time(ttt),:);
    u_true = reshape(true(1:51*154),[51 154]);
    v_true = reshape(true(51*154+1:2*51*154),[51 154]);

    POD = WDMKRR_POD(my_time(ttt),:);
    u_POD = reshape(POD(1:51*154),[51 154]);
    v_POD = reshape(POD(51*154+1:2*51*154),[51 154]);

    forecast = WDMKRR_forecast(my_time(ttt),:);
    u_forecast = reshape(forecast(1:51*154),[51 154]);
    v_forecast = reshape(forecast(51*154+1:2*51*154),[51 154]);

    %extract the plots for WKRR with DM kernel
    true_DM = WDMKRR_truth_DMDM(my_time(ttt),:);
    u_true_DM = reshape(true_DM(1:51*154),[51 154]);
    v_true_DM = reshape(true_DM(51*154+1:2*51*154),[51 154]);

    POD_DM = WDMKRR_POD_DMDM(my_time(ttt),:);
    u_POD_DM = reshape(POD_DM(1:51*154),[51 154]);
    v_POD_DM = reshape(POD_DM(51*154+1:2*51*154),[51 154]);

    forecast_DM = WDMKRR_forecast_DMDM(my_time(ttt),:);
    u_forecast_DM = reshape(forecast_DM(1:51*154),[51 154]);
    v_forecast_DM = reshape(forecast_DM(51*154+1:2*51*154),[51 154]);

    %extract plots for LSTM
    lstm_true_u = reshape(LSTM_truth_sol_u(ttt,:),[51 154]); %these are to make sure that the LSTM truth and POD are identical to WDMKRR truth and POD
    lstm_true_v = reshape(LSTM_truth_sol_v(ttt,:),[51 154]);

    lstm_pod_u = reshape(LSTM_POD_sol_u(ttt,:),[51 154]);
    lstm_pod_v = reshape(LSTM_POD_sol_v(ttt,:),[51 154]);

    lstm_forecast_u = reshape(LSTM_sol_u(ttt,:),[51 154]);
    lstm_forecast_v = reshape(LSTM_sol_v(ttt,:),[51 154]);

    %populate the cell plot structure
    MY_PLOTS{1,ttt} = u_true;
    MY_PLOTS{2,ttt} = u_POD;
    MY_PLOTS{3,ttt} = u_forecast;
    MY_PLOTS{4,ttt} = u_forecast_DM;
    MY_PLOTS{5,ttt} = lstm_forecast_u;
    MY_PLOTS{6,ttt} = v_true;
    MY_PLOTS{7,ttt} = v_POD;
    MY_PLOTS{8,ttt} = v_forecast;
    MY_PLOTS{9,ttt} = v_forecast_DM;
    MY_PLOTS{10,ttt} = lstm_forecast_v;
end

%big plot
figure(4)
set(gcf,'position',[12,31,1924,946])

count = 0;
for vertical = 1:10
    for horizontal = 1:5

        %update count
        count = count + 1;

        %plot it
        h = subplot(10,5,count);
        imagesc(x,y,MY_PLOTS{vertical,horizontal})

        %scale
        if vertical < 6
            clim([-100 250])
        else
            clim([-25 25])
        end
        
        %formatting
        axis image off
        colormap(turbo)
        set(gca, 'YDir', 'normal')

        %top labels
        if vertical == 1
            title(sprintf('%d\\Deltat', my_time(horizontal)), 'FontWeight', 'bold')
        end

        %side labels
        if horizontal == 1
            if vertical == 1 || vertical == 6
                ax = ylabel('True', 'FontWeight', 'bold');
                set(ax, 'Rotation', 0, 'HorizontalAlignment', 'right');
            elseif vertical == 2 || vertical == 7
                ax = ylabel('POD', 'FontWeight', 'bold');
                set(ax, 'Rotation', 0, 'HorizontalAlignment', 'right');
            elseif vertical == 3 || vertical == 8
                ax = ylabel('WKRR', 'FontWeight', 'bold');
                set(ax, 'Rotation', 0, 'HorizontalAlignment', 'right');
            elseif vertical == 4 || vertical == 9
                ax = ylabel('WKRR DM', 'FontWeight', 'bold');
                set(ax, 'Rotation', 0, 'HorizontalAlignment', 'right');
            elseif vertical == 5 || vertical == 10
                ax = ylabel('LSTM', 'FontWeight', 'bold');
                set(ax, 'Rotation', 0, 'HorizontalAlignment', 'right');
            end
        end
        ax.Visible = 'on';

        %colorbars
        if count == 5 || count == 30
            original_pos = h.Position;
            colorbar
            h.Position = original_pos;
        end

    end
end


%% compute error plot for LSTM

%plot the error for LSTM
for n = 1:32
    
    %plot
    figure(10)
    hold on
    plot(1:30,e_nmse(:,n),'color',[0.6 0.6 0.6])
    
end

%highlight chosen trajectory
plot(1:30,e_nmse(:,my_model),'--','color',[.72 .45 .2],'linewidth',2)

%compute mean
my_mean = mean(e_nmse,2);
plot(1:30,my_mean,'color',[.72 .45 .2],'linewidth',3)
xlim([0 30])
ylim([0 2])
box on
set(gca,'fontsize',14)
xlabel('\Deltat')
ylabel('Error')
title('LSTM')

%compute std
my_std  = std(e_nmse,0,2);
h_band = fill([1:30 30:-1:1],[my_mean + my_std; flipud(my_mean - my_std)],[.72 .45 .2],'EdgeColor','none','FaceAlpha',0.2);


%% compute the error plots for WiKRR

%error
WDMKRR_error = zeros(30,32);
WDMKRR_error_DMDM = zeros(30,32);

%plot the error for WiKRR
for nnn = 1:32

    %extract ground truth
    DM_truth = TrialResults(nnn).filter(1).u_true;
    DM_truth_DMDM = my_DM_results(nnn).filter(1).u_true;

    %extract forecast
    DM_sol = TrialResults(nnn).filter(1).ker_sol;
    DM_sol_DMDM = my_DM_results(nnn).filter(1).ker_sol;

    %undo per mode normalization
    DM_sol = DM_sol.*A_std + A_mu;
    DM_sol_DMDM = DM_sol_DMDM.*A_std + A_mu;

    %undo POD
    DM_sol = DM_sol * V';
    DM_sol_DMDM = DM_sol_DMDM * V';

    %undo physical normalization (make sure that mean is subtracted)
    DM_sol = DM_sol.*[u_std,v_std];
    DM_sol_DMDM = DM_sol_DMDM.*[u_std,v_std];
    
    %prune both for consistency with LSTM
    DM_truth = DM_truth(2:end,:) - [u_mean,v_mean];
    DM_sol = DM_sol(2:end,:);

    DM_truth_DMDM = DM_truth_DMDM(2:end,:) - [u_mean,v_mean];
    DM_sol_DMDM = DM_sol_DMDM(2:end,:);

    %error
    for ttt = 1:30
        WDMKRR_error(ttt,nnn) = mean((DM_truth(ttt,:) - DM_sol(ttt,:)).^2,2)./mean((DM_truth(ttt,:)).^2,2);
        WDMKRR_error_DMDM(ttt,nnn) = mean((DM_truth_DMDM(ttt,:) - DM_sol_DMDM(ttt,:)).^2,2)./mean((DM_truth_DMDM(ttt,:)).^2,2);
    end

    %plot
    figure(11)
    hold on
    plot(1:30,WDMKRR_error(:,nnn),'color',[0.6 0.6 0.6])
    plot(1:30,WDMKRR_error_DMDM(:,nnn),'color',[0.6 0.6 0.6])
end

%highlight chosen trajectory
plot(1:30,WDMKRR_error(:,my_model),'--','color',[0.45, 0.20, 0.60],'linewidth',2)
plot(1:30,WDMKRR_error_DMDM(:,my_model),'--','color',[1, 0, 0],'linewidth',2)

%compute mean
DM_my_mean = mean(WDMKRR_error,2);
DM_my_mean_DMDM = mean(WDMKRR_error_DMDM,2);
plot(1:30,DM_my_mean,'color',[0.45, 0.20, 0.60],'linewidth',3)
plot(1:30,DM_my_mean_DMDM,'color',[1, 0, 0],'linewidth',3)
xlim([0 30])
ylim([0 2])
box on
set(gca,'fontsize',14)
xlabel('\Deltat')
ylabel('Error')
title('WKRR')

%compute std
DM_my_std  = std(WDMKRR_error,0,2);
fill([1:30 30:-1:1],[DM_my_mean + DM_my_std; flipud(DM_my_mean - DM_my_std)],[0.75, 0.60, 0.85],'EdgeColor','none','FaceAlpha',0.2);

DM_my_std_DMDM  = std(WDMKRR_error_DMDM,0,2);
fill([1:30 30:-1:1],[DM_my_mean_DMDM + DM_my_std_DMDM; flipud(DM_my_mean_DMDM - DM_my_std_DMDM)],[1, 0.45, 0.45],'EdgeColor','none','FaceAlpha',0.2);


%% overlay the plots

%overlay it
figure(12)
hold on
plot(1:30,DM_my_mean,'color',[0.45, 0.20, 0.60],'linewidth',3)
fill([1:30 30:-1:1],[DM_my_mean + DM_my_std; flipud(DM_my_mean - DM_my_std)],[0.75, 0.60, 0.85],'EdgeColor','none','FaceAlpha',0.2);
plot(1:30,DM_my_mean_DMDM,'color',[1, 0, 0],'linewidth',3)
fill([1:30 30:-1:1],[DM_my_mean_DMDM + DM_my_std_DMDM; flipud(DM_my_mean_DMDM - DM_my_std_DMDM)],[1, 0.45, 0.45],'EdgeColor','none','FaceAlpha',0.2);
plot(1:30,my_mean,'color',[.72 .45 .2],'linewidth',3)
fill([1:30 30:-1:1],[my_mean + my_std; flipud(my_mean - my_std)],[.72 .45 .2],'EdgeColor','none','FaceAlpha',0.2);
xlim([0 30])
ylim([0 2])
box on
set(gca,'fontsize',14)
xlabel('\Deltat')
ylabel('Error')
legend('WKRR','std','WKRR DM','std','LSTM','std','Location','southeast')
title('Comparison')
