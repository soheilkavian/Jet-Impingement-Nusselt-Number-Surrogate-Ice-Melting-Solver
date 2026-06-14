% s01_train_NN.m
% =============================================================================
%   STEP 1 -- Load the COMSOL Nusselt-number database and train the baseline
%             data-driven neural network (NN) surrogate.
%
%   Pipeline
%   --------
%     1. Load simulation_database.mat (168 jet-impingement cases) and flatten
%        it into (X, y) with a case-grouped 80/20 train/test split.        [prepare_dataset.m]
%     2. Inputs  X = [Re, Pr_jet, Pr_wall, H/d, r/d]   (Re, Pr in log10).
%        Output  y = log10(Nu).
%     3. Train a [32 32] tanh feed-forward network (fitrnet; falls back to
%        feedforwardnet if the Statistics & ML Toolbox is absent).
%     4. Report R^2 / RMSE / MAE / MAPE (in linear Nu space) and a parity plot.
%     5. Save the trained model + scaling stats to ml_Nu_baseline.mat.
%
%   This network is the "NN" curve in the paper.  It interpolates well but
%   over/under-shoots when extrapolated beyond the training Reynolds range;
%   compare it with the physics-informed PINN trained in s02_train_PINN.m.
%
%   Run:  >> s01_train_NN
% =============================================================================
clear; clc; close all;
thisDir = fileparts(mfilename('fullpath'));
cd(thisDir);

% --- 1. data -----------------------------------------------------------------
ds = prepare_dataset();           % shared loader + split (rng(42))

% --- 2. train ----------------------------------------------------------------
rng(42);
useFitrnet = exist('fitrnet','file') == 2;
if useFitrnet
    fprintf('Training fitrnet [32 32] tanh ...\n');
    model = fitrnet(ds.Xtrain_s, ds.ytrain, ...
                    'LayerSizes',     [32 32], ...
                    'Activations',    'tanh', ...
                    'Standardize',    false, ...      % already z-scored
                    'IterationLimit', 1000, ...
                    'Verbose',        0);
    backend   = 'fitrnet';
    predictFn = @(Z) predict(model, Z);
else
    fprintf('fitrnet unavailable -- using feedforwardnet [32 32] ...\n');
    net = feedforwardnet([32 32], 'trainlm');        % tansig == tanh
    net.trainParam.showWindow = false;
    net.trainParam.epochs     = 500;
    net   = train(net, ds.Xtrain_s', ds.ytrain');
    model = net;  backend = 'feedforwardnet';
    predictFn = @(Z) net(Z')';
end

% --- 3. evaluate (metrics in linear Nu space) --------------------------------
yhat_tr = 10.^predictFn(ds.Xtrain_s);   y_tr = 10.^ds.ytrain;
yhat_te = 10.^predictFn(ds.Xtest_s);    y_te = 10.^ds.ytest;
[R2tr,RMSEtr,MAEtr,MAPEtr] = regMetrics(y_tr, yhat_tr);
[R2te,RMSEte,MAEte,MAPEte] = regMetrics(y_te, yhat_te);

fprintf('\n================ Baseline NN performance ================\n');
fprintf('          %8s %10s %10s %10s\n','R^2','RMSE','MAE','MAPE(%)');
fprintf('  Train : %8.4f %10.3f %10.3f %10.2f\n', R2tr,RMSEtr,MAEtr,MAPEtr);
fprintf('  Test  : %8.4f %10.3f %10.3f %10.2f\n', R2te,RMSEte,MAEte,MAPEte);
fprintf('=========================================================\n');

% --- 4. parity plot ----------------------------------------------------------
figure('Color','w','Name','Baseline NN -- parity','Position',[120 150 560 520]);
loglog(y_tr,yhat_tr,'.','MarkerSize',6,'Color',[0.17 0.43 0.61]); hold on;
loglog(y_te,yhat_te,'.','MarkerSize',9,'Color',[0.75 0.29 0.21]);
lo = min([y_tr;y_te])*0.7;  hi = max([y_tr;y_te])*1.3;
plot([lo hi],[lo hi],'--k','LineWidth',1);
axis([lo hi lo hi]); axis square; grid on;
xlabel('Nu  (COMSOL)'); ylabel('Nu  (NN)');
title(sprintf('Baseline NN   R^2_{test} = %.3f, MAPE_{test} = %.1f%%',R2te,MAPEte));
legend({'train','test','y = x'},'Location','northwest');

% --- 5. save -----------------------------------------------------------------
meta = struct('name','Baseline NN', ...
              'featNames',{ds.featNames}, ...
              'logFeatureIdx',[1 2 3], 'logTarget',true, ...
              'mu',ds.mu, 'sigma',ds.sigma, 'backend',backend, ...
              'hyperparams',struct('hidden',[32 32],'activations','tanh'), ...
              'testCases',ds.testCases, ...
              'test_metrics',struct('R2',R2te,'RMSE',RMSEte,'MAE',MAEte,'MAPE',MAPEte));
results = struct('X_test_raw',ds.Xtest_raw,'y_test_true',y_te,'y_test_pred',yhat_te);
save(fullfile(thisDir,'ml_Nu_baseline.mat'),'model','meta','results');
fprintf('\nSaved trained NN -> ml_Nu_baseline.mat\n');

% =============================================================================
function [R2,RMSE,MAE,MAPE] = regMetrics(y,yh)
    e = yh - y;
    RMSE = sqrt(mean(e.^2));  MAE = mean(abs(e));
    MAPE = 100*mean(abs(e)./max(abs(y),eps));
    R2   = 1 - sum(e.^2)/sum((y-mean(y)).^2);
end
