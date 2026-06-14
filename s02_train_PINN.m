% s02_train_PINN.m
% =============================================================================
%   STEP 2 -- Load the COMSOL database and train the physics-informed
%             neural network (PINN-A) surrogate.
%
%   Model
%   -----
%       log10(Nu) = Phi(X) * A_coef        (closed-form physics backbone)
%                 + NN_residual( z(X) )    (small data-driven correction)
%
%   with the correlation basis (Sieder-Tate-style, stagnation-regularized):
%
%       Phi(X) = [ 1, log10(Re), log10(Pr_jet), log10(Pr_jet/Pr_wall),
%                  log10(H/d), log10(r/d + 1) ]
%
%   A_coef is obtained by ordinary least squares on the training data; the
%   residual NN ([32 32] tanh) then learns the <1% MAPE deviation that the
%   correlation alone cannot capture.  Anchoring the network to the classical
%   Re/Pr scalings is what lets the PINN EXTRAPOLATE in Reynolds number far
%   better than the plain NN of s01 (the paper reports a 34% MAPE reduction in
%   Re-extrapolation), while still yielding a publishable closed-form fit.
%
%   Output:  trained_models/pinnA_model.mat  (fields A_coef, nnModel, mu,
%            sigma, featNames, metrics, ...) -- read by predict_pinnA.m.
%
%   Run:  >> s02_train_PINN
% =============================================================================
clear; clc; close all;
thisDir = fileparts(mfilename('fullpath'));
cd(thisDir);

% --- 1. data (identical split to s01) ----------------------------------------
ds = prepare_dataset();

% --- 2. closed-form backbone Phi and OLS fit ---------------------------------
Phi = @(Xr) [ones(size(Xr,1),1), ...
             log10(Xr(:,1)), ...                 % log Re
             log10(Xr(:,2)), ...                 % log Pr_jet
             log10(Xr(:,2)./Xr(:,3)), ...        % log (Pr_jet / Pr_wall)
             log10(Xr(:,4)), ...                 % log H/d
             log10(Xr(:,5) + 1)];                % log (r/d + 1)  (V3 stagnation reg)

Phi_tr = Phi(ds.Xtrain_raw);
Phi_te = Phi(ds.Xtest_raw);
A_coef = Phi_tr \ ds.ytrain;                     % 6 x 1 = [c0 a b m p q]
res_tr = ds.ytrain - Phi_tr*A_coef;              % residual the NN will learn

fprintf('Discovered closed-form exponents:\n');
fprintf('   Nu ~ %.3g * Re^%.3f * Pr_jet^%.3f * (Pr_jet/Pr_wall)^%.3f * (H/d)^%.3f * (r/d+1)^%.3f\n', ...
        10^A_coef(1), A_coef(2), A_coef(3), A_coef(4), A_coef(5), A_coef(6));

% --- 3. residual NN on standardized features ---------------------------------
rng(42);
nnModel = fitrnet(ds.Xtrain_s, res_tr, ...
                  'LayerSizes',     [32 32], ...
                  'Activations',    'tanh', ...
                  'Standardize',    false, ...
                  'IterationLimit', 1000, ...
                  'Verbose',        0);

% --- 4. evaluate (linear Nu space) -------------------------------------------
yhat_tr = 10.^(Phi_tr*A_coef + predict(nnModel, ds.Xtrain_s));  y_tr = 10.^ds.ytrain;
yhat_te = 10.^(Phi_te*A_coef + predict(nnModel, ds.Xtest_s));   y_te = 10.^ds.ytest;
R2=@(y,yh)1-sum((y-yh).^2)/sum((y-mean(y)).^2);
MAPE=@(y,yh)100*mean(abs((y-yh)./y));
m = struct('R2_train',R2(y_tr,yhat_tr),'R2_test',R2(y_te,yhat_te), ...
           'MAPE_train',MAPE(y_tr,yhat_tr),'MAPE_test',MAPE(y_te,yhat_te), ...
           'RMSE_test',sqrt(mean((y_te-yhat_te).^2)));
fprintf('\nPINN-A   test  R^2 = %.4f   MAPE = %.2f%%\n', m.R2_test, m.MAPE_test);

figure('Color','w','Name','PINN-A -- parity','Position',[120 150 560 520]);
loglog(y_tr,yhat_tr,'.','MarkerSize',6,'Color',[0.17 0.43 0.61]); hold on;
loglog(y_te,yhat_te,'.','MarkerSize',9,'Color',[0.75 0.29 0.21]);
lo=min([y_tr;y_te])*0.7; hi=max([y_tr;y_te])*1.3; plot([lo hi],[lo hi],'--k');
axis([lo hi lo hi]); axis square; grid on;
xlabel('Nu  (COMSOL)'); ylabel('Nu  (PINN)');
title(sprintf('PINN-A   R^2_{test} = %.3f, MAPE_{test} = %.1f%%', m.R2_test, m.MAPE_test));
legend({'train','test','y = x'},'Location','northwest');

% --- 5. bundle + save (fields become top-level variables in the .mat) --------
pinnA = struct('A_coef',A_coef,'nnModel',nnModel,'mu',ds.mu,'sigma',ds.sigma, ...
   'featNames',{ds.featNames}, ...
   'PhiForm','Phi=[1 log10(Re) log10(Pr_jet) log10(Pr_jet/Pr_wall) log10(H/d) log10(r/d+1)]', ...
   'metrics',m,'notes','log10(Nu)=Phi(X)*A_coef+predict(nnModel,z(X)); use predict_pinnA.', ...
   'created',datestr(now));                                          %#ok<TNOW1>
outDir = fullfile(thisDir,'trained_models');
if ~exist(outDir,'dir'); mkdir(outDir); end
save(fullfile(outDir,'pinnA_model.mat'),'-struct','pinnA');
fprintf('Saved trained PINN-A -> trained_models/pinnA_model.mat\n');

% --- 6. round-trip self-test through predict_pinnA ---------------------------
M = load(fullfile(outDir,'pinnA_model.mat'));
delta = max(abs(predict_pinnA(ds.Xtest_raw, M) - yhat_te));
fprintf('Round-trip check: max|Nu_loaded - Nu_trained| = %.2e (should be ~0)\n', delta);
