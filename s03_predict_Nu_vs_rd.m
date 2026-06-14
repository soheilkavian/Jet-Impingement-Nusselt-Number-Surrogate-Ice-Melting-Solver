% s03_predict_Nu_vs_rd.m
% =============================================================================
%   STEP 3 -- Load a TRAINED surrogate (NN or PINN) and use it to predict how
%             the local Nusselt number Nu varies with radial position r/d.
%
%   At start-up you are asked which model to use:
%       1 = Baseline NN   (ml_Nu_baseline.mat        from s01)
%       2 = PINN-A        (trained_models/pinnA_model.mat from s02)
%
%   The script then evaluates the chosen model over a sweep of r/d for TWO
%   representative operating points and plots Nu(r/d):
%
%       (a) LAMINAR     high-Prandtl oil jet   Re = 1305,  Pr_jet = 167,
%                                              Pr_wall = 167,  H/d = 4.85
%       (b) TURBULENT   water jet              Re = 4.0e4, Pr_jet = 6.1,
%                                              Pr_wall = 6.1,  H/d = 9.2
%
%   Both points are jet-impingement validation cases from the paper.  Only the
%   radial coordinate r/d is swept; the four other inputs are held fixed.
%
%   Run:  >> s03_predict_Nu_vs_rd
% =============================================================================
clear; clc; close all;
thisDir = fileparts(mfilename('fullpath'));
cd(thisDir);

% --- choose the model --------------------------------------------------------
choice = '';
while ~ismember(choice, {'1','2'})
    choice = strtrim(input('Which trained model?  [1] NN   [2] PINN  : ','s'));
end
useNN = strcmp(choice,'1');

if useNN
    f = fullfile(thisDir,'ml_Nu_baseline.mat');
    assert(isfile(f), 's01_train_NN.m has not produced ml_Nu_baseline.mat yet.');
    L = load(f,'model','meta');
    predictNu = @(Xraw) predictNN(Xraw, L.model, L.meta);
    modelName = 'Baseline NN';
else
    f = fullfile(thisDir,'trained_models','pinnA_model.mat');
    assert(isfile(f), 's02_train_PINN.m has not produced pinnA_model.mat yet.');
    M = load(f);
    predictNu = @(Xraw) predict_pinnA(Xraw, M);
    modelName = 'PINN-A';
end
fprintf('Using model: %s\n', modelName);

% --- two operating points ----------------------------------------------------
rd = linspace(0, 6.5, 200).';            % radial sweep
cases = struct( ...
  'name', {'Laminar (oil jet)','Turbulent (water jet)'}, ...
  'Re',   {1305,               4.0e4}, ...
  'Prj',  {167,                6.1}, ...
  'Prw',  {167,                6.1}, ...
  'Hd',   {4.85,               9.2});

figure('Color','w','Name',['Nu(r/d) -- ' modelName],'Position',[100 140 1080 460]);
for k = 1:2
    c   = cases(k);
    Xr  = [c.Re*ones(size(rd)), c.Prj*ones(size(rd)), ...
           c.Prw*ones(size(rd)), c.Hd*ones(size(rd)), rd];
    Nu  = predictNu(Xr);

    subplot(1,2,k);
    plot(rd, Nu, '-', 'LineWidth', 2.2, 'Color', [0.17 0.43 0.61]); grid on;
    xlabel('r / d'); ylabel('Nu');
    title({sprintf('%s  --  %s', modelName, c.name), ...
           sprintf('Re = %.0f,  Pr_{jet} = %.1f,  Pr_{wall} = %.1f,  H/d = %.2f', ...
                   c.Re, c.Prj, c.Prw, c.Hd)}, 'FontWeight','normal');
    xlim([0 max(rd)]);
    fprintf('%-22s : Nu(stag) = %.1f,  Nu(r/d=3) = %.1f\n', ...
            c.name, Nu(1), interp1(rd,Nu,3));
end
sgtitle(sprintf('Predicted radial Nusselt-number profile  (%s)', modelName));

% =============================================================================
function Nu = predictNN(Xraw, model, meta)
% Forward-evaluate the baseline NN: log10/z-score features, run the net,
% then undo the log10 on the output.  Handles both fitrnet and feedforwardnet.
    Xt        = Xraw;
    Xt(:,1:3) = log10(Xraw(:,1:3));
    Xs        = (Xt - meta.mu) ./ meta.sigma;
    if strcmpi(meta.backend, 'fitrnet')
        yl = predict(model, Xs);
    else
        yl = model(Xs').';            % feedforwardnet
    end
    Nu = 10.^yl;
end
