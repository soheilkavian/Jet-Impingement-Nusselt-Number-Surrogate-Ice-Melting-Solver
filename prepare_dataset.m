function ds = prepare_dataset()
% PREPARE_DATASET  Shared data-preparation helper for the Nu-surrogate trainers.
%
%   ds = PREPARE_DATASET() loads the COMSOL jet-impingement database
%   (simulation_database.mat, located in the same folder as this file),
%   flattens the 168 cases into a feature/label table, applies the log10
%   transform, and performs the SAME case-grouped 80/20 train/test split that
%   both the NN trainer (s01) and the PINN trainer (s02) rely on.  Sharing this
%   routine guarantees the two models are compared on identical data.
%
%   Features (raw)  X = [Re, Pr_jet, Pr_wall, H/d, r/d]
%   Label   (raw)   y = Nu
%
%   Transform: columns 1:3 (Re, Pr_jet, Pr_wall) and the label are taken in
%   log10 space (they span several orders of magnitude); H/d and r/d stay
%   linear.  The split is BY CASE (not by row) so that all r/d points of one
%   case stay together -- a random row split would leak information.
%
%   Returned struct ds contains, among others:
%     X_raw, y_raw, caseIdx          flattened raw data
%     Xt, yt                         log10-transformed features / label
%     isTrain, isTest, testCases     split masks
%     mu, sigma                      z-score stats (training set only)
%     Xtrain_s, Xtest_s              standardized features
%     Xtrain_raw, Xtest_raw          raw features per split
%     ytrain, ytest                  log10(Nu) per split
%     database                       the full loaded struct array

    rng(42);                                    % deterministic split

    thisDir = fileparts(mfilename('fullpath'));
    dbFile  = fullfile(thisDir, 'simulation_database.mat');
    if ~exist(dbFile, 'file')
        error(['Database not found:\n  %s\n' ...
               'The file simulation_database.mat must sit next to this script.'], dbFile);
    end

    S        = load(dbFile);
    database = S.database;
    nCases   = numel(database);

    % --- flatten the per-case radial profiles into one (X, y) table ----------
    X = [];  y = [];  caseIdx = [];
    for i = 1:nCases
        c = database(i);
        n = numel(c.r_over_d);
        if n == 0, continue; end
        block = [ repmat(c.Re,       n, 1), ...
                  repmat(c.Pr_jet,   n, 1), ...
                  repmat(c.Pr_wall,  n, 1), ...
                  repmat(c.H_over_d, n, 1), ...
                  c.r_over_d(:) ];
        X       = [X; block];                                  %#ok<AGROW>
        y       = [y; c.Nu(:)];                                %#ok<AGROW>
        caseIdx = [caseIdx; repmat(i, n, 1)];                  %#ok<AGROW>
    end
    mask    = all(isfinite(X),2) & isfinite(y) & y > 0;
    X       = X(mask,:);  y = y(mask);  caseIdx = caseIdx(mask);

    % --- log10 transform: Re, Pr_jet, Pr_wall and the label ------------------
    Xt        = X;
    Xt(:,1:3) = log10(X(:,1:3));
    yt        = log10(y);

    % --- 80/20 split BY CASE -------------------------------------------------
    allCases  = unique(caseIdx);
    nTest     = max(1, round(0.2 * numel(allCases)));
    testCases = allCases(randperm(numel(allCases), nTest));
    isTest    = ismember(caseIdx, testCases);
    isTrain   = ~isTest;

    % --- z-score from the TRAINING set only ----------------------------------
    mu    = mean(Xt(isTrain,:), 1);
    sigma = std(Xt(isTrain,:),  0, 1);
    sigma(sigma == 0) = 1;

    ds = struct();
    ds.database     = database;
    ds.featNames    = {'Re','Pr_jet','Pr_wall','H_over_d','r_over_d'};
    ds.X_raw        = X;
    ds.y_raw        = y;
    ds.caseIdx      = caseIdx;
    ds.Xt           = Xt;
    ds.yt           = yt;
    ds.isTrain      = isTrain;
    ds.isTest       = isTest;
    ds.testCases    = testCases;
    ds.mu           = mu;
    ds.sigma        = sigma;
    ds.Xtrain_raw   = X(isTrain,:);
    ds.Xtest_raw    = X(isTest,:);
    ds.Xtrain_s     = (Xt(isTrain,:) - mu) ./ sigma;
    ds.Xtest_s      = (Xt(isTest,:)  - mu) ./ sigma;
    ds.ytrain       = yt(isTrain);
    ds.ytest        = yt(isTest);
    ds.caseIdxTrain = caseIdx(isTrain);
    ds.caseIdxTest  = caseIdx(isTest);

    fprintf('Dataset : %d samples, %d cases (%d train / %d test).\n', ...
        size(X,1), numel(allCases), numel(allCases)-nTest, nTest);
end
