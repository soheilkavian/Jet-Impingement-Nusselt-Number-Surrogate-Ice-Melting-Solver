function Nu = predict_pinnA(Xraw, M)
% predict_pinnA  Evaluate the trained final PINN-A model on raw features.
%
%   Nu = predict_pinnA(Xraw, M)
%
%   INPUTS
%     Xraw : N x 5 matrix of raw features
%             columns = [Re, Pr_jet, Pr_wall, H/d, r/d]
%     M    : (a) struct loaded from pinnA_model.mat (M = load(...)), or
%            (b) char/string path to a pinnA_model.mat file
%
%   OUTPUT
%     Nu   : N x 1 predicted Nusselt number
%
%   The full PINN-A formula is:
%       log10(Nu) = Phi(Xraw) * A_coef   +   predict(nnModel, z(Xraw))
%
%   where
%       Phi(Xraw) = [1, log10(Re), log10(Pr_jet), log10(Pr_jet/Pr_wall),
%                    log10(H/d),   log10(r/d + 1)]
%       z(Xraw)   = ((log10([Re Pr_jet Pr_wall]) , H/d , r/d) - mu) ./ sigma
%
%   The closed-form Phi*A captures the Sieder-Tate-like physics; the NN
%   residual adds <1% MAPE correction.  The (r/d + 1) form (V3) keeps the
%   correlation contribution finite at the stagnation point.
%
%   Companion: train_pinnA_final.m  (produces pinnA_model.mat)

    if ischar(M) || (isstring(M) && isscalar(M))
        M = load(char(M));
    end

    % --- Phi (closed-form basis) ---
    Phi = [ones(size(Xraw,1),1), ...
           log10(Xraw(:,1)), ...
           log10(Xraw(:,2)), ...
           log10(Xraw(:,2)./Xraw(:,3)), ...
           log10(Xraw(:,4)), ...
           log10(Xraw(:,5) + 1)];

    % --- standardized features for the NN residual ---
    Xt        = Xraw;
    Xt(:,1:3) = log10(Xraw(:,1:3));          % log-transform Re, Pr_jet, Pr_wall
    Xs        = (Xt - M.mu) ./ M.sigma;

    % --- combine ---
    log10Nu = Phi * M.A_coef + predict(M.nnModel, Xs);
    Nu      = 10.^log10Nu;
end
