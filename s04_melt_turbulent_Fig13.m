% s04_melt_turbulent_Fig13.m
% =============================================================================
%   STEP 4 -- Load the trained PINN-A and run the reduced-order ice-melting
%             solver for the TURBULENT operating point, animating the
%             liquid-fraction field f(r,z,t) live.  Reproduces the left
%             ("present model") column of Figure 13 of the paper.
%
%   Physics (axisymmetric enthalpy / Stefan model)
%   ----------------------------------------------
%     * 2-D (r,z) finite-volume grid; enthalpy state -> (T, liquid fraction f).
%     * The PINN-A supplies the convective coefficient h(r,t) = Nu*k/d at the
%       moving melt interface, with Nu = PINN(Re, Pr_jet, Pr_wall(T_int),
%       H/d(r,t), r/d).  As the cavity deepens, the local stand-off H/d grows.
%     * Soft wash-out: fully-liquid cells relax toward jet enthalpy with
%       timescale tau_wash (open, constantly refreshed jet).  Conductivity is
%       set to 0 in fully-liquid cells (intentional -- see note at k_cell).
%
%   Turbulent operating point (paper Fig. 13):  v_jet = 14 m/s,  T_jet = 80 degC.
%   (Set v_jet = 8, T_jet = 40 for the laminar companion case.)
%
%   NOTE: this is a faithful but trimmed port of ice_melt_pinnA_v5.m -- the
%   passive glycerol tracer used only for visualisation in the original has
%   been removed so the script stays focused on the melting field.
%
%   Runtime: ~30-60 s (160x160 grid, explicit time stepping to t = 5 s).
%
%   Run:  >> s04_melt_turbulent_Fig13
% =============================================================================
clear; clc; close all;
thisDir = fileparts(mfilename('fullpath'));
cd(thisDir);

% --- 1. trained PINN-A -------------------------------------------------------
mf = fullfile(thisDir,'trained_models','pinnA_model.mat');
assert(isfile(mf), 'Run s02_train_PINN.m first to create pinnA_model.mat.');
pinnA   = load(mf);
NuPINNA = @(Xraw) predict_pinnA(Xraw, pinnA);
fprintf('PINN-A loaded.\n');

% --- 2. operating point (TURBULENT, Fig. 13) ---------------------------------
T_jet  = 80;        % degC   jet bulk temperature
v_jet  = 14.0;      % m/s    jet exit velocity
T_ice0 = -20;       % degC   initial ice temperature
d_jet  = 5e-3;      % m      nozzle diameter
H_d    = 4;         % -      initial nozzle-to-surface stand-off
tau_wash = 0.05;    % s      wash-out relaxation timescale
t_end    = 5;       % s      simulated time (Fig. 13 spans t = 1..5 s)
snap_t   = [1 2 3 5];               % times (s) captured for the montage

% --- glycerol jet property correlations (T in degC -> SI) --------------------
mu_g  = @(Tc) exp(-26.61 + 7950./(Tc+273.15));      % Pa.s
rho_g = @(Tc) 1273 - 0.65*Tc;                       % kg/m^3
k_g   = @(Tc) 0.286 + 1.0e-4*(Tc-20);               % W/m/K
cp_g  = @(Tc) 2386 + 6.4*(Tc-20);                   % J/kg/K
Pr_g  = @(Tc) mu_g(Tc).*cp_g(Tc)./k_g(Tc);
clampTw = @(Tw) max(-30, min(80, Tw));              % keep BL props in range

rho_jet = rho_g(T_jet);  mu_jet = mu_g(T_jet);
k_jet   = k_g(T_jet);    cp_jet = cp_g(T_jet);
Pr_jet  = mu_jet*cp_jet/k_jet;
Re_d    = rho_jet*v_jet*d_jet/mu_jet;
fprintf('Turbulent point: T_jet=%.0fC  v=%.0f m/s  ->  Re_d=%.0f  Pr_jet=%.1f\n', ...
        T_jet, v_jet, Re_d, Pr_jet);

% --- 3. ice / water properties -----------------------------------------------
rho_i=917; c_i=2090; k_i=2.22;  rho_w=1000; c_w=4186; k_w=0.60;  Lf=334e3;
P.rho_i=rho_i; P.c_i=c_i; P.rho_w=rho_w; P.c_w=c_w; P.H_LF=rho_i*Lf; P.T_melt=0;

% --- 4. axisymmetric finite-volume grid --------------------------------------
R_dom=8*d_jet; Z_dom=12e-3; nr=160; nz=160;
dr=R_dom/nr; dz=Z_dom/nz;
r_c=((1:nr).'-0.5)*dr; z_c=((1:nz)-0.5)*dz;
Ar=2*pi*((1:nr-1).'*dr)*dz; Az=2*pi*r_c*dr; V=2*pi*r_c*dr*dz;
rd=r_c/d_jet;

% --- 5. h(r,t) from PINN-A ---------------------------------------------------
h_of = @(Tw,Hd) NuPINNA([Re_d*ones(nr,1), Pr_jet*ones(nr,1), ...
                         Pr_g(clampTw(Tw)), Hd, rd]) * k_jet/d_jet;

% --- 6. initial enthalpy field and time stepping setup -----------------------
H_e = rho_i*c_i*T_ice0*ones(nr,nz);
H_jet = P.rho_i*P.c_i*P.T_melt + P.H_LF + P.rho_w*P.c_w*(T_jet - P.T_melt);
alpha_max = max(k_i/(rho_i*c_i), k_w/(rho_w*c_w));
dt = 0.30*min(dr,dz)^2/alpha_max;
n_step = ceil(t_end/dt);
plot_every = max(1, round(0.1/dt));      % refresh ~ every 0.1 s
w_wash = 1 - exp(-dt/max(tau_wash,eps));
[T,f] = pure_state(H_e,P);

% --- 7. live figure ----------------------------------------------------------
fig = figure('Color','w','Position',[60 80 900 760], ...
             'Name','Turbulent ice melting -- liquid fraction (Fig. 13 left column)');
axF = subplot(2,1,1);
imgF = imagesc(axF, r_c*1e3, z_c*1e3, f.'); axis(axF,'image');
set(axF,'YDir','reverse'); colormap(axF, flipud(bone(256))); caxis(axF,[0 1]);
cF=colorbar(axF); ylabel(cF,'liquid fraction f'); hold(axF,'on');
[~,cont]=contour(axF, r_c*1e3, z_c*1e3, f.', [0.5 0.5],'r','LineWidth',2);
xlabel(axF,'r (mm)'); ylabel(axF,'depth z (mm)');

axT = subplot(2,1,2);
imgT = imagesc(axT, r_c*1e3, z_c*1e3, T.'); axis(axT,'image');
set(axT,'YDir','reverse'); colormap(axT, jet(256)); caxis(axT,[T_ice0 T_jet]);
cT=colorbar(axT); ylabel(cT,'T (\circC)');
xlabel(axT,'r (mm)'); ylabel(axT,'depth z (mm)');
sgtitle(sprintf('Turbulent jet (T=%.0f\\circC, Re=%.0f) on ice -- present reduced-order model', T_jet, Re_d));
drawnow;

% --- 8. time loop ------------------------------------------------------------
snaps = {};  snap_times = [];  next_snap = 1;
t = 0;  tic;
for step = 1:n_step
    [T,f] = pure_state(H_e,P);

    % moving interface: topmost not-fully-liquid cell, and its temperature
    s_r = compute_melt_depth(f,z_c,dz,Z_dom,nr);
    Hd_r = H_d + s_r/d_jet;                       % stand-off grows as cavity deepens
    T_int = zeros(nr,1);  j_int = zeros(nr,1);
    for i=1:nr
        ji = find(f(i,:)<1,1,'first');
        if isempty(ji), T_int(i)=T_jet; j_int(i)=0;
        else,           T_int(i)=T(i,ji); j_int(i)=ji; end
    end

    h_r = h_of(T_int, Hd_r);                      % PINN-A convective coefficient

    % conduction (k = 0 in fully-liquid cells is intentional; wash-out governs them)
    kc = k_i*(f<=0) + (k_i+(k_w-k_i).*f).*(f>0 & f<1);
    kh = 2*kc(1:nr-1,:).*kc(2:nr,:)./(kc(1:nr-1,:)+kc(2:nr,:)+eps);
    kv = 2*kc(:,1:nz-1).*kc(:,2:nz)./(kc(:,1:nz-1)+kc(:,2:nz)+eps);
    Qe = -kh.*(T(2:nr,:)-T(1:nr-1,:))/dr.*Ar;
    Qt = -kv.*(T(:,2:nz)-T(:,1:nz-1))/dz.*Az;
    dQ = zeros(nr,nz);
    dQ(1:nr-1,:)=dQ(1:nr-1,:)-Qe; dQ(2:nr,:)=dQ(2:nr,:)+Qe;
    dQ(:,1:nz-1)=dQ(:,1:nz-1)-Qt; dQ(:,2:nz)=dQ(:,2:nz)+Qt;

    % convective heat injected into each interface cell
    for i=1:nr
        if j_int(i)>0
            dQ(i,j_int(i)) = dQ(i,j_int(i)) + h_r(i)*(T_jet-T(i,j_int(i)))*Az(i);
        end
    end

    H_e = H_e + dt*dQ./V;                          % explicit Euler

    % soft wash-out of fully-liquid cells toward jet enthalpy
    if w_wash>0
        for i=1:nr
            if j_int(i)>1
                H_e(i,1:j_int(i)-1) = (1-w_wash)*H_e(i,1:j_int(i)-1) + w_wash*H_jet;
            elseif j_int(i)==0
                H_e(i,:) = (1-w_wash)*H_e(i,:) + w_wash*H_jet;
            end
        end
    end
    t = t + dt;

    % live refresh + snapshot capture
    if mod(step,plot_every)==0 || step==n_step
        [T,f] = pure_state(H_e,P);
        set(imgF,'CData',f.'); set(imgT,'CData',T.');
        if isvalid(cont), delete(cont); end
        [~,cont]=contour(axF, r_c*1e3, z_c*1e3, f.', [0.5 0.5],'r','LineWidth',2);
        title(axF, sprintf('liquid fraction f(r,z)   --   t = %4.1f s', t));
        title(axT, sprintf('temperature T(r,z)   --   t = %4.1f s', t));
        drawnow limitrate;
    end
    if next_snap<=numel(snap_t) && t>=snap_t(next_snap)
        [~,fsnap]=pure_state(H_e,P);
        snaps{end+1}=fsnap; snap_times(end+1)=snap_t(next_snap);   %#ok<SAGROW>
        next_snap=next_snap+1;
    end
end
fprintf('Done in %.1f s wall-clock. Max melt depth = %.2f mm.\n', toc, max(s_r)*1e3);

% --- 9. Fig-13-style montage of the captured liquid-fraction snapshots --------
figure('Color','w','Position',[120 120 360 900], ...
       'Name','Fig. 13 (left column) -- present model, turbulent case');
ns=numel(snaps);
for s=1:ns
    subplot(ns,1,s);
    imagesc(r_c*1e3, z_c*1e3, snaps{s}.'); axis image; set(gca,'YDir','reverse');
    colormap(flipud(bone(256))); caxis([0 1]); hold on;
    contour(r_c*1e3, z_c*1e3, snaps{s}.', [0.5 0.5],'r','LineWidth',1.5);
    ylabel('z (mm)'); title(sprintf('t = %g s', snap_times(s)));
    if s==ns, xlabel('r (mm)'); end
end
sgtitle('Liquid fraction f -- present model (turbulent)');

% =============================================================================
% Local functions
% =============================================================================
function s_r = compute_melt_depth(f,z_c,dz,Z_dom,nr)
    s_r = zeros(nr,1);
    for i=1:nr
        fi=f(i,:);
        if fi(1)<0.5,            s_r(i)=0;
        elseif fi(end)>=0.5,     s_r(i)=Z_dom;
        else
            j=find(fi<0.5,1,'first'); f1=fi(j-1); f2=fi(j);
            s_r(i)=z_c(j-1)+dz*(f1-0.5)/(f1-f2+eps);
        end
    end
    s_r=max(0,min(Z_dom,s_r));
end

function [T,f] = pure_state(H,P)
% Enthalpy -> (temperature, liquid fraction) for pure water at melt point P.T_melt.
    H_sol = P.rho_i*P.c_i*P.T_melt;  H_liq = H_sol + P.H_LF;
    T = zeros(size(H)); f = zeros(size(H));
    is = H<H_sol; il = H>H_liq; im = ~is & ~il;
    T(is)=H(is)./(P.rho_i*P.c_i);                       f(is)=0;
    T(il)=P.T_melt+(H(il)-H_liq)./(P.rho_w*P.c_w);      f(il)=1;
    T(im)=P.T_melt;                                     f(im)=(H(im)-H_sol)./P.H_LF;
end
