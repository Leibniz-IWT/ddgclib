"""
detachment_rates_case.py
=======================

Fresh test case for the manuscript story:

    "Neglecting composition-dependent surface tension (the usual literature
     assumption of a fixed sigma) biases the predicted bubble detachment SIZE,
     detachment FREQUENCY and GAS-PRODUCTION rate; the bias is systematic and
     grows with mixture non-ideality."

It reuses the abstracted backend:
    * bubble_enrtl_electrostatic_toy : e-NRTL / Butler sigma_lv, Young-Laplace,
                                       max-closable-volume detachment
    * pulloff_models                 : Electrowetting(THETA), swappable sigma
    * bubble_kinetics                : Nernst polarization, reaction/growth/rate,
                                       Monte-Carlo uncertainty
    * systems                        : ElectrochemicalSystem registry

and adds only case orchestration + generalised figures, driven by a LIST of
``ElectrochemicalSystem`` objects so new systems (other salts, later regolith)
drop in without code changes.

Run:  python detachment_rates_case.py
Out:  fig/detachment_rates/*.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

import bubble_enrtl_electrostatic_toy as toy
from pulloff_models import ConstantSigma
from bubble_kinetics import (
    detachment_rate, monte_carlo, UncertainParam, ReactionRate, BubbleGrowth,
    NernstDiffusionLayer,
)
from systems import SYSTEMS, KOH_WATER, H2SO4_WATER

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_HERE, 'fig', 'detachment_rates')


def _style():
    plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.3,
                         'figure.dpi': 110, 'lines.linewidth': 2.0})


def _Vd(system, m, j=0.0, E_cell=None, sigma_model=None, electrowetting=True):
    """Detachment result for a system at bulk molality m (with Nernst-shifted
    local m if j>0), returning (V_d, D_d, sigma_eff, theta_eff, m_surface)."""
    m_surf = system.nernst().surface_molality(m, j) if j > 0 else m
    stm = sigma_model or system.sigma_model
    sigma = stm.sigma_lv(m_surf, system.salt)
    r = toy.detachment_volume(sigma, system.theta_0,
                              models=system.pulloff_models(electrowetting),
                              env=system.env(j=j, E_cell=E_cell))
    if not r['converged']:
        return (np.nan,) * 5
    return r['V'], r['D'], r['sigma_eff'], r['theta_eff'], m_surf


# ---------------------------------------------------------------------------
# Figure 1: generalised non-ideality -- sigma_lv(m) and the detachment bias
#           it induces, e-NRTL vs the fixed-sigma literature assumption.
# ---------------------------------------------------------------------------
def fig_nonideality(systems, m_max=None):
    _style()
    fig, (axS, axB) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    colors = ['C0', 'C3', 'C2', 'C4']
    for i, sys in enumerate(systems):
        c = colors[i % len(colors)]
        mmax = m_max or (10.0 if 'KOH' in sys.name else 6.0)
        m = np.linspace(0.05, mmax, 40)
        sig_enrtl = np.array([sys.sigma_lv(mi) for mi in m])
        sig_const = sig_enrtl[0]                      # literature: fix at low-m value
        axS.plot(m, 1e3 * sig_enrtl, color=c, label=f'{sys.name}: e-NRTL')
        axS.axhline(1e3 * sig_const, color=c, ls=':', alpha=0.7)
        # downstream detachment-diameter bias vs constant-sigma
        Dd_enrtl = np.array([_Vd(sys, mi)[1] for mi in m])
        stm_const = ConstantSigma(sig_const)
        Dd_const = np.array([_Vd(sys, mi, sigma_model=stm_const)[1] for mi in m])
        axB.plot(m, 100 * (Dd_enrtl - Dd_const) / Dd_const, color=c,
                 label=f'{sys.name}')
    axS.set_xlabel('bulk molality  m  (mol/kg)')
    axS.set_ylabel(r'$\sigma_{lv}$  (mN/m)')
    axS.set_title('(a) e-NRTL $\\sigma_{lv}(m)$ (solid) vs fixed-$\\sigma$ (dotted)\n'
                  'strongly non-linear; salt raises $\\sigma$')
    axS.legend(fontsize=8)
    axB.axhline(0, color='k', lw=1)
    axB.set_xlabel('bulk molality  m  (mol/kg)')
    axB.set_ylabel(r'detachment-diameter bias  $\Delta D_d / D_d$  (%)')
    axB.set_title('(b) model-form BIAS from neglecting $\\sigma(m)$\n'
                  'systematic, grows with molality')
    axB.legend(fontsize=8)
    fig.suptitle('Generalised mixture-non-ideality effect on bubble detachment',
                 y=1.02, fontsize=12)
    fig.tight_layout()
    _save(fig, 'fig1_nonideality.png')


# ---------------------------------------------------------------------------
# Figure 2: detachment size, frequency and gas rate vs current density,
#           with concentration polarization (Nernst) coupling j -> local m.
# ---------------------------------------------------------------------------
def fig_rates_vs_current(system, m_bulk=1.0):
    _style()
    j_grid = np.logspace(1, np.log10(0.8 * system.nernst().limiting_current(
        m_bulk * 997 / (1 + m_bulk * 0.018))), 30)
    Dd, freq, gas = [], [], []
    Dd_const = []
    stm_const = ConstantSigma(system.sigma_lv(m_bulk))    # fix at bulk value
    for j in j_grid:
        V, D, sig, th, ms = _Vd(system, m_bulk, j=j)
        res = detachment_rate(V, sig, th, ms, j, system.reaction(), system.growth())
        Dd.append(D * 1e3); freq.append(res.f_detach); gas.append(res.gas_rate_area)
        Vc, Dc, *_ = _Vd(system, m_bulk, j=j, sigma_model=stm_const)
        Dd_const.append(Dc * 1e3)
    Dd, freq, gas, Dd_const = map(np.array, (Dd, freq, gas, Dd_const))

    fig, (axD, axF) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axD.semilogx(j_grid, Dd, 'C0-o', ms=3, label='e-NRTL $\\sigma(m_{surf})$')
    axD.semilogx(j_grid, Dd_const, 'k--', label='fixed $\\sigma$ (literature)')
    axD.set_xlabel('current density  j  (A/m$^2$)')
    axD.set_ylabel(r'detachment diameter $D_d$  (mm)')
    axD.set_title('(a) $D_d$ vs $j$: Nernst polarization shifts local $m$,\n'
                  'e-NRTL and fixed-$\\sigma$ diverge')
    axD.legend(fontsize=8)
    axF.loglog(j_grid, freq, 'C2-o', ms=3, label='detachment frequency $f$')
    axF.set_xlabel('current density  j  (A/m$^2$)')
    axF.set_ylabel(r'detachment frequency $f$  (Hz)')
    axF.set_title('(b) $f = Q/V_d \\propto j$  (STUB kinetics: Faradaic + '
                  'constant-flux growth)')
    axF.legend(fontsize=8)
    fig.suptitle(f'Detachment rate chain: {system.name}  (m={m_bulk} mol/kg)',
                 y=1.02, fontsize=12)
    fig.tight_layout()
    _save(fig, 'fig2_rates_vs_current.png')


# ---------------------------------------------------------------------------
# Figure 3: uncertainty -- Monte-Carlo bands on D_d and f, AND the constant-
#           sigma BIAS shown to exceed the parameter scatter.
# ---------------------------------------------------------------------------
def fig_uncertainty(system, m_bulk=5.0, j=1000.0, n_mc=250):
    _style()

    def evaluate(draw):
        # perturb sigma (proxy for e-NRTL parameter uncertainty), theta_0, delta
        m_surf = NernstDiffusionLayer(delta=draw['delta'], D=system.D,
                                      n=system.polar_n, sign=system.polar_sign
                                      ).surface_molality(m_bulk, j)
        sigma = system.sigma_lv(m_surf) * draw['sigma_scale']
        th0 = np.deg2rad(system.theta_0_deg * draw['theta_scale'])
        r = toy.detachment_volume(sigma, th0, models=system.pulloff_models(),
                                  env=system.env(j=j))
        if not r['converged']:
            return {}
        growth = BubbleGrowth(site_density=draw['site_density'],
                              capture_eff=system.capture_eff)
        res = detachment_rate(r['V'], r['sigma_eff'], r['theta_eff'], m_surf, j,
                              system.reaction(), growth)
        return {'D_d_mm': res.D_d * 1e3, 'f_Hz': res.f_detach}

    params = [
        UncertainParam('sigma_scale', 1.0, rel_sigma=0.05),      # +/-5% sigma
        UncertainParam('theta_scale', 1.0, rel_sigma=0.10),      # +/-10% theta_0
        UncertainParam('delta', system.delta, dist='lognormal', rel_sigma=0.4),
        UncertainParam('site_density', system.site_density, dist='lognormal',
                       rel_sigma=0.7),
    ]
    mc = monte_carlo(evaluate, params, n=n_mc, seed=1)

    # constant-sigma prediction (the LITERATURE assumption: fix sigma at the
    # dilute / pure-water value ~72 mN/m, ignoring composition) at the same point
    stm_const = ConstantSigma(system.sigma_lv(0.05))     # dilute-limit sigma
    m_surf = system.nernst().surface_molality(m_bulk, j)
    Vc, Dc, *_ = _Vd(system, m_bulk, j=j, sigma_model=stm_const)

    fig, (axD, axF) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, key, lab, ref in [(axD, 'D_d_mm', 'detachment diameter $D_d$ (mm)',
                               Dc * 1e3),
                              (axF, 'f_Hz', 'detachment frequency $f$ (Hz)', None)]:
        s = mc[key]['samples']
        ax.hist(s, bins=30, color='C0', alpha=0.7, density=True,
                label='e-NRTL MC ($\\pm$ param unc.)')
        ax.axvline(mc[key]['p50'], color='C0', lw=2, label='median')
        ax.axvspan(mc[key]['p05'], mc[key]['p95'], color='C0', alpha=0.15,
                   label='90% band')
        if ref is not None:
            ax.axvline(ref, color='k', ls='--', lw=2,
                       label=f'fixed-$\\sigma$ (bias)')
        ax.set_xlabel(lab)
        ax.set_ylabel('density')
        ax.legend(fontsize=8)
    axD.set_title('(a) $D_d$: constant-$\\sigma$ bias vs parameter scatter')
    axF.set_title('(b) $f$: uncertainty propagates from size to rate')
    fig.suptitle(f'Uncertainty & model-form bias: {system.name}  '
                 f'(m={m_bulk}, j={j:.0f} A/m$^2$, N={n_mc})',
                 y=1.02, fontsize=12)
    fig.tight_layout()
    _save(fig, 'fig3_uncertainty.png')

    # console summary
    print('  D_d  e-NRTL median = %.3f mm  [90%%: %.3f-%.3f]  |  fixed-sigma = %.3f mm'
          % (mc['D_d_mm']['p50'], mc['D_d_mm']['p05'], mc['D_d_mm']['p95'], Dc * 1e3))
    print('  f    e-NRTL median = %.4f Hz  [90%%: %.4f-%.4f]'
          % (mc['f_Hz']['p50'], mc['f_Hz']['p05'], mc['f_Hz']['p95']))


def _save(fig, name):
    os.makedirs(_FIG, exist_ok=True)
    fig.savefig(os.path.join(_FIG, name), dpi=140, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f'Wrote {os.path.join(_FIG, name)}')


def main():
    systems = [KOH_WATER, H2SO4_WATER]
    print('=== Fig 1: generalised non-ideality ===')
    fig_nonideality(systems)
    print('=== Fig 2: detachment rate chain vs current ===')
    fig_rates_vs_current(KOH_WATER, m_bulk=1.0)
    print('=== Fig 3: uncertainty & model-form bias ===')
    fig_uncertainty(KOH_WATER, m_bulk=5.0, j=1000.0)


if __name__ == '__main__':
    main()
