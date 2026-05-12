# Toy-bubble models: e-NRTL → bubble thermodynamics in water electrolysis

This document records the mathematical framework used in
[`bubble_enrtl_toy.py`](bubble_enrtl_toy.py).
Nomenclature follows the Grok draft (`grok_report.pdf`) and standard
electrolyte-thermodynamics literature.

The pipeline is four coupled layers:

```
  (A)  Symmetric-reference e-NRTL         -> G^ex(T, P, {N_i}),  μ_i,  a_w,  γ_±
  (B)  Butler surface-tension equation    -> γ(m) = σ(m)
  (C)  Axisymmetric Young-Laplace shape   -> bubble profile, V_d, r_cl
  (D)  VLE & nucleation in H_2 bubble     -> y_H2, P_bub, r*
```

The starting thermodynamic definition of surface tension (Grok, p. 3) is

$$
  \gamma^{\,j}
  \;=\;
  \left(\frac{\partial G^{\,j}}{\partial A^{\,j}}\right)_{T^{\,j}, P^{\,j},\, N_i^{\,j}}
$$

and the goal of Layer B is to realise this derivative without resorting to a
Langmuir fit, by placing the same e-NRTL model on both sides of the
liquid-gas interface (Butler equation).


## 0. Nomenclature

| Symbol | Meaning | Units |
|--------|---------|-------|
| $T$, $P$ | temperature, pressure | K, Pa |
| $N_i$, $n_i$ | amount (extensive, molar) of species $i$ | mol |
| $x_i$ | mole fraction of species $i$ | – |
| $m$ | salt molality (mol / kg water) | mol kg$^{-1}$ |
| $z_i$ | signed charge of ion $i$ | – |
| $\nu_+$, $\nu_-$, $\nu=\nu_++\nu_-$ | stoichiometric coefficients | – |
| $I_x$ | ionic strength, mole-fraction basis | – |
| $I_m$ | ionic strength, molality basis | mol kg$^{-1}$ |
| $A_\phi$ | Debye-Hückel parameter (water, 298 K) ≈ 0.392 | – |
| $\rho$ | PDH closest-approach parameter = 14.9 | – |
| $\alpha_{ij}$ | NRTL non-randomness factor (fixed = 0.2) | – |
| $\tau_{ij}$ | NRTL binary interaction parameter | – |
| $G_{ij}=\exp(-\alpha_{ij}\tau_{ij})$ | NRTL local-composition factor | – |
| $G^{\text{ex}}$ | total excess Gibbs energy | J |
| $\mu_i$ | chemical potential of species $i$ | J mol$^{-1}$ |
| $a_i=x_i\,\gamma_i$ | activity of species $i$ | – |
| $\gamma_\pm$ | mean ionic activity coefficient (molality basis) | – |
| $a_w$ | water activity | – |
| $\phi$ | osmotic coefficient | – |
| $\gamma = \sigma$ | surface tension | N m$^{-1}$ |
| $A_i$ | partial molar surface area of species $i$ | m$^2$ mol$^{-1}$ |
| $\sigma_i^{0}$ | surface tension of (hypothetical) pure component $i$ | N m$^{-1}$ |
| $k_H$ | Henry's constant (solubility form), $c = k_H\, p$ | mol kg$^{-1}$ Pa$^{-1}$ |
| $k_s$ | Sechenov (salting-out) coefficient | L mol$^{-1}$ |
| $R_\text{bub}$ | bubble radius | m |
| $\theta_g$ | contact angle measured through the gas phase | rad |
| $V_d$, $D_d$ | detachment volume, diameter | m$^3$, m |
| $r^*$ | critical nucleation radius | m |
| $S$, $S_0$ | supersaturation (effective, reference in pure water) | – |

Subscripts / superscripts used:
$c$ = cation, $a$ = anion, $w$ = water/solvent, $s$ = (lumped) salt, $B$ =
bulk, $S$ = (interfacial) surface phase, $0$ = reference / pure-water value.


## 1. Layer A — symmetric-reference e-NRTL

### 1.1 Total molar Gibbs energy of the liquid phase $j$

Following the modern (Chen/Song & Chen, Aspen-style) symmetric convention,

$$
  G^{\,j} = \sum_i x_i\, G_i^{0}(T, P)
          + R T \sum_i x_i\,\ln(x_i\,\gamma_i)
          + G^{\text{ex},\,\text{PDH}}
          + G^{\text{ex},\,\text{NRTL}},
$$

with the excess part decomposed into a long-range Pitzer-Debye-Hückel
term and a short-range local-composition (NRTL) term.

### 1.2 Long-range term (Pitzer-Debye-Hückel, PDH)

Symmetric reference on a mole-fraction basis (Chen & Evans, *AIChE J.* **32**,
444, 1986; Song & Chen, *Ind. Eng. Chem. Res.* **48**, 7788, 2009):

$$
  \frac{G^{\text{ex},\,\text{PDH}}}{n_t R T}
  \;=\;
  -\left(\frac{1000}{M_w}\right)^{\!1/2}
  \frac{4\,A_\phi\,I_x}{\rho}\,
  \ln\!\bigl(1+\rho\sqrt{I_x}\bigr),
$$

with mole-fraction ionic strength

$$
  I_x = \tfrac12 \sum_i z_i^{\,2}\, x_i,
  \qquad
  M_w = 18.015\ \text{g mol}^{-1},\quad
  A_\phi = 0.392,\quad
  \rho = 14.9.
$$

This term alone gives the DH limiting law $\ln\gamma_\pm \to -|z_+z_-|\,A_\phi\sqrt{I_m}$
as $m\to 0$, and is entirely responsible for the initial droop of $\gamma_\pm$
in Fig. 1.  Crucially it carries the charge structure, which is why KOH
($|z_c z_a|=1$, $I_m=m$) and H$_2$SO$_4$ ($|z_c z_a|=2$, $I_m=3m$) diverge
already at low $m$ — see Fig. 6.

### 1.3 Short-range term (local-composition NRTL)

For a single strong electrolyte + solvent with like-ion repulsion enforced
(the "lumped-salt" symmetric form used in the toy):

$$
  \frac{G^{\text{ex},\,\text{NRTL}}}{n_t R T}
  \;=\;
  x_w\,X_s \left[
    \frac{\tau_{sw}\,G_{sw}}{x_w + X_s\,G_{sw}}
   +\frac{\tau_{ws}\,G_{ws}}{X_s + x_w\,G_{ws}}
  \right],
$$

where $X_s = x_c + x_a$ is the total ionic mole fraction,
$G_{ij}=\exp(-\alpha\,\tau_{ij})$, and $\alpha=0.2$.  Only two parameters
per salt: $\tau_{sw}$ (salt-pair → solvent) and $\tau_{ws}$ (solvent →
salt-pair).  Like-ion repulsion is enforced by lumping the ion pair into
a single "salt" species so that $\tau_{cc}=\tau_{aa}=0$ by construction.

> **Toy-level caveat.**  The full Chen symmetric e-NRTL keeps cation and
> anion as separate interacting species with charge-weighted effective
> mole fractions $X_i = |z_i|\,x_i$ for ions.  This gives slightly different
> short-range non-ideality for asymmetric salts (e.g. H$_2$SO$_4$).  We
> adopt the lumped form here because (i) the PDH part already carries the
> dominant charge-asymmetry at low to moderate $m$, and (ii) the ion-pair
> interaction parameters for single-solvent binaries are effectively
> indistinguishable in their observable consequences (γ$_\pm$, $a_w$,
> osmotic coefficient) — the parameters would just recalibrate.

### 1.4 Chemical potentials and activity coefficients

Excess chemical potentials follow from

$$
  \mu_i^{\text{ex}}\big/RT
  \;=\;
  \left.\frac{\partial}{\partial n_i}
        \!\left(\frac{G^{\text{ex}}}{RT}\right)\right|_{T,P,\,n_{k\ne i}},
$$

and activities by

$$
  a_i = x_i\,\gamma_i = x_i\,\exp\!\bigl(\mu_i^{\text{ex}}/RT\bigr).
$$

In the code these partials are computed by symmetric finite difference on
the extensive $G^{\text{ex}}$ (`chemical_potentials` in
`bubble_enrtl_toy.py`).  This sidesteps hand-derivation errors and is
stable to well below 1 mol/kg.

### 1.5 Water activity and osmotic coefficient

At molality $m$ with $n_w = 1000/M_w$ mol water kg$^{-1}$,

$$
  a_w(m) = x_w\,\exp\!\bigl(\mu_w^{\text{ex}}/RT\bigr),
  \qquad
  \phi(m) = -\frac{\ln a_w(m)}{M_w^{\text{[kg]}}\,\nu\,m}\,.
$$

### 1.6 Mean ionic activity coefficient via Gibbs-Duhem

To avoid the unsymmetric-to-molality reference-state transformation, we
integrate the solvent activity using Gibbs-Duhem:

$$
  \ln\gamma_\pm(m) \;=\; \bigl(\phi(m)-1\bigr)
   + \int_0^{m}\frac{\phi(m')-1}{m'}\,\mathrm d m'.
$$

This gives $\gamma_\pm$ on the standard molality basis (pure water at
infinite dilution as reference). Integral is evaluated with
`scipy.integrate.cumulative_trapezoid` on a 40-point molality grid.


## 2. Layer B — Butler surface-tension equation

### 2.1 Thermodynamic definition (Grok, eq. p. 3)

$$
  \gamma^{\,j} \;=\;
  \left(\frac{\partial G^{\,j}}{\partial A^{\,j}}\right)_{T,P,\{N_i\}}.
$$

### 2.2 Butler equation (Butler, *Proc. R. Soc. A* **135**, 348, 1932)

For a flat liquid-gas interface in local thermodynamic equilibrium with the
bulk, the chemical potential of each species must be equal in the surface
phase $S$ and in the bulk $B$.  Subtracting the reference-state contribution
and using $\mu_i = \mu_i^0 + RT\ln a_i$ gives one Butler equation per species:

$$
  \boxed{\;
    \sigma \;=\; \sigma_i^{0}
    \;+\; \frac{RT}{A_i}\,\ln\!\frac{a_i^{S}}{a_i^{B}}
    \qquad \text{(each $i$)}.\;
  }
$$

For a binary water + lumped-salt system this is a 2 × 2 system in two
unknowns $(\sigma,\, x_s^{S})$.  We eliminate $\sigma$ by equating the water
and salt equations, giving one nonlinear equation in $x_s^{S}$
(see `butler_surface_tension` in the code), solved with `scipy.optimize.brentq`.

### 2.3 Activity coefficients at the surface

The Li & Lu family of papers (e.g. Li & Lu, *Langmuir* **17**, 3532, 2001;
also Hu & Lee, *Fluid Phase Equil.* **158**, 1043, 1999) couple the surface
phase to its *own* e-NRTL evaluated at the surface composition.  For a
TOY we adopt the simpler bulk-eNRTL approximation:
$\gamma_i^{S}(x^{S}) = \gamma_i^{B}(x = x^{S})$, i.e. the *same* e-NRTL
function evaluated at the surface composition.  This preserves the
Butler structure and gets the trend right — a full surface-phase
recalibration is a later refinement.

### 2.4 Partial molar surface area

We use the Stefan-Guggenheim / Sprow-Prausnitz form

$$
  A_i = f\,N_A^{\,1/3}\,V_i^{\,2/3},
  \qquad f \approx 1.091,
$$

with $V_i$ the partial molar volume. Numerical values used in the toy are
tabulated in §5.


## 3. Layer C — Axisymmetric Young-Laplace shape & detachment

### 3.1 Young-Laplace ODE

For a sessile bubble sitting on a horizontal solid at $z=0$, with apex at
$z=H$ (top) and buoyancy pulling the bubble upward, the arc-length
parameterisation from the apex $(s=0)$ is

$$
  \frac{\mathrm d r}{\mathrm d s} = \cos\varphi,\quad
  \frac{\mathrm d z_d}{\mathrm d s} = \sin\varphi,\quad
  \frac{\mathrm d \varphi}{\mathrm d s} = \frac{2}{b} - \beta\,z_d - \frac{\sin\varphi}{r},
$$

with $z_d = H - z$ the depth below apex, $b$ the apex radius of
curvature (shooting parameter), and
$\beta = (\rho_l-\rho_g)\,g / \sigma$.  Initial conditions $r(0)=0$,
$z_d(0)=0$, $\varphi(0)=0$; near-apex seeding uses the spherical expansion
$r\simeq b\sin(s/b)$, $z_d\simeq b(1-\cos(s/b))$ to avoid the $r=0$
singularity.

### 3.2 Contact-angle boundary condition

The contact angle $\theta_g$ is measured *through the gas phase* at the
three-phase line.  The contact line is reached when the tangent-angle
$\varphi$ satisfies

$$
  \varphi_{\rm cl} = \pi - \theta_g.
$$

Integration terminates on this event (`event_phi` in `young_laplace_shape`).
A secondary event catches pinch-off ($r\to 0$ again).

### 3.3 Fritz-type detachment force balance

Quasi-static detachment: buoyancy lifting the bubble off the electrode
equals the vertical component of surface tension at the pinning line,

$$
  (\rho_l - \rho_g)\,g\,V_{\rm bub}
  \;=\;
  2\pi\,r_{\rm cl}\,\sigma\,\sin\theta_g
  \qquad\text{(Fritz, } \textit{Phys. Z.}\ \textbf{36}, 379, 1935\text{)}.
$$

In the toy this is rooted on the shooting parameter $b$:
`detachment_volume` scans $b$ over a multiple of the capillary length
$\ell_c = \sqrt{\sigma/\Delta\rho\,g}$, brackets the sign change, and
refines with `brentq`.  Output: $V_d(\sigma,\theta_g)$ and
$D_d = (6V_d/\pi)^{1/3}$.


## 4. Layer D — VLE and nucleation

### 4.1 Effective Henry's constant (salting-out)

Two standard options, both produce the same trend in the toy:

1. **Sechenov** (Sechenov, *Z. Phys. Chem.* **4**, 117, 1889):
   $k_H^{\text{eff}}(m) = k_H^{0}\,10^{-k_s\,c_{\text{salt}}}$.
   Used in the toy with tabulated $k_s$ values.

2. **Water-activity form** (Krichevsky–Kasarnovsky-like approximation):
   $k_H^{\text{eff}}(m) \approx k_H^{0}\,/\,a_w(m)$, with $a_w$ from
   the same e-NRTL.  No extra parameter beyond §1.

Either route shifts solubility of a neutral gas (H$_2$) through the
non-ideal water.

### 4.2 Bubble composition (H$_2$ + H$_2$O vapour)

Raoult with activity for water + Laplace for total pressure:

$$
  p_{\rm H_2O} = a_w(m)\,p_{\rm sat}(T),
  \quad
  P_{\rm bub} = P_{\rm atm} + \frac{2\sigma(m)}{R_{\rm bub}},
  \quad
  y_{\rm H_2} = \frac{P_{\rm bub} - p_{\rm H_2O}}{P_{\rm bub}}.
$$

### 4.3 Critical nucleation radius

At fixed dissolved-H$_2$ concentration corresponding to a reference
supersaturation $S_0$ in pure water, the effective local supersaturation in
salty electrolyte becomes

$$
  S_{\text{eff}}(m) \;=\; S_0 \,\frac{k_H^{0}}{k_H^{\text{eff}}(m)},
$$

so that salting-out at constant dissolved mass **increases** the driving
force.  The Young-Laplace critical radius is then

$$
  r^{*}(m) \;=\; \frac{2\,\sigma(m)}{(S_{\text{eff}}-1)\,P_{\rm atm}}.
$$

The compound behaviour of $\sigma(m)\uparrow$ and $S_{\rm eff}(m)\uparrow$ pulls
$r^*$ down sharply as $m$ rises (Fig. 5 right panel).


## 5. Parameters used in the toy

### 5.1 e-NRTL parameters

| Salt | $z_c$ | $z_a$ | $\nu_c$ | $\nu_a$ | $\tau_{sw}$ | $\tau_{ws}$ | $\alpha$ |
|------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| KOH      | +1 | −1 | 1 | 1 | 11.5 | −4.5 | 0.2 |
| H$_2$SO$_4$ | +1 | −2 | 2 | 1 | 13.0 | −5.2 | 0.2 |

Values are "representative" — tuned only to give the qualitatively correct
$\gamma_\pm(m)$ and $a_w(m)$ shapes (DH droop at small $m$, NRTL rebound
at large $m$).  Published Aspen-style parameters for KOH-H$_2$O exist
(e.g. Thomsen & Rasmussen, *Chem. Eng. Sci.* **54**, 1787, 1999) and
should replace these once quantitative fidelity is wanted.

### 5.2 Butler parameters

| Species | $\sigma_i^{0}$ (N/m) | $A_i$ (m$^2$ mol$^{-1}$) | $V_i$ (m$^3$ mol$^{-1}$) |
|---------|-------:|-------:|-------:|
| H$_2$O | 0.07197 | 6.30 × 10$^{4}$ | 1.80 × 10$^{-5}$ |
| KOH    | 0.170   | 8.40 × 10$^{4}$ | 2.70 × 10$^{-5}$ |
| H$_2$SO$_4$ | 0.155 | 1.31 × 10$^{5}$ | 5.30 × 10$^{-5}$ |

### 5.3 Henry / salting-out / nucleation

| Parameter | Value | Note |
|-----------|------:|------|
| $k_H^{0}$ (H$_2$, 25 °C) | 7.8 × 10$^{-9}$ mol kg$^{-1}$ Pa$^{-1}$ | Sander, *Atmos. Chem. Phys.* **15**, 4399, 2015 |
| $k_s$(KOH) | 0.134 L mol$^{-1}$ | ≈ Sechenov coefficient for H$_2$ |
| $k_s$(H$_2$SO$_4$) | 0.099 L mol$^{-1}$ | ≈ Sechenov coefficient for H$_2$ |
| $\theta_g$ | 30° | hydrophobic cavity, Raman-setup-like |
| $S_0$ | 2.0 | reference supersaturation in pure water |

### 5.4 General

| Parameter | Value |
|-----------|------:|
| $T$ | 298.15 K |
| $P_{\rm atm}$ | 1.01325 × 10$^{5}$ Pa |
| $\rho_l$ | 997.05 kg m$^{-3}$ (water, 25 °C) |
| $\rho_g$ | 0.0899 kg m$^{-3}$ (H$_2$, 25 °C, 1 bar) |
| $p_{\rm sat}(T)$ | 3169 Pa (water, 25 °C) |


## 6. Salt vs. ions vs. electric field — what the toy captures

A common confusion for experimentalists: "you're varying *salt* molality,
not *ion* concentration or *electric field* — aren't those the bigger
effects?".  For a single, completely-dissociated strong electrolyte these
three quantities are **not** independent:

1. **Ion concentration** in the toy is set by salt molality via
   stoichiometry:
   $c_+ = \nu_+ m,\ c_- = \nu_- m$.  KOH at 3 mol kg$^{-1}$ means 3 mol kg$^{-1}$
   K$^+$ *and* 3 mol kg$^{-1}$ OH$^-$.  We ARE varying ion concentration when
   we vary $m$.

2. **Ionic strength** is related to $m$ through charge structure:
   $I_m = \tfrac12\sum_i \nu_i z_i^{\,2}\,m$.
   For KOH, $I_m = m$; for H$_2$SO$_4$, $I_m = 3m$.  The PDH term in §1.2
   captures this exactly — that is why KOH and H$_2$SO$_4$ diverge in
   Fig. 1 even at "the same molality".

3. **Ion-identity effects beyond ionic strength** come entirely from the
   e-NRTL short-range $\tau$ parameters.  Fig. 6 (right panel) demonstrates
   that plotting $\sigma$ vs. $I_m$ does **not** collapse the two salts
   onto one curve — specific-ion behaviour is a real, separate effect
   captured by e-NRTL.

4. **Electric-field / double-layer effects are NOT in this toy.**
   They act at the *electrode-electrolyte* interface (polarised Pt),
   through:
   - the Gouy-Chapman-Stern diffuse-layer structure;
   - the Lippmann equation, $\partial\sigma/\partial E = -q_s$, linking
     electrode surface charge $q_s$ to electrocapillary depression;
   - the Frumkin correction to activation overpotential.

   The bubble-electrolyte interface described here is **not** polarised
   in the same way: it carries a weak ζ-potential (typically −30 to −50 mV
   for H$_2$ bubbles in water) from preferential ion adsorption, which is
   already implicit in the Butler surface-phase composition.  A full
   treatment of electric-field effects on σ would add a Lippmann term:
   $\sigma(E) = \sigma_0 - \tfrac12\,C_{\rm dl}\,(E-E_{\rm pzc})^2$
   with double-layer capacitance $C_{\rm dl}$ and potential-of-zero-charge
   $E_{\rm pzc}$.  This is a genuinely separate toy layer.


## 7. Known limitations and extensions

| # | Simplification in the toy | Path to improvement |
|---|---------------------------|--------------------|
| 1 | "Lumped-salt" NRTL (not full Chen-symmetric with separate cation/anion) | Replace with full Chen 1986 / Aspen symmetric e-NRTL (two tau pairs per ion-solvent pair) |
| 2 | Bulk e-NRTL used for surface activities in Butler | Couple a surface-phase e-NRTL as in Li & Lu 2001 |
| 3 | $\tau$ parameters chosen by trend, not fitted to data | Use published Aspen-databank parameters (Thomsen & Rasmussen 1999 for KOH-H$_2$O; Clegg & Brimblecombe 1995 for H$_2$SO$_4$-H$_2$O) |
| 4 | Fritz-style *vertical-force* detachment | Pinch-off from the actual YL shape (watch $r_{\rm neck}\to 0$) |
| 5 | Quasi-static bubble (no inertia, no viscous drag, no Marangoni) | Couple $\sigma(m)$ into dynamic integrators in `ddgclib/dynamic_integrators/` |
| 6 | No double-layer / electrocapillarity | Add Lippmann correction to $\sigma$ at polarised surface |
| 7 | Sechenov form for salting-out | Derive salting-out directly from e-NRTL activity coefficient of dissolved H$_2$ (needs $\tau$ parameters for H$_2$-water and H$_2$-ion) |
| 8 | Ideal-gas bubble (no H$_2$ fugacity correction) | Use Peng-Robinson or similar EoS for $\phi_{\rm H_2}(T,P)$ |
| 9 | Single solvent (water) | Extend to mixed solvents (water + methanol etc. for CO$_2$ electrolysis) |
|10 | Constant $T$, $P$ | Add $T$, $P$ dependence of all parameters (heavy but mechanical) |


## 8. References

### Primary literature that this toy is a commentary on

- Sepahi, F., Pande, N., Chong, K. L., Mul, G., Verzicco, R., Lohse, D.,
  Mei, B. T., Krug, D.
  **"The effect of buoyancy driven convection on the growth and
  dissolution of bubbles on electrodes."**
  *Electrochim. Acta* **403**, 139616 (2022).
  doi:10.1016/j.electacta.2021.139616
  (the "Krug paper" — uses linear ${\beta}$, ideal gas, constant
  $k_H$, no composition-dependent $\sigma$).

- Raman, A., Peñas, P., van der Meer, D., Lohse, D., Gardeniers, H.,
  Fernández Rivas, D.
  **"Potential response of single successive constant-current-driven
  electrolytic hydrogen bubbles spatially separated from the electrode."**
  *Electrochim. Acta* **425**, 140691 (2022).
  doi:10.1016/j.electacta.2022.140691
  (the "Rivas paper" — fixed $\sigma = 72$ mN m$^{-1}$; concentration
  overpotential inferred by subtraction).

### Thermodynamic model foundations

- Butler, J. A. V.
  *Proc. R. Soc. A* **135**, 348 (1932). — Butler equation for surface
  tension.

- Chen, C.-C., Evans, L. B.
  "A local composition model for the excess Gibbs energy of aqueous
  electrolyte systems."
  *AIChE J.* **32**, 444 (1986). — symmetric e-NRTL.

- Song, Y., Chen, C.-C.
  "Symmetric electrolyte nonrandom two-liquid activity coefficient
  model."
  *Ind. Eng. Chem. Res.* **48**, 7788 (2009). — modern symmetric form.

- Li, Z.-B., Lu, B. C.-Y.
  "Surface tension of aqueous electrolyte solutions at high
  concentrations — representation and prediction."
  *Langmuir* **17**, 3532 (2001). — Butler + e-NRTL coupling.

- Hu, Y.-F., Lee, H.
  "Thermodynamic models for the surface tension of electrolyte
  solutions."
  *Fluid Phase Equil.* **158-160**, 1043 (1999). — surface-phase
  formulation.

- Thomsen, K., Rasmussen, P.
  "Modeling of vapor-liquid-solid equilibrium in gas-aqueous
  electrolyte systems."
  *Chem. Eng. Sci.* **54**, 1787 (1999). — published KOH-H$_2$O
  Aspen-style parameters.

- Fritz, W.
  *Phys. Z.* **36**, 379 (1935). — detachment force balance.

- Sechenov, I. M.
  *Z. Phys. Chem.* **4**, 117 (1889). — salting-out linear correlation.

- Sander, R.
  "Compilation of Henry's law constants (version 4.0) for water as
  solvent."
  *Atmos. Chem. Phys.* **15**, 4399 (2015). — tabulated $k_H$ for H$_2$.
