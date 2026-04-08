import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import chaospy as cp
from scipy.interpolate import griddata
import poiseuille_solver as poiseuille
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os

# --- Configuration & Nominal Values ---
plt.style.use('seaborn-v0_8-darkgrid')
FRAME_DIR = "pce_analysis_frames"
os.makedirs(FRAME_DIR, exist_ok=True)

H = poiseuille.H
N_REF = 200
y_ref = np.linspace(0, H, N_REF)

# Defining Nominal Values for Analytical reference
MU_NOM = 1.0
DPDX_NOM = -8.0
RHO_NOM = 1.0

# Defining Intervals (+/- 10% for Mu/P, +/- 5% for Rho)
MU_INT = [MU_NOM * 0.9, MU_NOM * 1.1]
DP_INT = [DPDX_NOM * 1.1, DPDX_NOM * 0.9]  # -8.8 to -7.2
RHO_INT = [RHO_NOM * 0.95, RHO_NOM * 1.05]

T_END, DT = 1.0, 0.005
SAVE_EVERY = 10
NUM_CORES = cpu_count()


def run_cfd_instance(params):
    """Parallelized solver worker."""
    mu_val, dpdx_val, rho_val = params
    poiseuille.mu, poiseuille.dPdx, poiseuille.rho = mu_val, dpdx_val, rho_val
    poiseuille.nu = mu_val / rho_val

    unit_mesh = poiseuille.build_unit_mesh()
    HC, bV = poiseuille.build_tiled_mesh(unit_mesh, n_tiles=5)
    poiseuille._HC_ref = HC

    history = []
    t, step = 0.0, 0
    while t < T_END:
        if step % SAVE_EVERY == 0:
            y_p = np.array([float(v.x_a[1]) for v in HC.V])
            u_p = np.array([float(v.u[0]) for v in HC.V])
            y_rounded = np.round(y_p, 8)
            unq_y, idx, counts = np.unique(y_rounded, return_inverse=True, return_counts=True)
            unq_u = np.zeros_like(unq_y);
            np.add.at(unq_u, idx, u_p);
            unq_u /= counts
            u_i = griddata(unq_y, unq_u, y_ref, method='linear', fill_value=0)
            history.append(np.nan_to_num(u_i))

        for v in HC.V:
            if not poiseuille.wall_criterion(v):
                accel = poiseuille.dudt_fn(v)
                v.u[0] += accel[0] * DT
                v.x_a[0] += v.u[0] * DT
                poiseuille._safe_move(v, v.x_a, HC, bV)
        t += DT;
        step += 1
    return np.array(history)


def create_animation(y, means, vars, sobols, samples, nominal_ana, errors, n_sims):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    t_history = np.arange(0, len(means)) * SAVE_EVERY * DT

    # Hardware & Input Info Text
    info_text = (f"SIMULATION METADATA:\n"
                 f"Parallel Cores: {NUM_CORES}\n"
                 f"Total CFD Runs: {n_sims}\n"
                 f"$\mu \in {MU_INT}$ Pa·s\n"
                 f"$dP/dx \in {DP_INT}$ Pa/m\n"
                 f"$\\rho \in {RHO_INT}$ kg/m³")

    def update(frame):
        for ax in [ax1, ax2, ax3]: ax.clear()
        t_curr = t_history[frame]
        mean, std = means[frame], np.sqrt(vars[frame])

        # 1. Profile Panel
        ax1.plot(nominal_ana[frame], y, 'k--', alpha=0.5, label='Analytical (Nominal)')
        ax1.plot(mean, y, color='royalblue', lw=3, label='PCE Mean')
        ax1.fill_betweenx(y, mean - 2 * std, mean + 2 * std, color='royalblue', alpha=0.2, label='95% CI')
        ax1.set_title(f"Velocity Uncertainty (t = {t_curr:.2f}s)", fontsize=14, fontweight='bold')
        ax1.set_xlim(-0.05, 1.3);
        ax1.legend(loc='lower right')
        ax1.text(0.05, 0.95, info_text, transform=ax1.transAxes, verticalalignment='top',
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # 2. Sobol Panel (3 variables)
        labels = [r'Sens. $\mu$', r'Sens. $dP/dx$', r'Sens. $\rho$']
        colors = ['tab:red', 'tab:green', 'tab:purple']
        for i in range(len(sobols[frame])):
            ax2.plot(sobols[frame][i], y, color=colors[i], lw=2.5, label=labels[i])
        ax2.set_title("Global Sensitivity Analysis (Sobol Indices)", fontsize=14)
        ax2.set_xlim(0, 1.1);
        ax2.legend(loc='center right')

        # 3. PDF Panel + Centerline Mean
        curr_samples = samples[frame]
        mean_val = np.mean(curr_samples)
        ax3.hist(curr_samples, bins=50, density=True, color='skyblue', edgecolor='white', alpha=0.7)
        ax3.axvline(mean_val, color='red', linestyle='--', lw=2.5, label=f'Centerline Mean: {mean_val:.3f}')
        ax3.set_title("Centerline Velocity Probability Density", fontsize=14)
        ax3.set_xlabel("Velocity [m/s]");
        ax3.legend()

        # 4. L2 Error Panel
        ax4.plot(t_history[:frame + 1], errors[:frame + 1], 'r-', lw=2)
        ax4.set_yscale('log');
        ax4.set_title("L2 Convergence Rigor", fontsize=14)
        ax4.set_ylabel("L2 Error Norm");
        ax4.set_xlabel("Time [s]")

        plt.tight_layout()
        plt.savefig(f"{FRAME_DIR}/frame_{frame:03d}.png", bbox_inches='tight')

    ani = FuncAnimation(fig, update, frames=len(means), interval=100)
    ani.save("pce_rigorous_dynamic.gif", writer='pillow')
    print(f" Animation saved. Check the '{FRAME_DIR}' folder for HQ frames.")


if __name__ == "__main__":
    # Joint distribution for 3 parameters
    dist = cp.J(cp.Uniform(*MU_INT), cp.Uniform(*DP_INT), cp.Uniform(*RHO_INT))

    # Using Order 5 quadrature nodes for 3 parameters = 6^3 = 216 sims
    nodes, weights = cp.generate_quadrature(6, dist, rule="gaussian")

    print(f" Launching {nodes.shape[1]} parallel simulations on {NUM_CORES} cores...")
    with Pool(NUM_CORES) as pool:
        raw_results = list(tqdm(pool.imap(run_cfd_instance, nodes.T), total=nodes.shape[1]))

    results = np.array(raw_results)
    means, vars, sobols, samples_h, nominal_ana, errors = [], [], [], [], [], []

    print("⏳ Processing PCE for each time step...")
    for f in tqdm(range(results.shape[1]), desc="PCE Calculation"):
        t_f = f * SAVE_EVERY * DT
        # Order 4 PCE expansion
        model = cp.fit_quadrature(cp.generate_expansion(4, dist), nodes, weights, results[:, f, :])

        m_vals = cp.E(model, dist)
        u_ana = (1.0 / (2.0 * MU_NOM)) * (-DPDX_NOM) * (y_ref * (H - y_ref))  # Steady state reference

        means.append(m_vals)
        vars.append(cp.Var(model, dist))
        sobols.append(cp.Sens_m(model, dist))
        samples_h.append(model[N_REF // 2](*dist.sample(5000)))
        nominal_ana.append(u_ana)
        errors.append(np.sqrt(np.mean((m_vals - u_ana) ** 2)))

    create_animation(y_ref, means, vars, sobols, samples_h, nominal_ana, errors, nodes.shape[1])