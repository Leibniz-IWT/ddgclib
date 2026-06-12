from pathlib import Path
import os
import tempfile

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter


NEATPLOT_STYLE = SCRIPT_DIR / "neatplot-main" / "standard.mplstyle"
OUTPUT_PATH = SCRIPT_DIR / "chart.png"
TIME_OUTPUT_PATH = SCRIPT_DIR / "chart_vs_t.png"

# Pitois et al. (2000), Fig. 5: ruby spheres with R = 4 mm separating at
# constant gap rate v = dD/dt = 5 um/s.
PITOIS_RADIUS_MM = 4.0
PITOIS_FIG5_GAP_RATE_MM_S = 5.0e-3


EXP_D_OVER_R = np.array(
    [
        0.014403481854782,
        0.019300000000000,
        0.023587192093071,
        0.028302414461984,
        0.033166898401356,
        0.037986935617921,
        0.042569921040376,
        0.047499765555437,
        0.051967121934836,
        0.056470352302334,
        0.061304411920954,
        0.066003127630689,
        0.070366380348669,
        0.075100720316807,
        0.079819428415061,
        0.085182169652359,
        0.092208347873809,
        0.097048578239299,
        0.101370508634981,
        0.108834059521728,
        0.118227619608521,
        0.129680242729548,
        0.146372328112875,
        0.160984243023503,
        0.184443915271577,
        0.186311945235594,
        0.202973726366585,
    ],
    dtype=float,
)

EXP_FORCE_MN = np.array(
    [
        1.515311539463195,
        1.130000000000000,
        0.955216555657008,
        0.822460624401661,
        0.743810614199630,
        0.655674232696210,
        0.574562744753398,
        0.517342047749411,
        0.458265451472290,
        0.416512293755708,
        0.376315664380932,
        0.337105214458683,
        0.306748087602010,
        0.286922242059943,
        0.267636737727435,
        0.238270688094022,
        0.218305191654015,
        0.198684060873300,
        0.178384121904434,
        0.158325437116782,
        0.139272315154625,
        0.119413121762400,
        0.099728317139538,
        0.079873442448703,
        0.059741521119530,
        0.052749970637026,
        0.042572497546823,
    ],
    dtype=float,
)

SIM_D_OVER_R = np.array(
    [
        0.014391354238989,
        0.015002734539752,
        0.015640087786907,
        0.016304517375417,
        0.016997173575190,
        0.017719255522445,
        0.018472013295667,
        0.019256750079771,
        0.020074824422183,
        0.020927652584785,
        0.021816710995764,
        0.022743538805623,
        0.023709740551788,
        0.024716988936397,
        0.025700112659239,
        0.026791916970794,
        0.027930103828236,
        0.029116643676764,
        0.030353590670958,
        0.031643086230963,
        0.032987362749745,
        0.034388747457840,
        0.035849666452292,
        0.037372648896743,
        0.038960331399956,
        0.040615462580355,
        0.042340907824466,
        0.044139654247521,
        0.046014815864785,
        0.047969638982594,
        0.050007507818398,
        0.052131950359571,
        0.054346644471111,
        0.056655424262812,
        0.059062286726930,
        0.061571398657828,
        0.064187103865599,
        0.066913930696122,
        0.069756599870603,
        0.072720032658153,
        0.075809359395560,
        0.078824693497421,
        0.082173361316841,
        0.085664288822500,
        0.089303519557508,
        0.093097353809625,
        0.097052359518410,
        0.101175383645725,
        0.105473564029299,
        0.109954341739853,
        0.114625473963181,
        0.119495047429506,
        0.124571492413341,
        0.129863597328095,
        0.135380523940704,
    ],
    dtype=float,
)

SIM_FORCE_MN = np.array(
    [
        1.551528385152140,
        1.536156281485830,
        1.520936480267470,
        1.490947764247040,
        1.447069717127450,
        1.400991179659750,
        1.342941277212700,
        1.280903705899960,
        1.218694526308020,
        1.159506637081370,
        1.097714636916710,
        1.039215633240590,
        0.983834136898348,
        0.931404010838657,
        0.886168854506430,
        0.843130617389270,
        0.802182602518957,
        0.763223294839722,
        0.729780337252968,
        0.696067917742758,
        0.667226426584123,
        0.637989864707824,
        0.613079059011306,
        0.589140914911431,
        0.566137454086941,
        0.545388117132852,
        0.525399258011356,
        0.504884639279270,
        0.486380261185711,
        0.467389169240123,
        0.450259026566740,
        0.431602597498823,
        0.413719195344371,
        0.395590826364709,
        0.376378312068051,
        0.354550942597078,
        0.330680338897515,
        0.313057844206980,
        0.303089252208188,
        0.291254898893000,
        0.281279514492154,
        0.272322828788611,
        0.262342005061026,
        0.252726985561915,
        0.241652981510519,
        0.222042138334079,
        0.209165212017808,
        0.200998195199393,
        0.192190845944574,
        0.176154953671598,
        0.166767368833204,
        0.157487544580974,
        0.148354343457462,
        0.139403359146578,
        0.132303257170707,
    ],
    dtype=float,
)


def elapsed_time_from_d_over_r(d_over_r: np.ndarray) -> np.ndarray:
    d_over_r = np.asarray(d_over_r, dtype=float)
    return PITOIS_RADIUS_MM * (d_over_r - float(d_over_r[0])) / PITOIS_FIG5_GAP_RATE_MM_S


def _format_log_axes(ax) -> None:
    decimal_log_formatter = FuncFormatter(
        lambda value, _pos: f"{float(value):g}" if value > 0.0 else ""
    )
    ax.xaxis.set_major_formatter(decimal_log_formatter)
    ax.yaxis.set_major_formatter(decimal_log_formatter)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def main() -> None:
    plt.style.use(str(NEATPLOT_STYLE))

    fig, ax = plt.subplots()
    ax.scatter(
        EXP_D_OVER_R,
        EXP_FORCE_MN,
        s=16,
        color="#111111",
        label="Pitois et al. (2000)",
        zorder=3,
    )
    ax.plot(
        SIM_D_OVER_R,
        SIM_FORCE_MN,
        color="#d62828",
        linewidth=1.4,
        label="Axisymmetric simulation",
        zorder=5,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 0.30)
    ax.set_ylim(0.02, 2.0)
    ax.set_xlabel(r"$D/R$")
    ax.set_ylabel(r"$|F|$ [mN]")
    ax.set_title("Pitois 2000 Fig. 5: dynamic experiment vs axisym")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend()

    _format_log_axes(ax)

    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT_PATH)

    exp_t_s = elapsed_time_from_d_over_r(EXP_D_OVER_R)
    sim_t_s = elapsed_time_from_d_over_r(SIM_D_OVER_R)

    fig, ax = plt.subplots()
    ax.scatter(
        exp_t_s,
        EXP_FORCE_MN,
        s=16,
        color="#111111",
        label="Pitois et al. (2000)",
        zorder=3,
    )
    ax.plot(
        sim_t_s,
        SIM_FORCE_MN,
        color="#d62828",
        linewidth=1.4,
        label="Axisymmetric simulation",
        zorder=5,
    )

    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.7)
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$|F|$ [mN]")
    ax.set_title("Pitois 2000 Fig. 5: reconstructed time comparison, linear scale")
    ax.grid(True, which="major", alpha=0.28)
    ax.legend()

    fig.savefig(TIME_OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(TIME_OUTPUT_PATH)


if __name__ == "__main__":
    main()
