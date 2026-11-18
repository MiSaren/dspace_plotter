"""Reusable measurement plots extracted from Lab_plotter.ipynb.

load_measurement returns an independent, in-memory SimpleNamespace after closing
its recording. Built-in plots retain the notebook units, labels and limits, and
use the caller's Matplotlib style. No backend, display or export is configured.
FIGURE_TYPES maps GUI names to (function, keyword options, export suffix).
For custom panels, open LabDataMat in a with block, select from data.keys(), and
pass lists of readable keys to plot_signals; each component keeps its own raster.
"""
import ast
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from lab_plotting import LabDataMat, transient_window
from plot_style import downsample

@dataclass
class BaseValues:
    # pylint: disable=too-many-instance-attributes
    U_nom: float
    I_nom: float
    f_nom: float
    P_nom: float
    tau_nom: float
    n_p: int

    def __post_init__(self):
        self.u = np.sqrt(2/3)*self.U_nom
        self.i = np.sqrt(2)*self.I_nom
        self.w = 2*np.pi*self.f_nom
        self.w_M = self.w/self.n_p
        self.psi = self.u/self.w
        self.p = 1.5*self.u*self.i
        self.Z = self.u/self.i
        self.L = self.Z/self.w
        self.C = 1/(self.Z*self.w)
        self.tau = self.n_p*self.p/self.w


DEFAULT_BASE = BaseValues(U_nom=460, I_nom=8.8, f_nom=60,
                          tau_nom=29.2, P_nom=5.6e3, n_p=2)


def load_measurement(path, motor_base=None, *, x_step=None, extra_signals=None, **window_options):
    """Load preset signals and detach requested extra components on their own rasters."""
    motor_base = DEFAULT_BASE if motor_base is None else motor_base
    _validate_window(0, None, x_step)
    with LabDataMat(path) as data:
        origin, limits = transient_window(data, **window_options)
        context = SimpleNamespace(
            file_path=data.file_path.resolve(), name=data.file_path.stem,
            base=replace(motor_base), t=data.t.copy(), t_offset=origin,
            t_span=limits, x_step=x_step, loaded=[], signal_rows=[], extra_signals={})

        if extra_signals is not None:
            if isinstance(extra_signals, (str, bytes)):
                raise ValueError('extra_signals must be an iterable of signal keys')
            for key in extra_signals:
                if not isinstance(key, str) or not key:
                    raise ValueError('Extra signal keys must be nonempty strings')
                try:
                    signals = data.signal_info(key)
                    if not signals:
                        raise ValueError('no components found')
                    components = []
                    for signal in signals:
                        true_key = signal['key']
                        time = np.asarray(data.time(true_key), dtype=float)
                        values = np.asarray(data[true_key])
                        if (time.ndim != 1 or not time.size or not np.all(np.isfinite(time))
                                or np.any(np.diff(time) <= 0)):
                            raise ValueError(f"component '{true_key}' needs a finite increasing time vector")
                        if (values.ndim != 1 or values.size != time.size
                                or not np.issubdtype(values.dtype, np.number)
                                or np.iscomplexobj(values)):
                            raise ValueError(f"component '{true_key}' needs real values matching its time")
                        components.append((true_key, time.copy(), values.copy()))
                    context.extra_signals[key] = components
                except (KeyError, TypeError, ValueError, IndexError) as error:
                    raise ValueError(f"{context.name}: cannot load extra signal '{key}': {error}") from error

        def get(key, as_complex=False):
            context.loaded.append(key)
            try:
                y = data.complex(key) if as_complex else data[key]
            except (KeyError, ValueError) as error:
                print(f"{context.name}: Warning: {error.args[0]}")
                return np.full(context.t.size, np.nan,
                               dtype=complex if as_complex else float)
            t_y = data.time(key)
            if not np.array_equal(t_y, context.t):
                idx = np.clip(np.searchsorted(t_y, context.t, side="right") - 1,
                              0, t_y.size - 1)
                y = y[..., idx]
            return y.copy()

        context.theta_m = get("theta_m_out")
        context.w_m_meas = get("wm_meas")
        context.i_cs = get("i_c_abc", as_complex=True)
        context.u_dc = get("u_dc_lim")
        context.w_m_ref = get("wm_ref")
        context.tau_Lref = get("tauLref" if "tauLref" in data or "TL_ref" not in data
                               else "TL_ref")
        context.tau_Mref_lim = get("tauM_ref_lim")
        context.psi_sref = get("psis_ref")
        context.i_ref = get("cur_is_ref", as_complex=True)
        context.theta_mest = get("theta_m_out")
        context.w_mest = get("w_m")
        context.tau_Mest = get("tau_m")
        context.psi_sest = get("psi_s", as_complex=True)
        context.i_s = get("i_s", as_complex=True)
        context.u_inj = get("u_sd_inj", as_complex=True)
        # Model-version-dependent observer mon elements: check the signal table.
        context.eps_inj = get("mon[3]")
        context.w_inj = get("mon[4]")
        context.u_c = get("ucs_realized", as_complex=True)
        context.w_m = context.w_mest
        context.i_c = context.i_cs*np.exp(-1j*context.theta_m)
        context.u_s = np.abs(context.u_c)
        context.theta_err_M = np.degrees(np.angle(np.exp(
            1j*(context.theta_m - context.theta_mest))))/context.base.n_p
        context.eps_inj_M = np.degrees(context.eps_inj)

        for key in context.loaded:
            if key not in data:
                context.signal_rows.append((key, "-", "NOT IN RECORDING", "", "", ""))
                continue
            for signal in data.signal_info(key):
                y = data[signal["key"]]
                route = signal["path"].removeprefix("Model Root/")
                if not np.array_equal(data.time(signal["key"]), context.t):
                    route += (f"  [{data.raster_names[signal['raster']]}, "
                              f"{signal['samples']} samples]")
                context.signal_rows.append((
                    signal["key"], signal["name"], route,
                    f"{np.min(y):.4g}", f"{np.max(y):.4g}", f"{np.mean(y):.4g}"))
    return context


def plot_window(context, t_offset=None, t_span=None, x_step=None):
    origin = context.t_offset if t_offset is None else t_offset
    limits = context.t_span if t_span is None else t_span
    step = context.x_step if x_step is None else x_step
    origin, limits, step = _validate_window(origin, limits, step)
    return context.t - origin, limits, step


def plot_overview(context, t_offset=None, t_span=None, bottom="voltage", x_step=None):
    # Speed, flux, torque, currents and voltages (bottom="voltage")
    # or injection voltages (bottom="injection").
    t, t_span, x_step = plot_window(context, t_offset, t_span, x_step)
    base = context.base
    w_m_ref = context.w_m_ref
    w_mest = context.w_mest
    psi_sref = context.psi_sref
    psi_sest = context.psi_sest
    tau_Mref_lim = context.tau_Mref_lim
    tau_Mest = context.tau_Mest
    tau_Lref = context.tau_Lref
    i_c = context.i_c
    u_dc = context.u_dc
    u_s = context.u_s
    u_inj = context.u_inj
    if bottom not in ("voltage", "injection"):
        raise ValueError("bottom must be voltage or injection")

    width = plt.rcParams["figure.figsize"][0]
    height = plt.rcParams["figure.figsize"][1]*3*4/5
    fig, axes = plt.subplots(5, 1, figsize=(width, height), sharex=True)
    for ax in axes:
        ax.xaxis.set_major_locator(MultipleLocator(1 if x_step is None else x_step))
        ax.grid(True, which="major", linestyle="-", linewidth=0.8, color="#B0B0B0")

    # Speed
    axes[0].step(t, w_m_ref/base.w, "--", where="post",
                 label=r"$\omega_\mathrm{m}^\mathrm{{ref}}$", linewidth=1.3)
    axes[0].plot(t, w_mest/base.w, label=r"$\omega_\mathrm{m}$", alpha=0.8)
    axes[0].set_ylabel(r"$\omega_\mathrm{m}$ (p.u.)")

    # Flux linkage
    axes[1].plot(t, np.abs(psi_sref)/base.psi, "--",
                 label=r"$\psi_\mathrm{s}^\mathrm{ref}$", linewidth=1.3)
    axes[1].step(t, np.abs(psi_sest)/base.psi, where="post",
                 label=r"$\hat\psi_\mathrm{s}$", alpha=0.8)
    axes[1].set_ylabel(r"$\psi_\mathrm{s}$ (p.u.)")
    axes[1].set_ylim([-0.05, 1.2])

    # Torque
    axes[2].plot(t, tau_Mref_lim/base.tau, "--",
                 label=r"$\overline{\tau}_\mathrm{m}^\mathrm{ref}$", linewidth=1.3)
    axes[2].plot(t, tau_Mest/base.tau, label=r"$\hat \tau_\mathrm{m}$", alpha=0.8)
    axes[2].plot(t, tau_Lref, "--", label=r"$\tau_\mathrm{L}$", linewidth=1.3)
    axes[2].set_ylabel(r"$\tau_\mathrm{m}$ (p.u.)")
    axes[2].set_ylim([-1.1, 2])

    # Currents
    axes[3].step(t, i_c.real/base.i, where="post", label=r"$i_\mathrm{d}$")
    axes[3].step(t, i_c.imag/base.i, where="post", label=r"$i_\mathrm{q}$")
    # axes[3].step(t, i_ref.real/base.i, where="post", label=r"$i_\mathrm{d}^\mathrm{ref}$")
    # axes[3].step(t, i_ref.imag/base.i, where="post", label=r"$i_\mathrm{q}^\mathrm{ref}$")
    axes[3].set_ylabel(r"$i_\mathrm{d}, i_\mathrm{q}$ (p.u.)")

    # Voltages
    if bottom == "voltage":
        axes[4].plot(t, u_dc/np.sqrt(3)/base.u, "--",
                     label=r"$u_\mathrm{dc}/\sqrt{3}$", ds="steps-post")
        axes[4].plot(t, u_s/base.u, label=r"$u_\mathrm{s}$", ds="steps-post")
        axes[4].set_ylabel("Voltage (p.u.)")
    else:
        axes[4].plot(t, u_inj.real, label=r"$u_\mathrm{d}^{\mathrm{inj}}$",
                     ds="steps-post")
        axes[4].plot(t, u_inj.imag, label=r"$u_\mathrm{q}^{\mathrm{inj}}$",
                     ds="steps-post")
        axes[4].set_ylabel("Injection Voltages (V)")

    for ax in axes:
        ax.legend(loc="best")
    axes[-1].set_xlim(t_span)
    axes[-1].set_xlabel("Time (s)")
    fig.align_ylabels()
    return fig, axes


def plot_hfsi(context, t_offset=None, t_span=None, x_step=None):
    # Left: speed, flux, torque. Right: currents, angle errors, injection and weight.
    # x_step: spacing of the time ticks (s), None = automatic
    t, t_span, x_step = plot_window(context, t_offset, t_span, x_step)
    base = context.base
    w_m_ref = context.w_m_ref
    w_mest = context.w_mest
    w_m_meas = context.w_m_meas
    psi_sref = context.psi_sref
    psi_sest = context.psi_sest
    tau_Mref_lim = context.tau_Mref_lim
    tau_Mest = context.tau_Mest
    tau_Lref = context.tau_Lref
    i_c = context.i_c
    eps_inj_M = context.eps_inj_M
    theta_err_M = context.theta_err_M
    theta_mest = context.theta_mest
    theta_m = context.theta_m
    width = 2*plt.rcParams["figure.figsize"][0]
    height = plt.rcParams["figure.figsize"][1]*3*4/5*3/5
    fig, axes = plt.subplots(3, 2, figsize=(width, height), sharex=True,
                             layout="constrained")
    for ax in axes.flat:
        if x_step is not None:
            ax.xaxis.set_major_locator(MultipleLocator(x_step))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))  # minor lines between majors
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(True, which="major", linestyle="-", linewidth=0.8, color="#B0B0B0")
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, color="#C8C8C8")
    (ax_w, ax_i), (ax_psi, ax_err), (ax_tau, ax_u) = axes

    # Speed
    ax_w.step(t, w_m_ref/base.w, "--", where="post",
              label=r"$\omega_\mathrm{m}^\mathrm{{ref}}$", linewidth=1.3)
    ax_w.plot(t, w_mest/base.w, label=r"$\omega_\mathrm{m}$", alpha=0.8)
    ax_w.plot(t, (w_m_meas/2)/base.w, label=r"$\omega_\mathrm{m}^\mathrm{meas}$", alpha=0.8)
    ax_w.set_ylabel(r"$\omega_\mathrm{m}$ (p.u.)")
    ax_w.set_ylim([-0.2, 0.2])
    # ax_w.set_ylim([-0.15, 1.15])


    # Flux linkage
    ax_psi.plot(t, np.abs(psi_sref)/base.psi, "--",
                label=r"$\psi_\mathrm{s}^\mathrm{ref}$", linewidth=1.3)
    ax_psi.step(t, np.abs(psi_sest)/base.psi, where="post",
                label=r"$\hat\psi_\mathrm{s}$", alpha=0.8)
    ax_psi.set_ylabel(r"$\psi_\mathrm{s}$ (p.u.)")
    ax_psi.set_ylim([-0.05, 1.2])

    # Torque
    ax_tau.plot(t, tau_Mref_lim/base.tau_nom, "--",
                label=r"$\overline{\tau}_\mathrm{m}^\mathrm{ref}$", linewidth=1.3)
    ax_tau.plot(t, tau_Mest/base.tau_nom, label=r"$\hat \tau_\mathrm{m}$", alpha=0.8)
    ax_tau.plot(t, tau_Lref/.88, "--", label=r"$\tau_\mathrm{L}$", linewidth=1.3)
    ax_tau.set_ylabel(r"$\tau_\mathrm{m}$ (per rated)")
    ax_tau.set_ylim([-3, 3])

    # Currents
    ax_i.step(t, i_c.real/base.i, where="post", label=r"$i_\mathrm{d}$")
    ax_i.step(t, i_c.imag/base.i, where="post", label=r"$i_\mathrm{q}$")
    ax_i.set_ylabel(r"$i_\mathrm{d}, i_\mathrm{q}$ (p.u.)")
    # ax_i.set_ylim([-1.5, 1])
    ax_i.set_ylim([-2, 2])

    # Mechanical angle error, injection error signal behind it
    # (not grey: it blends into the grid lines)
    ax_err.plot(t, eps_inj_M, color="C1", linewidth=1, alpha=0.7, zorder=1.5,
                label=r"$\varepsilon_\mathrm{i}$")
    ax_err.plot(t, theta_err_M, color="C0", zorder=2.5,
                label=r"$\tilde\vartheta$")
    ax_err.set_ylabel("Error (mech. deg)")
    ax_err.set_ylim([-10, 10])  # eps_inj spikes when the injection starts or stops

    # # Injection voltage, d-axis
    # ax_u.plot(t, u_inj.real, color="C0", label=r"$\mathbf{u}_\mathrm{i}$",
    #           ds="steps-post")
    # ax_u.set_ylabel(r"$\mathbf{u}_\mathrm{i}$ (V)")
    # ax_u.tick_params(axis="y")
    # angles
    wrap = lambda x: np.angle(np.exp(1j*x))  # wrap to (-pi, pi]
    ax_u.plot(t, wrap(theta_mest), color="C0", label=r"$\hat\vartheta_\mathrm{m}$")
    ax_u.plot(t, wrap(theta_m), color="C1", label=r"$\vartheta_\mathrm{m}$")
    ax_u.set_ylabel(r"$\vartheta_\mathrm{m}$ (rad)")
    ax_u.set_ylim([-np.pi, np.pi])
    ax_u.set_yticks([-np.pi, 0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
    ax_u.tick_params(axis="y")

    for ax in (ax_w, ax_psi, ax_tau, ax_i, ax_err):
        ax.legend(loc="upper center", bbox_to_anchor=(0.7, 1.0))
    for ax in (ax_tau, ax_w, ax_psi):
        ax.legend(loc="lower center", bbox_to_anchor=(0.34, 0))  # a bit left of center
    ax_err.legend(loc="upper right")
    ax_u.legend(loc="lower right")
    lines = ax_u.get_lines() #+ ax_weight.get_lines()
    # ax_weight.legend(lines, [line.get_label() for line in lines], loc="lower right")
    for ax in axes[-1]:
        ax.set_xlim(t_span)
        ax.set_xlabel("Time (s)")
    fig.align_ylabels(axes)
    return fig, axes


def plot_presentation(context, t_offset=None, t_span=None, x_step=None,
                      include_plots=("speeds", "fluxes", "torques", "currents"),
                      font_size=16):
    include_plots = tuple(include_plots)
    allowed = {"speeds", "fluxes", "torques", "currents", "voltages"}
    if not include_plots or set(include_plots) - allowed:
        raise ValueError("include_plots must be a nonempty selection of presentation panels")
    t, t_span, x_step = plot_window(context, t_offset, t_span, x_step)
    base = context.base
    w_m_ref = context.w_m_ref
    psi_sref = context.psi_sref
    psi_sest = context.psi_sest
    tau_Mref_lim = context.tau_Mref_lim
    tau_Mest = context.tau_Mest
    tau_Lref = context.tau_Lref
    i_c = context.i_c
    u_dc = context.u_dc
    u_s = context.u_s
    w_m = context.w_m
    M = r"\mathrm{m}"
    with plt.rc_context({key: font_size for key in (
            "font.size", "axes.labelsize", "legend.fontsize",
            "xtick.labelsize", "ytick.labelsize")}):
        n_plots = len(include_plots)
        fig, axes = plt.subplots(n_plots, 1, figsize=(8, 2*n_plots), sharex=True,
                                 squeeze=False, layout="tight")
        axes = axes[:, 0]

        for ax in axes:
            if x_step is not None:
                ax.xaxis.set_major_locator(MultipleLocator(x_step))
            ax.minorticks_on()
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.grid(True, which='major', linestyle='-', linewidth=0.8)
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.7)

        for ax, plot_type in zip(axes, include_plots):
            ax.set_xlim(t_span)

            if plot_type == 'speeds':
                ax.step(t, w_m_ref/base.w, "--", where="post", label=r"$\omega_{m}^\mathrm{{ref}}$")
                ax.plot(t, w_m/base.w, label=r"$\omega_{m}$")
                ax.legend(loc='lower right')
                ax.set_ylabel(r"$\omega_{m}$ (p.u.)")

            elif plot_type == 'fluxes':
                ax.plot(t, np.abs(psi_sref)/base.psi, '--', label=r"$\psi_\mathrm{s}^\mathrm{ref}$")
                ax.step(t, np.abs(psi_sest)/base.psi, where="post", label=r"$\hat\psi_\mathrm{s}$")
                ax.legend(loc='right')
                ax.set_ylim([-0.1, 1.25])
                ax.set_ylabel(r"$\psi_\mathrm{s}$ (p.u.)")

            elif plot_type == 'torques':
                ax.plot(t, tau_Mest/base.tau, label=rf"$\hat \tau_{M}$")
                # ax.plot(t, tau_Mref/base.tau, "--", label=rf"$\tau_{M}^\mathrm{{ref}}$")
                ax.step(t, tau_Mref_lim/base.tau, "--", label=rf"$\tau_{M}^\mathrm{{ref,lim}}$")
                ax.step(t, tau_Lref, "--", label=r"$\tau_\mathrm{L}$")
                ax.legend(loc='right')
                ax.set_ylim([-0.1, 1.5])
                ax.set_ylabel(rf"$\tau_{M}$ (p.u.)")

            elif plot_type == 'currents':
                ax.step(t, i_c.real/base.i, where="post", label=r"$i_\mathrm{d}$")
                ax.step(t, i_c.imag/base.i, where="post", label=r"$i_\mathrm{q}$")
                # ax.step(t, i_ref.real/base.i, where="post", label=r"$i_\mathrm{sd,ref}$")
                # ax.step(t, i_ref.imag/base.i, where="post", label=r"$i_\mathrm{sq,ref}$")
                ax.legend(loc='right')
                ax.set_ylabel(r"$\mathbf{i}_\mathrm{s}$ (p.u.)")

            elif plot_type == 'voltages':
                ax.plot(t, u_s/base.u, label=r"$u_\mathrm{c}$", ds="steps-post")
                ax.plot(t, u_dc/np.sqrt(3)/base.u, "--", label=r"$u_\mathrm{dc}/\sqrt{3}$",
                        ds="steps-post")
                ax.legend(loc='right')
                ax.set_ylim([0.0, 1.2])
                ax.set_ylabel("Voltage (p.u.)")

        fig.align_ylabels()
        axes[-1].set_xlabel("Time (s)")
        return fig, axes


def _validate_window(t_offset, t_span, x_step):
    origin = float(t_offset)
    if not np.isfinite(origin):
        raise ValueError("t_offset must be finite")
    limits = None
    if t_span is not None:
        limits = np.asarray(t_span, dtype=float)
        if (limits.shape != (2,) or not np.all(np.isfinite(limits))
                or limits[1] <= limits[0]):
            raise ValueError("t_span must contain two finite, increasing limits")
        limits = tuple(float(value) for value in limits)
    step = None if x_step is None else float(x_step)
    if step is not None and (not np.isfinite(step) or step <= 0):
        raise ValueError("x_step must be finite and positive")
    return origin, limits, step


FIGURE_TYPES = {
    "overview": (plot_overview, {}, "_1"),
    "injection-overview": (plot_overview, {"bottom": "injection"}, "_2"),
    "hfsi": (plot_hfsi, {}, "_hfsi_2"),
    "presentation": (plot_presentation, {}, "_online_meas"),
}


def scalar_expression(expression, base):
    """Evaluate finite scalar arithmetic without executing Python code.

    Allowed names: n_p, sqrt3, u_base, i_base, w_base, psi_base, tau_base,
    tau_nom. Only +, -, *, / and parentheses are supported.
    """
    text = str(expression).strip()
    if len(text) > 120:
        raise ValueError('Scalar expression is too long')
    try:
        tree = ast.parse(text, mode='eval')
    except SyntaxError as error:
        raise ValueError('Invalid scalar expression') from error
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError('Scalar expression is too complex')
    names = {'n_p': base.n_p, 'sqrt3': np.sqrt(3), 'u_base': base.u,
             'i_base': base.i, 'w_base': base.w, 'psi_base': base.psi,
             'tau_base': base.tau, 'tau_nom': base.tau_nom}

    def calculate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in names:
            return float(names[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = calculate(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = calculate(node.left), calculate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        raise ValueError('Use numbers, base names, +, -, *, / and parentheses only')

    try:
        result = calculate(tree.body)
    except (ZeroDivisionError, OverflowError) as error:
        raise ValueError('Scalar expression must produce a finite number') from error
    if not np.isfinite(result):
        raise ValueError('Scalar expression must produce a finite number')
    return result


def subplot_traces(kind, groups=()):
    """Describe every plotted trace and its editable pre-normalization gain.

    The stage column documents non-affine derived operations that remain intact.
    Offsets use the selected/stage signal's input units, before p.u. conversion.
    """
    speed = [('wm_ref', '', '1', 'w'), ('w_m', '', '1', 'w')]
    flux = [('psis_ref', 'abs', '1', 'psi'), ('psi_s', 'abs(complex)', '1', 'psi')]
    torque = [('tauM_ref_lim', '', '1', 'tau'), ('tau_m', '', '1', 'tau'),
              ('tauLref / TL_ref', 'recorded p.u.', '1', '')]
    currents = [('i_c_abc + theta_m_out', 'real(complex current * exp(-j angle))', '1', 'i'),
                ('i_c_abc + theta_m_out', 'imag(complex current * exp(-j angle))', '1', 'i')]
    voltage = [('u_dc_lim', '', '1/sqrt3', 'u'), ('ucs_realized', 'abs(complex)', '1', 'u')]
    if kind == 'custom':
        return [{'panel': panel, 'title': f'Alikuvaaja {panel + 1}',
                 'id': f'custom:{panel}:{key}', 'source': key, 'stage': 'recorded value',
                 'scale': '1', 'offset': '0', 'divisor': ''}
                for panel, group in enumerate(groups) for key in group]
    if kind == 'hfsi':
        torque = [('tauM_ref_lim', '', '1', 'tau_nom'), ('tau_m', '', '1', 'tau_nom'),
                  ('tauLref / TL_ref', 'recorded p.u.; legacy rated-torque calibration', '1/0.88', '')]
        panels = [('Nopeus', speed + [('wm_meas', '', '1/2', 'w')]),
                  ('Virrat', currents), ('Vuo', flux),
                  ('Kulmavirhe', [('mon[3]', 'degrees', '1', ''),
                                 ('theta_m_out - theta_m_out', 'wrapped mechanical degrees', '1', '')]),
                  ('Momentti', torque),
                  ('Kulma', [('theta_m_out (observer)', 'wrap to (-pi, pi]', '1', ''),
                             ('theta_m_out (measured)', 'wrap to (-pi, pi]', '1', '')])]
    elif kind in ('overview', 'injection-overview', 'presentation'):
        if kind == 'presentation':
            torque = [torque[1], torque[0], torque[2]]
        panels = [('Nopeus', speed), ('Vuo', flux), ('Momentti', torque), ('Virrat', currents)]
        if kind == 'overview':
            panels.append(('Jännitteet', voltage))
        elif kind == 'injection-overview':
            panels.append(('Injektiojännitteet', [('u_sd_inj', 'real(complex)', '1', ''),
                                                 ('u_sd_inj', 'imag(complex)', '1', '')]))
    else:
        raise ValueError(f'Unknown figure type: {kind}')
    return [{'panel': panel, 'title': title, 'id': f'{kind}:{panel}:{line}',
             'source': source, 'stage': stage, 'scale': scale, 'offset': '0', 'divisor': divisor}
            for panel, (title, traces) in enumerate(panels)
            for line, (source, stage, scale, divisor) in enumerate(traces)]


def _apply_trace_transforms(context, kind, axes, transforms):
    if not transforms:
        return
    panels = np.asarray(axes).flat
    lines = {panel: iter(axis.get_lines()) for panel, axis in enumerate(panels)}
    modified = set()
    for trace in subplot_traces(kind):
        line = next(lines[trace['panel']])
        if trace['id'] not in transforms:
            continue
        change = transforms[trace['id']]
        scale = scalar_expression(change['scale'], context.base)
        offset = scalar_expression(change['offset'], context.base)
        default = scalar_expression(trace['scale'], context.base)
        divisor = getattr(context.base, trace['divisor']) if trace['divisor'] else 1.
        # Undo only the legacy affine gain; retain complex/angle processing.
        values = np.asarray(line.get_ydata()) * divisor / default
        line.set_ydata((values * scale + offset) / divisor)
        modified.add(trace['panel'])
    for panel, axis in enumerate(np.asarray(axes).flat):
        if panel in modified:
            axis.relim()
            axis.autoscale(enable=True, axis='y')


SIGNAL_UNITS = ('raw', 'A', 'V', 'rad/s', 'Nm', 'Wb', 'Vs',
                'rad/s (mechanical)', 'rpm (mechanical)',
                'p.u. (A)', 'p.u. (V)', 'p.u. (rad/s)', 'p.u. (Nm)', 'p.u. (Wb)',
                'p.u. (rad/s mechanical)', 'p.u. (Vs)')


def signal_scale(unit, base, per_unit, *, torque_mode='per_unit'):
    """Return scale and displayed unit; already-p.u. signals need a quantity.

    raw explicitly leaves a signal unchanged. rad/s denotes electrical speed;
    mechanical speed uses w_M and rpm is converted to rad/s. Vs and Wb share
    the flux base. Preset angular units and model-specific calibrations remain.
    """
    if torque_mode not in ('per_unit', 'rated'):
        raise ValueError("torque_mode must be 'per_unit' or 'rated'")
    if unit not in SIGNAL_UNITS:
        raise ValueError(f"Unknown signal unit: {unit}")
    if unit == 'raw':
        return 1., 'raw'
    already_pu = unit.startswith('p.u. (')
    physical = unit[6:-1] if already_pu else unit
    mechanical = physical in ('rad/s (mechanical)', 'rpm (mechanical)', 'rad/s mechanical')
    reference = base.w_M if mechanical else {'A': base.i, 'V': base.u, 'rad/s': base.w,
                                             'Nm': base.tau, 'Wb': base.psi, 'Vs': base.psi}[physical]
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError(f"Invalid base value for {physical}")
    if physical == 'Nm' and torque_mode == 'rated':
        if not np.isfinite(base.tau_nom) or base.tau_nom <= 0:
            raise ValueError('Invalid rated torque')
        return (reference if already_pu else 1.) / base.tau_nom, 'per rated'
    factor = 2*np.pi/60 if physical == 'rpm (mechanical)' else 1.
    if per_unit:
        return (1. if already_pu else factor/reference), 'p.u.'
    return (reference if already_pu else factor), ('rad/s' if mechanical else physical)


def _apply_preset_units(context, figure_type, axes, per_unit, torque_mode):
    base = context.base
    if torque_mode == 'rated':
        signal_scale('Nm', base, per_unit, torque_mode=torque_mode)
    if figure_type == 'hfsi':
        panels = [(axes[0, 0], base.w, 'rad/s'), (axes[1, 0], base.psi, 'Wb'),
                  (axes[0, 1], base.i, 'A')]
        torque_axis = axes[2, 0]
        factor = 1. if torque_mode == 'rated' else (base.tau_nom/base.tau if per_unit else base.tau_nom)
        label = torque_axis.get_ylabel().replace(
            'per rated', 'per rated' if torque_mode == 'rated' else ('p.u.' if per_unit else 'Nm'))
        _scale_axis(torque_axis, factor, label)
    else:
        panels = [(axes[0], base.w, 'rad/s'), (axes[1], base.psi, 'Wb'),
                  (axes[2], base.tau, 'Nm'), (axes[3], base.i, 'A')]
        if torque_mode == 'rated':
            torque_axis = axes[2]
            _scale_axis(torque_axis, base.tau/base.tau_nom,
                        torque_axis.get_ylabel().replace('p.u.', 'per rated'))
        if figure_type == 'overview':
            panels.append((axes[4], base.u, 'V'))
        elif figure_type == 'injection-overview' and per_unit is not None:
            axis = axes[4]
            _scale_axis(axis, 1./base.u if per_unit else 1.,
                        'Injection Voltages (p.u.)' if per_unit else 'Injection Voltages (V)')
    if per_unit is False:
        for axis, reference, unit in panels:
            if torque_mode == 'rated' and unit == 'Nm':
                continue
            _scale_axis(axis, reference, axis.get_ylabel().replace('p.u.', unit))


def _scale_axis(axis, factor, label):
    limits = np.asarray(axis.get_ylim()) * factor
    for line in axis.get_lines():
        line.set_ydata(np.asarray(line.get_ydata()) * factor)
    axis.set_ylim(limits)
    axis.set_ylabel(label)


def plot_measurement(context, figure_type, *, per_unit=None, transforms=None,
                     disabled_traces=None, added_traces=None, units=None,
                     torque_mode='per_unit', downsample_output=True):
    """Render a preset, optionally removing native traces or adding detached signals.

    Added trace transforms precede conversion from units[key]; missing units are
    raw. None preserves preset legacy units and selects physical units for additions.
    Rated torque uses base.tau_nom independently of other quantities' units.
    """
    try:
        function, options, _ = FIGURE_TYPES[figure_type]
    except (KeyError, TypeError):
        raise ValueError(f"figure_type must select from {tuple(FIGURE_TYPES)}") from None
    if torque_mode not in ('per_unit', 'rated'):
        raise ValueError("torque_mode must be 'per_unit' or 'rated'")
    traces = subplot_traces(figure_type)
    native_ids = {trace['id'] for trace in traces}
    panel_count = max(trace['panel'] for trace in traces) + 1
    if isinstance(disabled_traces, (str, bytes)):
        raise ValueError('disabled_traces must be a collection of trace IDs')
    disabled = set(disabled_traces or ())
    if disabled - native_ids:
        raise ValueError(f'Unknown preset trace IDs: {disabled - native_ids}')
    if added_traces is not None and not isinstance(added_traces, list):
        raise ValueError('added_traces must be a list')
    prepared = []
    ids = set(native_ids)
    for trace in added_traces or []:
        if not isinstance(trace, dict):
            raise ValueError('Each added trace must be a dictionary')
        trace_id, panel, key = trace.get('id'), trace.get('panel'), trace.get('key')
        if not isinstance(trace_id, str) or not trace_id or trace_id in ids:
            raise ValueError('Added trace IDs must be unique nonempty strings')
        ids.add(trace_id)
        if type(panel) is not int or not 0 <= panel < panel_count:
            raise ValueError(f'Invalid added trace panel: {panel}')
        if not isinstance(key, str) or not key:
            raise ValueError('Added signal keys must be nonempty strings')
        components = getattr(context, 'extra_signals', {}).get(key)
        if not components:
            raise ValueError(f"Extra signal '{key}' was not loaded")
        change = (transforms or {}).get(trace_id, {})
        scale = scalar_expression(change.get('scale', '1'), context.base)
        offset = scalar_expression(change.get('offset', '0'), context.base)
        for true_key, time, values in components:
            unit = (units or {}).get(true_key, (units or {}).get(key, 'raw'))
            factor, display_unit = signal_scale(unit, context.base, bool(per_unit),
                                                torque_mode=torque_mode)
            prepared.append((panel, f'{true_key} ({display_unit})',
                             time - context.t_offset, (values * scale + offset) * factor))
    existing = set(plt.get_fignums())
    try:
        fig, axes = function(context, **options)
        _apply_trace_transforms(context, figure_type, axes, transforms)
        if per_unit is not None or torque_mode == 'rated':
            _apply_preset_units(context, figure_type, axes, per_unit, torque_mode)
        if figure_type == 'hfsi' and torque_mode == 'rated':
            torque_axis = axes[2, 0]
            load_line = torque_axis.lines[2]
            load_id = 'hfsi:4:2'
            # The preset input is already base p.u.; its legacy 1/0.88 gain is
            # retained only when the caller explicitly supplies a transform.
            undo_legacy = 1. if load_id in (transforms or {}) else .88
            load_line.set_ydata(np.asarray(load_line.get_ydata()) * undo_legacy
                                * context.base.tau / context.base.tau_nom)
            if load_id in (transforms or {}):
                torque_axis.relim()
                torque_axis.autoscale(enable=True, axis='y')
        if figure_type == 'injection-overview':
            axes[4].legend(loc='center')
        flat = list(np.asarray(axes).flat)
        lines = {panel: iter(list(axis.get_lines())) for panel, axis in enumerate(flat)}
        affected = set()
        for trace in traces:
            line = next(lines[trace['panel']])
            if trace['id'] in disabled:
                line.remove()
                affected.add(trace['panel'])
        added_panels = set()
        for panel, label, time, values in prepared:
            flat[panel].plot(time, values, label=label,
                             drawstyle='steps-post' if time.size < 100 else 'default')
            affected.add(panel)
            added_panels.add(panel)
        for panel in affected:
            axis = flat[panel]
            legend = axis.get_legend()
            handles, labels = axis.get_legend_handles_labels()
            if legend is not None:
                if handles:
                    # Rebuild the entries in place to retain the preset's placement and styling.
                    title = legend.get_title().get_text()
                    location = legend._loc
                    legend._init_legend_box(handles, labels)
                    legend.set_title(title)
                    legend.set_loc(location)
                else:
                    legend.remove()
            elif handles:
                axis.legend(handles, labels)
            if panel in added_panels:
                axis.relim(visible_only=True)
                axis.autoscale(enable=True, axis='y')
        if downsample_output and figure_type in ('injection-overview', 'presentation'):
            downsample(fig)
        return fig, axes
    except Exception:
        for number in set(plt.get_fignums()) - existing:
            plt.close(number)
        raise


def plot_signals(data, groups, t_offset=0, t_span=None, x_step=None, *,
                 per_unit=False, base=None, units=None, transforms=None,
                 torque_mode='per_unit', columns=1):
    """Plot signals in one panel per list, on each component's own raster.

    Arrange panels in row-major order using 1 to 8 columns; return only the
    occupied axes as a one-dimensional array in group order.

    units maps custom:{panel_index}:{key}, component keys, or selected keys to
    SIGNAL_UNITS, in that precedence order. Missing entries remain raw; p.u.
    inputs include their physical quantity so they can be denormalized.
    Rated torque uses base.tau_nom independently of other quantities' units.
    Per-line legend units distinguish mixed quantities on the same subplot.

    Select readable keys using data.keys(). Vector keys expand through
    data.signal_info(), so legends use the actual component keys rather than
    invented indices. All signal and window validation precedes figure creation.
    The recording remains owned by the caller; returned lines hold copied data.
    """
    if isinstance(columns, bool) or not isinstance(columns, int) or not 1 <= columns <= 8:
        raise ValueError("columns must be an integer between 1 and 8")
    origin, limits, step = _validate_window(t_offset, t_span, x_step)
    if (not isinstance(groups, list) or not groups
            or any(not isinstance(group, list) or not group for group in groups)):
        raise ValueError("groups must be a nonempty list of nonempty key lists")
    motor_base = DEFAULT_BASE if base is None else base
    panels = []
    for panel_index, group in enumerate(groups):
        panel = []
        for key in group:
            if not isinstance(key, str) or not key:
                raise ValueError("Signal keys must be nonempty readable strings")
            signals = data.signal_info(key)
            if not signals:
                raise ValueError(f"Signal '{key}' has no components")
            for signal in signals:
                true_key = signal["key"]
                time = np.asarray(data.time(true_key), dtype=float)
                values = np.asarray(data[true_key])
                if (time.ndim != 1 or not time.size
                        or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0)):
                    raise ValueError(f"Signal '{true_key}' needs a finite increasing time vector")
                if (values.ndim != 1 or values.size != time.size
                        or not np.issubdtype(values.dtype, np.number)
                        or np.iscomplexobj(values)):
                    raise ValueError(f"Signal '{true_key}' must be a real vector matching its time")
                unit_map = units or {}
                trace_unit = unit_map.get(f'custom:{panel_index}:{key}',
                                          unit_map.get(true_key, unit_map.get(key, 'raw')))
                factor, unit = signal_scale(trace_unit, motor_base, per_unit, torque_mode=torque_mode)
                change = (transforms or {}).get(f'custom:{panel_index}:{key}', {'scale': '1', 'offset': '0'})
                scale = scalar_expression(change['scale'], motor_base)
                offset = scalar_expression(change['offset'], motor_base)
                label = f"{true_key} ({unit})" if units is not None else true_key
                panel.append((label, time.copy() - origin, (values.copy()*scale + offset)*factor))
        panels.append(panel)
    existing = set(plt.get_fignums())
    try:
        columns = min(columns, len(panels))
        rows = (len(panels) + columns - 1) // columns
        default_width, default_height = plt.rcParams['figure.figsize']
        size = (max(default_width, columns*3.2), max(default_height, rows*1.7))
        fig, grid = plt.subplots(rows, columns, sharex=True, squeeze=False, layout="tight", figsize=size)
        flat = grid.ravel()
        for ax in flat[len(panels):]:
            fig.delaxes(ax)
        axes = flat[:len(panels)]
        for index, (ax, panel) in enumerate(zip(axes, panels)):
            for key, time, values in panel:
                ax.plot(time, values, label=key,
                        drawstyle="steps-post" if time.size < 100 else "default")
            if step is not None:
                ax.xaxis.set_major_locator(MultipleLocator(step))
            ax.legend(loc="upper right")
            ax.grid(True)
            bottom = index + columns >= len(panels)
            ax.tick_params(axis="x", labelbottom=bottom)
            if bottom:
                ax.set_xlabel("Time (s)")
        if limits is not None:
            axes[-1].set_xlim(limits)
        return fig, axes
    except Exception:
        for number in set(plt.get_fignums()) - existing:
            plt.close(number)
        raise
