"""
OPTIFORGE TEAM 8
PATH B CONTROL EXPERIMENT — CAUSALITY / TIME-MARCHING PINN

Question
--------
Is the long-time failure of the standard PINN primarily caused by
full-domain (non-causal) optimization?

Controlled comparison
---------------------
A) Standard PINN:
   one network trained over 0 <= t <= 100.

B) Causal/time-marching PINN:
   same architecture, PDE, loss weights, optimizer, BCs and IC treatment,
   but trained sequentially over:
       [0,20] -> [20,40] -> [40,60] -> [60,80] -> [80,100]

Corrected PDE
-------------
U_t = -mu U - r U_xx - 3 U_xxxx
      - 2 gamma (U^2)_xx + (2/3)(U^3)_xx

Residual:
R = U_t + mu U + r U_xx + 3 U_xxxx
    + 2 gamma (U^2)_xx - (2/3)(U^3)_xx

Loss in BOTH experiments
------------------------
L = L_PDE + 10 L_IC + L_BC

The spectral solution is used ONLY for evaluation.
"""

from pathlib import Path
import time
import platform
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import torch
import torch.nn as nn

# ============================================================
# 0. SETTINGS
# ============================================================

SEED = 7

L = 100.0
N = 128

GAMMA = 1.0
MU = 0.01
R = 2.0 - GAMMA**2

T_FINAL = 100.0
T_SUB = 20.0

N_WINDOWS = int(
    T_FINAL / T_SUB
)

# Reference/evaluation grid
N_TIME = 101

# PINN architecture
WIDTH = 64

# Training
GLOBAL_STEPS = 3000
WINDOW_STEPS = 1500

NR_GLOBAL = 1800
NR_WINDOW = 1000

NIC = 384
NBC = 192

LEARNING_RATE = 1e-3

W_PDE = 1.0
W_IC = 10.0
W_BC = 1.0

HERE = Path(__file__).resolve().parent

OUT = (
    HERE
    / "zonal_jet_outputs_causality_control"
)

OUT.mkdir(
    exist_ok=True
)

np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

DTYPE = torch.float32

print("="*72)
print("PATH B — CORRECTED CAUSALITY CONTROL")
print("="*72)

print("device:", device)
print("torch:", torch.__version__)
print("OS:", platform.platform())


# ============================================================
# 1. SPECTRAL REFERENCE
# ============================================================

x = np.linspace(
    0.0,
    L,
    N,
    endpoint=False
)

k = (
    2.0*np.pi
    * np.fft.fftfreq(
        N,
        d=L/N
    )
)

rng = np.random.default_rng(
    SEED
)

initial_modes = [
    (3, 0.20),
    (4, 0.18),
    (5, 0.14),
    (6, 0.10)
]

phases = {}

u0 = np.zeros(
    N
)

for mode, amplitude in initial_modes:

    phase = rng.uniform(
        0.0,
        2.0*np.pi
    )

    phases[mode] = phase

    u0 += (
        amplitude
        * np.cos(
            2.0*np.pi
            * mode
            * x
            / L
            + phase
        )
    )

u0 -= u0.mean()


def spectral_rhs(
    t,
    u
):

    uh = np.fft.fft(
        u
    )

    uxx = np.fft.ifft(
        -(k**2)*uh
    ).real

    uxxxx = np.fft.ifft(
        (k**4)*uh
    ).real

    u2xx = np.fft.ifft(
        -(k**2)
        * np.fft.fft(
            u*u
        )
    ).real

    u3xx = np.fft.ifft(
        -(k**2)
        * np.fft.fft(
            u*u*u
        )
    ).real

    return (
        -MU*u
        -R*uxx
        -3.0*uxxxx
        -2.0*GAMMA*u2xx
        +(2.0/3.0)*u3xx
    )


ts = np.linspace(
    0.0,
    T_FINAL,
    N_TIME
)

print(
    "\nComputing spectral reference..."
)

tic = time.perf_counter()

sol = solve_ivp(
    spectral_rhs,
    (0.0, T_FINAL),
    u0,
    t_eval=ts,
    method="BDF",
    rtol=2e-6,
    atol=1e-8,
    max_step=5.0
)

if not sol.success:

    raise RuntimeError(
        sol.message
    )

Uref = sol.y.T

spectral_seconds = (
    time.perf_counter()
    - tic
)

print(
    f"Spectral solve: "
    f"{spectral_seconds:.6f} s"
)


# ============================================================
# 2. SAME NETWORK FOR BOTH EXPERIMENTS
# ============================================================

class ParameterizedNet(
    nn.Module
):

    def __init__(
        self,
        time_scale
    ):

        super().__init__()

        self.time_scale = float(
            time_scale
        )

        self.net = nn.Sequential(

            nn.Linear(
                2,
                WIDTH
            ),

            nn.Tanh(),

            nn.Linear(
                WIDTH,
                WIDTH
            ),

            nn.Tanh(),

            nn.Linear(
                WIDTH,
                WIDTH
            ),

            nn.Tanh(),

            nn.Linear(
                WIDTH,
                1
            )
        )

    def forward(
        self,
        xv,
        tv
    ):

        z = torch.cat(
            [
                2.0*xv/L - 1.0,

                2.0*tv/self.time_scale
                - 1.0
            ],
            dim=1
        )

        return self.net(
            z
        )


def D(
    y,
    z
):

    return torch.autograd.grad(
        y,
        z,
        torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]


def analytic_initial_condition(
    xq
):

    value = torch.zeros_like(
        xq
    )

    for mode, amplitude in initial_modes:

        value += (
            amplitude
            * torch.cos(
                2.0*np.pi
                * mode
                * xq
                / L
                + phases[mode]
            )
        )

    return value


# ============================================================
# 3. CORRECTED PDE RESIDUAL
# ============================================================

def pde_residual(
    model,
    xr,
    tr
):

    U = model(
        xr,
        tr
    )

    U_t = D(
        U,
        tr
    )

    U_x = D(
        U,
        xr
    )

    U_xx = D(
        U_x,
        xr
    )

    U_xxx = D(
        U_xx,
        xr
    )

    U_xxxx = D(
        U_xxx,
        xr
    )

    U2_xx = D(
        D(
            U*U,
            xr
        ),
        xr
    )

    U3_xx = D(
        D(
            U*U*U,
            xr
        ),
        xr
    )

    residual = (
        U_t
        + MU*U
        + R*U_xx
        + 3.0*U_xxxx
        + 2.0*GAMMA*U2_xx
        - (2.0/3.0)*U3_xx
    )

    return residual


# ============================================================
# 4. PERIODIC BC LOSS
# ============================================================

def periodic_bc_loss(
    model,
    tmax,
    nbc=NBC
):

    tau = (
        tmax
        * torch.rand(
            nbc,
            1,
            device=device,
            dtype=DTYPE
        )
    )

    x_left = torch.zeros(
        nbc,
        1,
        device=device,
        dtype=DTYPE,
        requires_grad=True
    )

    x_right = torch.full(
        (
            nbc,
            1
        ),
        L,
        device=device,
        dtype=DTYPE,
        requires_grad=True
    )

    U_left = model(
        x_left,
        tau
    )

    U_right = model(
        x_right,
        tau
    )

    Ux_left = D(
        U_left,
        x_left
    )

    Ux_right = D(
        U_right,
        x_right
    )

    value_loss = torch.mean(
        (
            U_left
            - U_right
        )**2
    )

    slope_loss = torch.mean(
        (
            Ux_left
            - Ux_right
        )**2
    )

    return (
        value_loss
        + slope_loss
    )


# ============================================================
# 5. STANDARD GLOBAL PINN
# ============================================================

def train_standard_pinn():

    print(
        "\n"
        + "="*72
    )

    print(
        "TRAINING STANDARD FULL-DOMAIN PINN"
    )

    print(
        "="*72
    )

    torch.manual_seed(
        SEED
    )

    model = ParameterizedNet(
        T_FINAL
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        SEED
    )

    xr = (
        L
        * torch.rand(
            NR_GLOBAL,
            1,
            generator=generator,
            device=device,
            dtype=DTYPE
        )
    ).requires_grad_(True)

    tr = (
        T_FINAL
        * torch.rand(
            NR_GLOBAL,
            1,
            generator=generator,
            device=device,
            dtype=DTYPE
        )
    ).requires_grad_(True)

    xic = (
        L
        * torch.rand(
            NIC,
            1,
            generator=generator,
            device=device,
            dtype=DTYPE
        )
    )

    history = []

    tic = time.perf_counter()

    for iteration in range(
        GLOBAL_STEPS+1
    ):

        optimizer.zero_grad()

        residual = pde_residual(
            model,
            xr,
            tr
        )

        Lpde = torch.mean(
            residual**2
        )

        predicted_ic = model(
            xic,
            torch.zeros_like(xic)
        )

        Lic = torch.mean(
            (
                predicted_ic
                - analytic_initial_condition(
                    xic
                )
            )**2
        )

        Lbc = periodic_bc_loss(
            model,
            T_FINAL
        )

        total = (
            W_PDE*Lpde
            + W_IC*Lic
            + W_BC*Lbc
        )

        total.backward()

        optimizer.step()

        if iteration % 100 == 0:

            history.append(
                [
                    iteration,
                    Lpde.item(),
                    Lic.item(),
                    Lbc.item(),
                    total.item()
                ]
            )

            print(
                f"standard "
                f"{iteration:5d} | "
                f"PDE={Lpde.item():.3e} | "
                f"IC={Lic.item():.3e} | "
                f"BC={Lbc.item():.3e} | "
                f"total={total.item():.3e}"
            )

    seconds = (
        time.perf_counter()
        - tic
    )

    return (
        model,
        np.asarray(history),
        seconds
    )


# ============================================================
# 6. CAUSAL / TIME-MARCHING PINN
# ============================================================

def train_causal_pinn():

    print(
        "\n"
        + "="*72
    )

    print(
        "TRAINING CAUSAL TIME-MARCHING PINN"
    )

    print(
        "="*72
    )

    x_grid = torch.tensor(
        x[:, None],
        dtype=DTYPE,
        device=device
    )

    # Original physical IC for first window.
    u_ic_current = torch.tensor(
        u0[:, None],
        dtype=DTYPE,
        device=device
    )

    causal_times = [
        0.0
    ]

    causal_fields = [
        u0.copy()
    ]

    histories = []

    window_models = []

    total_training_seconds = 0.0

    for window in range(
        N_WINDOWS
    ):

        t_start = (
            window
            * T_SUB
        )

        t_end = (
            (window+1)
            * T_SUB
        )

        print(
            "\n"
            + "-"*72
        )

        print(
            f"WINDOW {window+1}/{N_WINDOWS}: "
            f"physical time "
            f"{t_start:g} -> {t_end:g}"
        )

        print(
            "-"*72
        )

        # Same architecture.
        # Local network time is tau in [0,T_SUB].
        torch.manual_seed(
            SEED + window
        )

        model = ParameterizedNet(
            T_SUB
        ).to(
            device
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE
        )

        generator = torch.Generator(
            device=device
        )

        generator.manual_seed(
            SEED + 100 + window
        )

        history = []

        tic = time.perf_counter()

        for iteration in range(
            WINDOW_STEPS+1
        ):

            optimizer.zero_grad()

            xr = (
                L
                * torch.rand(
                    NR_WINDOW,
                    1,
                    generator=generator,
                    device=device,
                    dtype=DTYPE
                )
            ).requires_grad_(True)

            tau_r = (
                T_SUB
                * torch.rand(
                    NR_WINDOW,
                    1,
                    generator=generator,
                    device=device,
                    dtype=DTYPE
                )
            ).requires_grad_(True)

            residual = pde_residual(
                model,
                xr,
                tau_r
            )

            Lpde = torch.mean(
                residual**2
            )

            tau_zero = torch.zeros(
                N,
                1,
                device=device,
                dtype=DTYPE
            )

            U_start = model(
                x_grid,
                tau_zero
            )

            Lic = torch.mean(
                (
                    U_start
                    - u_ic_current
                )**2
            )

            Lbc = periodic_bc_loss(
                model,
                T_SUB
            )

            total = (
                W_PDE*Lpde
                + W_IC*Lic
                + W_BC*Lbc
            )

            total.backward()

            optimizer.step()

            if iteration % 100 == 0:

                history.append(
                    [
                        iteration,
                        Lpde.item(),
                        Lic.item(),
                        Lbc.item(),
                        total.item()
                    ]
                )

                print(
                    f"window={window+1} "
                    f"iter={iteration:4d} | "
                    f"PDE={Lpde.item():.3e} | "
                    f"IC={Lic.item():.3e} | "
                    f"BC={Lbc.item():.3e} | "
                    f"total={total.item():.3e}"
                )

        total_training_seconds += (
            time.perf_counter()
            - tic
        )

        histories.append(
            np.asarray(
                history
            )
        )

        window_models.append(
            model
        )

        # ----------------------------------------------------
        # Save predictions inside this window.
        # ----------------------------------------------------

        local_times = np.linspace(
            0.0,
            T_SUB,
            21
        )

        # Starting point is already stored from previous window.
        if window > 0:

            local_times = local_times[
                1:
            ]

        with torch.no_grad():

            for tau_value in local_times:

                tau_tensor = torch.full(
                    (
                        N,
                        1
                    ),
                    float(
                        tau_value
                    ),
                    device=device,
                    dtype=DTYPE
                )

                prediction = model(
                    x_grid,
                    tau_tensor
                )

                physical_time = (
                    t_start
                    + tau_value
                )

                causal_times.append(
                    physical_time
                )

                causal_fields.append(
                    prediction
                    .cpu()
                    .numpy()
                    .ravel()
                )

        # ----------------------------------------------------
        # Pass final state to next window.
        # ----------------------------------------------------

        with torch.no_grad():

            tau_final = torch.full(
                (
                    N,
                    1
                ),
                T_SUB,
                device=device,
                dtype=DTYPE
            )

            u_ic_current = model(
                x_grid,
                tau_final
            ).detach()

    causal_times = np.asarray(
        causal_times
    )

    causal_fields = np.asarray(
        causal_fields
    )

    order = np.argsort(
        causal_times
    )

    causal_times = causal_times[
        order
    ]

    causal_fields = causal_fields[
        order
    ]

    return (
        causal_times,
        causal_fields,
        histories,
        window_models,
        total_training_seconds
    )


# ============================================================
# 7. EVALUATE STANDARD PINN
# ============================================================

def evaluate_global_model(
    model
):

    xx = torch.tensor(
        x[:, None],
        dtype=DTYPE,
        device=device
    )

    prediction = []

    model.eval()

    with torch.no_grad():

        for tv in ts:

            tt = torch.full_like(
                xx,
                float(tv)
            )

            prediction.append(
                model(
                    xx,
                    tt
                )
                .cpu()
                .numpy()
                .ravel()
            )

    return np.asarray(
        prediction
    )


# ============================================================
# 8. METRICS
# ============================================================

def relative_l2(
    prediction,
    reference
):

    return (
        np.linalg.norm(
            prediction-reference,
            axis=1
        )
        /
        np.maximum(
            np.linalg.norm(
                reference,
                axis=1
            ),
            1e-12
        )
    )


def jet_count(
    field
):

    threshold = (
        0.2
        * max(
            float(
                np.max(field)
            ),
            1e-9
        )
    )

    count = 0

    for index in range(
        len(field)
    ):

        if (
            field[index] > threshold
            and field[index] > field[index-1]
            and field[index] > field[
                (index+1)
                % len(field)
            ]
        ):

            count += 1

    return count


def energy(
    field
):

    # E(t) = 1/2 integral U^2 d eta
    return (
        0.5
        * np.trapezoid(
            field**2,
            x=x,
            axis=1
        )
    )


# ============================================================
# 9. RUN BOTH CONTROL EXPERIMENTS
# ============================================================

standard_model, H_standard, standard_seconds = (
    train_standard_pinn()
)

U_standard = evaluate_global_model(
    standard_model
)

(
    causal_times,
    U_marching,
    H_causal,
    causal_models,
    causal_seconds
) = train_causal_pinn()


# ============================================================
# 10. INTERPOLATE CAUSAL SOLUTION ONTO REFERENCE GRID
# ============================================================

U_causal = np.zeros_like(
    Uref
)

for spatial_index in range(
    N
):

    U_causal[
        :,
        spatial_index
    ] = np.interp(
        ts,
        causal_times,
        U_marching[
            :,
            spatial_index
        ]
    )


# ============================================================
# 11. COMPUTE METRICS
# ============================================================

R_standard = relative_l2(
    U_standard,
    Uref
)

R_causal = relative_l2(
    U_causal,
    Uref
)

J_ref = np.asarray(
    [
        jet_count(v)
        for v in Uref
    ]
)

J_standard = np.asarray(
    [
        jet_count(v)
        for v in U_standard
    ]
)

J_causal = np.asarray(
    [
        jet_count(v)
        for v in U_causal
    ]
)

E_ref = energy(
    Uref
)

E_standard = energy(
    U_standard
)

E_causal = energy(
    U_causal
)


# ============================================================
# 12. PRINT RESULTS
# ============================================================

print(
    "\n"
    + "="*72
)

print(
    "CAUSALITY CONTROL RESULTS"
)

print(
    "="*72
)

for target_time in [
    5,
    20,
    40,
    60,
    80,
    90,
    100
]:

    index = int(
        np.argmin(
            np.abs(
                ts-target_time
            )
        )
    )

    improvement = (
        (
            R_standard[index]
            - R_causal[index]
        )
        /
        max(
            R_standard[index],
            1e-12
        )
        * 100.0
    )

    print(
        f"t={ts[index]:5.1f} | "
        f"L2 std/causal="
        f"{R_standard[index]:.4f}/"
        f"{R_causal[index]:.4f} | "
        f"improvement={improvement:+.1f}% | "
        f"jets ref/std/causal="
        f"{J_ref[index]}/"
        f"{J_standard[index]}/"
        f"{J_causal[index]}"
    )


final_improvement = (
    (
        R_standard[-1]
        - R_causal[-1]
    )
    /
    max(
        R_standard[-1],
        1e-12
    )
    * 100.0
)

print(
    "\nFinal relative L2:"
)

print(
    "standard =",
    R_standard[-1]
)

print(
    "causal   =",
    R_causal[-1]
)

print(
    f"relative improvement = "
    f"{final_improvement:.2f}%"
)

print(
    "\nFinal jets "
    "reference/standard/causal =",
    J_ref[-1],
    J_standard[-1],
    J_causal[-1]
)

print(
    "\nTraining seconds:"
)

print(
    f"standard = "
    f"{standard_seconds:.2f}"
)

print(
    f"causal   = "
    f"{causal_seconds:.2f}"
)

print(
    f"spectral = "
    f"{spectral_seconds:.6f}"
)


# ============================================================
# 13. DECISION LOGIC
# ============================================================

print(
    "\n"
    + "="*72
)

print(
    "PATH B CAUSALITY DECISION"
)

print(
    "="*72
)

ratio = (
    R_causal[-1]
    /
    max(
        R_standard[-1],
        1e-12
    )
)

if ratio < 0.50:

    conclusion = (
        "STRONG evidence for a causality problem: "
        "time-marching reduced the final error by "
        "more than 50%."
    )

elif ratio < 0.80:

    conclusion = (
        "MODERATE evidence that causality contributes "
        "to the failure."
    )

elif ratio < 1.00:

    conclusion = (
        "Only SMALL improvement from time-marching. "
        "Causality may contribute, but is unlikely "
        "to be the dominant failure mechanism."
    )

else:

    conclusion = (
        "Time-marching did NOT improve the final error. "
        "This control does not support causality as "
        "the dominant explanation."
    )

print(
    conclusion
)


# ============================================================
# 14. FIGURE — FINAL FIELD + ERROR HISTORY
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        12,
        4.8
    )
)

axes[0].plot(
    x,
    Uref[-1],
    linewidth=2,
    label="Spectral reference"
)

axes[0].plot(
    x,
    U_standard[-1],
    "--",
    linewidth=2,
    label="Standard PINN"
)

axes[0].plot(
    x,
    U_causal[-1],
    "-.",
    linewidth=2,
    label="Causal PINN"
)

axes[0].set_xlabel(
    "eta"
)

axes[0].set_ylabel(
    "U"
)

axes[0].set_title(
    "Final field"
)

axes[0].grid(
    alpha=0.3
)

axes[0].legend()


axes[1].plot(
    ts,
    R_standard,
    linewidth=2,
    label="Standard PINN"
)

axes[1].plot(
    ts,
    R_causal,
    linewidth=2,
    label="Causal PINN"
)

for boundary in np.arange(
    T_SUB,
    T_FINAL,
    T_SUB
):

    axes[1].axvline(
        boundary,
        linestyle=":",
        alpha=0.45
    )

axes[1].set_xlabel(
    "time"
)

axes[1].set_ylabel(
    "Relative L2 error"
)

axes[1].set_title(
    "Causality control"
)

axes[1].grid(
    alpha=0.3
)

axes[1].legend()

fig.tight_layout()

fig.savefig(
    OUT
    / "01_causality_control.png",
    dpi=200
)


# ============================================================
# 15. FIGURE — JET COUNTS
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        8,
        4.8
    )
)

ax.plot(
    ts,
    J_ref,
    linewidth=2,
    label="Reference"
)

ax.plot(
    ts,
    J_standard,
    "--",
    label="Standard PINN"
)

ax.plot(
    ts,
    J_causal,
    "-.",
    label="Causal PINN"
)

for boundary in np.arange(
    T_SUB,
    T_FINAL,
    T_SUB
):

    ax.axvline(
        boundary,
        linestyle=":",
        alpha=0.4
    )

ax.set_xlabel(
    "time"
)

ax.set_ylabel(
    "Detected eastward jets"
)

ax.set_title(
    "Physical jet-count control"
)

ax.grid(
    alpha=0.3
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUT
    / "02_jet_count_control.png",
    dpi=200
)


# ============================================================
# 16. FIGURE — ENERGY
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        8,
        4.8
    )
)

ax.plot(
    ts,
    E_ref,
    linewidth=2,
    label="Reference"
)

ax.plot(
    ts,
    E_standard,
    "--",
    label="Standard PINN"
)

ax.plot(
    ts,
    E_causal,
    "-.",
    label="Causal PINN"
)

ax.set_xlabel(
    "time"
)

ax.set_ylabel(
    "E(t) = 1/2 integral U^2 d eta"
)

ax.set_title(
    "Energy / amplitude recovery"
)

ax.grid(
    alpha=0.3
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUT
    / "03_energy_control.png",
    dpi=200
)


# ============================================================
# 17. FIGURE — FOURIER ERROR AT FINAL TIME
# ============================================================

standard_error = (
    U_standard[-1]
    - Uref[-1]
)

causal_error = (
    U_causal[-1]
    - Uref[-1]
)

S_standard = (
    np.abs(
        np.fft.rfft(
            standard_error
        )
    )**2
    / N**2
)

S_causal = (
    np.abs(
        np.fft.rfft(
            causal_error
        )
    )**2
    / N**2
)

modes = np.arange(
    1,
    21
)

fig, ax = plt.subplots(
    figsize=(
        8,
        4.8
    )
)

ax.semilogy(
    modes,
    S_standard[
        1:21
    ] + 1e-16,
    "o-",
    label="Standard error"
)

ax.semilogy(
    modes,
    S_causal[
        1:21
    ] + 1e-16,
    "s-",
    label="Causal error"
)

ax.set_xlabel(
    "Fourier mode"
)

ax.set_ylabel(
    "Error energy"
)

ax.set_title(
    "Final Fourier-error comparison"
)

ax.grid(
    alpha=0.3
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUT
    / "04_fourier_error_control.png",
    dpi=200
)


# ============================================================
# 18. 3D SPACE-TIME COMPARISON
# ============================================================

ETA, TIME = np.meshgrid(
    x,
    ts
)

zmin = float(
    min(
        Uref.min(),
        U_standard.min(),
        U_causal.min()
    )
)

zmax = float(
    max(
        Uref.max(),
        U_standard.max(),
        U_causal.max()
    )
)

fig = plt.figure(
    figsize=(
        17,
        5.8
    )
)

fields = [
    (
        Uref,
        "Spectral reference"
    ),

    (
        U_standard,
        "Standard PINN"
    ),

    (
        U_causal,
        "Causal PINN"
    )
]

for panel, (
    field,
    title
) in enumerate(
    fields,
    start=1
):

    ax = fig.add_subplot(
        1,
        3,
        panel,
        projection="3d"
    )

    ax.plot_surface(
        ETA,
        TIME,
        field,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        rcount=min(
            80,
            len(ts)
        ),
        ccount=min(
            100,
            len(x)
        ),
        vmin=zmin,
        vmax=zmax
    )

    ax.set_xlabel(
        "eta"
    )

    ax.set_ylabel(
        "time"
    )

    ax.set_zlabel(
        "U"
    )

    ax.set_zlim(
        zmin,
        zmax
    )

    ax.set_title(
        title
    )

    ax.view_init(
        elev=28,
        azim=-125
    )

fig.suptitle(
    "Path B causality control"
)

fig.tight_layout()

fig.savefig(
    OUT
    / "05_3D_causality_control.png",
    dpi=190
)


# ============================================================
# 19. SAVE NUMERICAL RESULTS
# ============================================================

np.savez(
    OUT
    / "causality_control_results.npz",
    x=x,
    ts=ts,
    Uref=Uref,
    U_standard=U_standard,
    U_causal=U_causal,
    R_standard=R_standard,
    R_causal=R_causal,
    J_ref=J_ref,
    J_standard=J_standard,
    J_causal=J_causal,
    E_ref=E_ref,
    E_standard=E_standard,
    E_causal=E_causal
)


with open(
    OUT
    / "causality_control_results.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "OPTIFORGE TEAM 8 — PATH B CAUSALITY CONTROL\n"
    )

    f.write(
        "============================================\n"
    )

    f.write(
        "Corrected cubic coefficient = 2/3\n"
    )

    f.write(
        f"window_size={T_SUB}\n"
    )

    f.write(
        f"number_of_windows={N_WINDOWS}\n"
    )

    f.write(
        f"standard_final_relative_L2="
        f"{R_standard[-1]:.8f}\n"
    )

    f.write(
        f"causal_final_relative_L2="
        f"{R_causal[-1]:.8f}\n"
    )

    f.write(
        f"final_relative_improvement_percent="
        f"{final_improvement:.4f}\n"
    )

    f.write(
        f"final_jets_ref_standard_causal="
        f"{J_ref[-1]}/"
        f"{J_standard[-1]}/"
        f"{J_causal[-1]}\n"
    )

    f.write(
        f"standard_training_seconds="
        f"{standard_seconds:.8f}\n"
    )

    f.write(
        f"causal_training_seconds="
        f"{causal_seconds:.8f}\n"
    )

    f.write(
        f"spectral_seconds="
        f"{spectral_seconds:.8f}\n"
    )

    f.write(
        f"decision={conclusion}\n"
    )

    for target_time in [
        5,
        20,
        40,
        60,
        80,
        90,
        100
    ]:

        index = int(
            np.argmin(
                np.abs(
                    ts-target_time
                )
            )
        )

        f.write(
            f"t={ts[index]:.1f},"
            f"L2_std={R_standard[index]:.8f},"
            f"L2_causal={R_causal[index]:.8f},"
            f"jets="
            f"{J_ref[index]}/"
            f"{J_standard[index]}/"
            f"{J_causal[index]}\n"
        )


print(
    "\nSaved all causality-control results to:"
)

print(
    OUT
)

print(
    "\nThe spectral reference was used only for evaluation."
)

plt.show()
