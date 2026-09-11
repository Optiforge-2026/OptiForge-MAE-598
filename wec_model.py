"""
Computational model for the Sea-Wave Energy Converter (WEC) optimization problem,
following the formulation in Sections 3-13 of the project report:

    x = [Rb, ts, Hb, N, B, g, RL]

    Rb : buoy outer radius            [m]
    ts : hull shell wall thickness    [m]
    Hb : buoy height                  [m]
    N  : number of coil turns         [turns]  (integer; relaxed to continuous for the NLP)
    B  : magnetic flux density        [T]
    g  : magnet-coil air gap          [m]
    RL : external load resistance     [Ohm]

Environmental inputs (fixed, not decision variables): wave amplitude A, wave
angular frequency omega (Eqs. 7-8 of the report).

All closed-form relations below (hoop stress -> min. thickness, shell mass,
material cost, Faraday/back-EMF constant, average electrical power) are taken
directly from Sections 5, 9-12 of the report. Sub-models that the report
specifies only in words (translator geometry, copper/magnet cost, thermal
current-density limit, generator stroke, freeboard) are made explicit here as
engineering closure assumptions -- flagged with `# ASSUMPTION` -- because the
report leaves them as physical requirements without a closed-form expression.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Fixed physical / material constants (Sections 5, 12.1 of the report)
# ---------------------------------------------------------------------------
RHO_W      = 1025.0      # seawater density [kg/m^3]              (Eq. 73)
CP         = 1.2         # dynamic pressure coefficient            (Eq. 73)
U_CURRENT  = 1.5         # characteristic current/drag velocity [m/s] (Eq. 72)

SIGMA_Y    = 21.9e6      # HDPE yield strength [Pa]                (Eq. 68)
FS         = 5.0         # structural factor of safety             (Eq. 70)
SIGMA_ALLOW = SIGMA_Y / FS

RHO_POLY   = 950.0       # HDPE density [kg/m^3]
C_POLY     = 2.5         # HDPE material cost [$/kg]

RHO_CU     = 8960.0      # copper density [kg/m^3]
C_CU       = 9.0         # copper cost [$/kg]
A_WIRE0    = 2.0e-6      # nominal copper-wire cross-section [m^2] (~AWG 15) # ASSUMPTION

RHO_MAG    = 7500.0      # NdFeB magnet density [kg/m^3]
C_MAG_PER_M = 2000.0     # magnet-stack cost per unit stack length [$/m]  # ASSUMPTION
MAG_FRAC   = 0.5         # fraction of translator volume occupied by magnet # ASSUMPTION

C_BOP      = 130000.0    # fixed mooring + installation ("balance of plant") cost [$]
                          # kept consistent with the report's Table 2 narrative
                          # ("... dominated by fixed mooring and installation costs")

KT         = 0.06        # translator radius as a fraction of buoy radius  # ASSUMPTION
LM_FRAC    = 0.30         # magnet-stack length as a fraction of buoy height # ASSUMPTION
K_FILL     = 0.6          # copper winding fill factor                     # ASSUMPTION

# ---------------------------------------------------------------------------
# Environmental / sea-state inputs (Eqs. 7-8) -- fixed, representative
# mid-period swell condition
# ---------------------------------------------------------------------------
A_WAVE      = 1.00        # wave-induced displacement amplitude [m]
T_WAVE      = 6.5         # wave period [s]
OMEGA_WAVE  = 2.0 * np.pi / T_WAVE   # wave angular frequency [rad/s]

# ---------------------------------------------------------------------------
# Engineering / equipment limits (Section 4)
# ---------------------------------------------------------------------------
V_MAX_TRANSLATOR = 6.0     # generator design velocity limit [m/s]   (Eq. 10)
A_MAX_TRANSLATOR = 40.0    # generator design accel. limit  [m/s^2]  (Eq. 11)
V_RATED    = 800.0        # rated voltage [V]                        (Eq. 12)
I_RATED    = 80.0         # rated current [A]                        (Eq. 13)
J_MAX      = 5.0e6        # max current density [A/m^2]              (Eq. 15)
B_MAX      = 1.30         # magnetic saturation limit [T]            (Eq. 16)
DPHIDX_MAX = 10.0          # max flux gradient [Wb/m]                (Eq. 17) # ASSUMPTION scale
M_MAX      = 25000.0      # mass budget [kg]                         (Eq. 18)
C_MAX      = 200000.0     # cost budget [$]                          (Eq. 19)

D_MIN, D_MAX = 0.8, 3.0    # buoy diameter bounds [m]                 (Eq. 20)
H_MIN, H_MAX = 0.8, 2.5    # buoy height bounds [m]                   (Eq. 21)
N_MIN, N_MAX = 50, 800     # coil-turn bounds [turns]                 (Eq. 22)

# Normalization references for the scalarized objective (Eq. 59)
P_REF = 5000.0     # W
M_REF = 5000.0     # kg
C_REF = 150000.0   # $


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------
def hydrodynamic_pressure():
    """Eq. (24)/(72): equivalent hydrodynamic pressure on the buoy shell."""
    return 0.5 * RHO_W * U_CURRENT**2 * CP


def hoop_stress(Rb, ts, Pw):
    """Eq. (25)/(74): thin-wall hoop stress."""
    return Pw * Rb / ts


def hull_mass(Rb, ts, Hb):
    """Eq. (28)-(29)/(79)-(80): thin-walled cylindrical shell mass."""
    Vs = np.pi * Hb * (2.0 * Rb * ts - ts**2)
    return RHO_POLY * Vs, Vs


def translator_geometry(Rb, Hb):
    rt = KT * Rb                      # translator/coil radius     # ASSUMPTION
    Ac = np.pi * rt**2                # coil cross-sectional area
    Lm = LM_FRAC * Hb                 # magnet stack length         # ASSUMPTION
    return rt, Ac, Lm


def ke_backemf(N, B, Rb, g, Hb):
    """
    Eq. (42): Ke = N * dPhi/dx.
    dPhi/dx is approximated by a linear magnetic-circuit closure
    dPhi/dx ~ B * Ac / g  (flux collapses to zero across the air gap g).   # ASSUMPTION
    """
    rt, Ac, Lm = translator_geometry(Rb, Hb)
    dphidx = B * Ac / g
    Ke = N * dphidx
    return Ke, dphidx, Ac, rt, Lm


def average_power(N, B, Rb, g, RL, Hb):
    """Eq. (55): Pe(x) = Ke^2 A^2 omega^2 / (2 RL)."""
    Ke, dphidx, Ac, rt, Lm = ke_backemf(N, B, Rb, g, Hb)
    Pe = (Ke**2) * (A_WAVE**2) * (OMEGA_WAVE**2) / (2.0 * RL)
    return Pe, Ke, dphidx, Ac, rt, Lm


def electrical_signals(Ke):
    """Peak voltage/current for sinusoidal translator motion (Eqs. 43-45)."""
    v_peak = A_WAVE * OMEGA_WAVE
    V_peak = Ke * v_peak
    return V_peak, v_peak


def copper_mass(N, rt, Ac):
    """Simplified copper-mass model: N turns of mean length 2*pi*rt,
    fixed wire cross-section A_WIRE0.                                     # ASSUMPTION
    """
    turn_length = 2.0 * np.pi * rt
    Vcu = N * turn_length * A_WIRE0
    return RHO_CU * Vcu, Vcu


def magnet_mass(Ac, Lm):
    Vmag = MAG_FRAC * Ac * Lm
    return RHO_MAG * Vmag, Vmag


def current_density(I_peak, N, Ac):
    """Approximate winding current density: N turns share area Ac*K_FILL."""
    Awire_eff = (Ac * K_FILL) / max(N, 1.0)
    return I_peak / Awire_eff


def total_mass(x):
    Rb, ts, Hb, N, B, g, RL = x
    Mhull, _ = hull_mass(Rb, ts, Hb)
    _, Ac, Lm = translator_geometry(Rb, Hb)
    rt = KT * Rb
    Mcu, _ = copper_mass(N, rt, Ac)
    Mmag, _ = magnet_mass(Ac, Lm)
    return Mhull + Mcu + Mmag


def total_cost(x):
    Rb, ts, Hb, N, B, g, RL = x
    Mhull, _ = hull_mass(Rb, ts, Hb)
    _, Ac, Lm = translator_geometry(Rb, Hb)
    rt = KT * Rb
    Mcu, _ = copper_mass(N, rt, Ac)
    Ccu = C_CU * Mcu
    Cmag = C_MAG_PER_M * Lm
    return C_BOP + C_POLY * Mhull + Ccu + Cmag


def full_evaluation(x):
    """Return every physical quantity needed for objective + constraints."""
    Rb, ts, Hb, N, B, g, RL = x
    Pw = hydrodynamic_pressure()
    sigma_h = hoop_stress(Rb, ts, Pw)

    Pe, Ke, dphidx, Ac, rt, Lm = average_power(N, B, Rb, g, RL, Hb)
    V_peak, v_peak = electrical_signals(Ke)
    I_peak = V_peak / RL
    Jc = current_density(I_peak, N, Ac)

    Mtotal = total_mass(x)
    Ctotal = total_cost(x)

    hd = Mtotal / (RHO_W * np.pi * Rb**2)   # draft, Eq. (34)

    return dict(Pw=Pw, sigma_h=sigma_h, Pe=Pe, Ke=Ke, dphidx=dphidx, Ac=Ac,
                rt=rt, Lm=Lm, V_peak=V_peak, v_peak=v_peak, I_peak=I_peak,
                Jc=Jc, Mtotal=Mtotal, Ctotal=Ctotal, hd=hd)


# ---------------------------------------------------------------------------
# Objective (Eq. 59) and constraints (Section 4)
# ---------------------------------------------------------------------------
def objective(x, weights):
    wP, wM, wC = weights
    q = full_evaluation(x)
    J = (-wP * q['Pe'] / P_REF
         + wM * q['Mtotal'] / M_REF
         + wC * q['Ctotal'] / C_REF)
    return J


def constraints_list():
    """
    Return a list of scipy-style inequality constraint functions, g(x) >= 0.
    Every constraint is normalized by its own limiting scale so that all
    entries are O(1); this materially improves SLSQP conditioning relative
    to leaving physical quantities (stress ~1e6, power ~1e3-1e4, etc.) in
    their raw, wildly different units.
    """
    def g_stress(x):
        q = full_evaluation(x)
        return (SIGMA_ALLOW - q['sigma_h']) / SIGMA_ALLOW

    def g_voltage(x):
        q = full_evaluation(x)
        return (V_RATED - q['V_peak']) / V_RATED

    def g_current(x):
        q = full_evaluation(x)
        return (I_RATED - q['I_peak']) / I_RATED

    def g_currdensity(x):
        q = full_evaluation(x)
        return (J_MAX - q['Jc']) / J_MAX

    def g_bsat(x):
        return (B_MAX - x[4]) / B_MAX

    def g_fluxgrad(x):
        q = full_evaluation(x)
        return (DPHIDX_MAX - q['dphidx']) / DPHIDX_MAX

    def g_stroke(x):
        # translator stroke must fit within the available internal buoy height
        Hb = x[2]
        return (0.6 * Hb - A_WAVE) / H_MAX

    def g_freeboard(x):
        q = full_evaluation(x)
        Hb = x[2]
        return (0.9 * Hb - q['hd']) / H_MAX

    def g_mass(x):
        q = full_evaluation(x)
        return (M_MAX - q['Mtotal']) / M_MAX

    def g_cost(x):
        q = full_evaluation(x)
        return (C_MAX - q['Ctotal']) / C_MAX

    cons = [g_stress, g_voltage, g_current, g_currdensity, g_bsat,
            g_fluxgrad, g_stroke, g_freeboard, g_mass, g_cost]
    names = ['stress', 'voltage', 'current', 'current_density', 'B_saturation',
             'flux_gradient', 'stroke_fit', 'freeboard', 'mass_budget', 'cost_budget']
    return cons, names


BOUNDS = [
    (D_MIN / 2.0, D_MAX / 2.0),   # Rb
    (0.004, 0.03),                 # ts
    (H_MIN, H_MAX),                # Hb
    (N_MIN, N_MAX),                 # N (relaxed continuous)
    (0.2, B_MAX),                   # B
    (0.003, 0.015),                 # g
    (5.0, 200.0),                   # RL
]

VAR_NAMES = ['Rb', 'ts', 'Hb', 'N', 'B', 'g', 'RL']
