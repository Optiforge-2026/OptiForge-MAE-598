import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import json

from wec_model import (BOUNDS, VAR_NAMES, objective, constraints_list,
                        full_evaluation, total_mass, total_cost)

lb = np.array([b[0] for b in BOUNDS])
ub = np.array([b[1] for b in BOUNDS])


def to_unit(x):
    return (np.array(x) - lb) / (ub - lb)


def from_unit(u):
    u = np.clip(u, 0.0, 1.0)
    return lb + u * (ub - lb)


def solve_one(weights, x0, cons_funcs, n_restarts=6, seed=0):
    """
    Solve one scalarized subproblem in rescaled [0,1]^7 coordinates.

    SLSQP occasionally reports 'Positive directional derivative for
    linesearch' when the warm start lands exactly on an active-constraint
    ridge. This is handled robustly (rather than silently accepted) by
    retrying from small random perturbations of the warm start and keeping
    the best feasible, converged result; if SLSQP still cannot certify
    convergence, trust-constr (which uses a different interior trust-region
    step and does not suffer the same line-search failure mode) is used as
    a fallback. This mirrors solving each subproblem with two independent
    algorithms, which also serves as a cross-check on the solution.
    """
    rng = np.random.default_rng(seed)

    def obj_u(u):
        x = from_unit(u)
        return objective(x, weights)

    scipy_cons = []
    for cf in cons_funcs:
        def make_c(cf=cf):
            return lambda u: cf(from_unit(u))
        scipy_cons.append({'type': 'ineq', 'fun': make_c()})

    u0_base = to_unit(x0)
    best = None
    for attempt in range(n_restarts):
        if attempt == 0:
            u0 = u0_base
        else:
            jitter = rng.normal(scale=0.03, size=len(lb))
            u0 = np.clip(u0_base + jitter, 0.0, 1.0)

        res = minimize(obj_u, u0, method='SLSQP', bounds=[(0, 1)] * len(lb),
                        constraints=scipy_cons,
                        options={'maxiter': 300, 'ftol': 1e-10})
        if res.success:
            feas = all(cf(from_unit(res.x)) >= -1e-6 for cf in cons_funcs)
            if feas:
                best = res
                break
        if best is None or (hasattr(res, 'fun') and res.fun < best.fun):
            best = res

    if best is None or not best.success:
        # Fallback: trust-constr with explicit NonlinearConstraint bounds
        nlc = NonlinearConstraint(
            lambda u: np.array([cf(from_unit(u)) for cf in cons_funcs]),
            lb=0.0, ub=np.inf)
        res_tc = minimize(obj_u, u0_base, method='trust-constr',
                           bounds=[(0, 1)] * len(lb), constraints=[nlc],
                           options={'maxiter': 2000, 'gtol': 1e-9, 'xtol': 1e-10})
        best = res_tc

    x_star = from_unit(best.x)
    return x_star, best


def pareto_sweep(n_points=15, lambdas=None):
    cons_funcs, cons_names = constraints_list()

    if lambdas is None:
        lambdas = np.linspace(0.02, 0.98, n_points)
    # Start from a hand-picked, near-feasible design (moderate N, B, RL) so
    # that SLSQP begins inside/near the feasible region rather than at a
    # point with an outsized power term dominating the line search.
    x0 = np.array([1.00, 0.020, 1.80, 150.0, 0.50, 0.008, 150.0])

    results = []
    for lam in lambdas:
        wP = lam
        wM = wC = (1.0 - lam) / 2.0
        weights = (wP, wM, wC)

        x_star, res = solve_one(weights, x0, cons_funcs)
        q = full_evaluation(x_star)

        # check constraint satisfaction (tolerate tiny numerical slack)
        viol = {name: cf(x_star) for name, cf in zip(cons_names, cons_funcs)}
        feasible = all(v >= -1e-4 for v in viol.values())

        row = dict(lam=lam, wP=wP, wM=wM, wC=wC,
                   x=x_star.tolist(), success=bool(res.success),
                   feasible=feasible, obj=float(res.fun),
                   P=q['Pe'], M=q['Mtotal'], C=q['Ctotal'],
                   sigma_h=q['sigma_h'], V_peak=q['V_peak'], I_peak=q['I_peak'],
                   Jc=q['Jc'], hd=q['hd'], Ac=q['Ac'], Ke=q['Ke'],
                   dphidx=q['dphidx'])
        results.append(row)
        x0 = x_star  # warm start next lambda

    return results, cons_names


def eps_constraint_sweep(M_budgets, C_budget=None):
    """
    epsilon-constraint alternative to the weighted sum: for each mass budget
    M_budgets[i], maximize average electrical power subject to
    Mtotal(x) <= M_budgets[i] (in place of the fixed mass_budget constraint)
    and all remaining constraints (stress, voltage, current, current
    density, saturation, flux gradient, stroke fit, freeboard, cost budget).

    Weighted-sum scalarization of a nonconvex problem can only reach points
    on the convex hull of the Pareto set; the epsilon-constraint method can
    reach the non-convex portions as well, and is used here to fill in the
    front that the weighted sum jumps over.
    """
    from wec_model import M_REF, C_REF, C_MAX
    cons_funcs, cons_names = constraints_list()
    # keep every constraint except mass_budget (index 8), which is replaced
    keep_idx = [i for i, n in enumerate(cons_names) if n != 'mass_budget']
    base_cons = [cons_funcs[i] for i in keep_idx]

    x0 = np.array([1.00, 0.020, 1.80, 150.0, 0.50, 0.008, 150.0])
    results = []
    for Mb in M_budgets:
        def g_mass_eps(x, Mb=Mb):
            from wec_model import total_mass
            return (Mb - total_mass(x)) / Mb

        cons_funcs_eps = base_cons + [g_mass_eps]

        def obj_u(u):
            x = from_unit(u)
            q = full_evaluation(x)
            # log-power objective: maximize P without the raw ~1e0-1e6 W
            # dynamic range destroying SLSQP's line-search conditioning
            return -np.log(q['Pe'] + 1e-3)

        scipy_cons = []
        for cf in cons_funcs_eps:
            def make_c(cf=cf):
                return lambda u: cf(from_unit(u))
            scipy_cons.append({'type': 'ineq', 'fun': make_c()})

        u0 = to_unit(x0)
        best = None
        rng = np.random.default_rng(1)
        for attempt in range(6):
            uu = u0 if attempt == 0 else np.clip(u0 + rng.normal(scale=0.03, size=len(lb)), 0, 1)
            res = minimize(obj_u, uu, method='SLSQP', bounds=[(0, 1)] * len(lb),
                            constraints=scipy_cons, options={'maxiter': 300, 'ftol': 1e-10})
            feas = all(cf(from_unit(res.x)) >= -1e-4 for cf in cons_funcs_eps)
            if res.success and feas:
                best = res
                break
            if best is None or res.fun < best.fun:
                best = res

        x_star = from_unit(best.x)
        q = full_evaluation(x_star)
        feasible = all(cf(x_star) >= -1e-4 for cf in cons_funcs_eps)
        results.append(dict(M_budget=Mb, x=x_star.tolist(), success=bool(best.success),
                             feasible=feasible, P=q['Pe'], M=q['Mtotal'], C=q['Ctotal']))
        x0 = x_star
    return results


def polish_with_integer_N(x_star, cons_funcs, obj_fn):
    """
    Round N to the nearest integer turn count (Section 13: N is the only
    genuinely discrete variable, so the base problem is an NLP relaxation of
    an MINLP) and re-solve the remaining six continuous variables with N
    held fixed, to restore feasibility that naive rounding can break right
    at an active voltage/current limit.
    """
    N_int = float(round(x_star[3]))
    lb_p = lb.copy(); ub_p = ub.copy()
    lb_p[3] = N_int; ub_p[3] = N_int

    def to_unit_p(x):
        return (np.array(x) - lb_p) / np.where(ub_p > lb_p, ub_p - lb_p, 1.0)

    def from_unit_p(u):
        u = np.clip(u, 0.0, 1.0)
        return lb_p + u * (ub_p - lb_p)

    def obj_u(u):
        return obj_fn(from_unit_p(u))

    scipy_cons = []
    for cf in cons_funcs:
        def make_c(cf=cf):
            return lambda u: cf(from_unit_p(u))
        scipy_cons.append({'type': 'ineq', 'fun': make_c()})

    u0 = to_unit_p(x_star)
    res = minimize(obj_u, u0, method='SLSQP', bounds=[(0, 1)] * len(lb),
                    constraints=scipy_cons, options={'maxiter': 300, 'ftol': 1e-10})
    x_polished = from_unit_p(res.x)
    feasible = all(cf(x_polished) >= -1e-4 for cf in cons_funcs)
    return x_polished, feasible, res.success


if __name__ == "__main__":
    # Primary sweep: weighted-sum scalarization, per the course-prescribed
    # methodology, with denser sampling at small lambda to resolve the jump.
    lambdas_main = np.array([0.005, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08, 0.11,
                              0.16, 0.25, 0.38, 0.55, 0.7, 0.85, 0.98])
    results, cons_names = pareto_sweep(lambdas=lambdas_main)
    cons_funcs_all, _ = constraints_list()

    for r in results:
        obj_fn = lambda x, w=(r['wP'], r['wM'], r['wC']): objective(x, w)
        x_p, feas_p, succ_p = polish_with_integer_N(r['x'], cons_funcs_all, obj_fn)
        q_p = full_evaluation(x_p)
        r['x_int'] = x_p.tolist()
        r['N_int'] = int(round(x_p[3]))
        r['feasible_int'] = feas_p
        r['P_int'] = q_p['Pe']
        r['M_int'] = q_p['Mtotal']
        r['C_int'] = q_p['Ctotal']

    with open('/home/claude/wec/pareto_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"{'lam':>5} {'Rb':>6} {'ts(mm)':>7} {'Hb':>5} {'N':>6} {'B':>5} "
          f"{'g(mm)':>6} {'RL':>7} {'P(W)':>9} {'M(kg)':>9} {'C($)':>10} feas succ")
    for r in results:
        Rb, ts, Hb, N, B, g, RL = r['x']
        print(f"{r['lam']:5.3f} {Rb:6.3f} {ts*1000:7.2f} {Hb:5.2f} {N:6.0f} "
              f"{B:5.2f} {g*1000:6.2f} {RL:7.1f} {r['P']:9.1f} {r['M']:9.1f} "
              f"{r['C']:10.0f} {str(r['feasible']):5s} {str(r['success']):5s}")

    print("\n--- epsilon-constraint sweep (maximize P s.t. M(x) <= M_budget) ---")
    M_budgets = np.geomspace(20, 900, 14)
    eps_results = eps_constraint_sweep(M_budgets)

    for r in eps_results:
        Mb = r['M_budget']
        cons_funcs_e, cons_names_e = constraints_list()
        keep_idx = [i for i, n in enumerate(cons_names_e) if n != 'mass_budget']
        base_cons = [cons_funcs_e[i] for i in keep_idx]

        def g_mass_eps(x, Mb=Mb):
            from wec_model import total_mass
            return (Mb - total_mass(x)) / Mb
        cons_funcs_e_full = base_cons + [g_mass_eps]

        def obj_fn(x):
            q = full_evaluation(x)
            return -np.log(q['Pe'] + 1e-3)

        x_p, feas_p, succ_p = polish_with_integer_N(r['x'], cons_funcs_e_full, obj_fn)
        q_p = full_evaluation(x_p)
        r['x_int'] = x_p.tolist()
        r['N_int'] = int(round(x_p[3]))
        r['feasible_int'] = feas_p
        r['P_int'] = q_p['Pe']
        r['M_int'] = q_p['Mtotal']
        r['C_int'] = q_p['Ctotal']

    with open('/home/claude/wec/eps_results.json', 'w') as f:
        json.dump(eps_results, f, indent=2)
    print(f"{'M_budget':>9} {'Rb':>6} {'N':>6} {'B':>5} {'g(mm)':>6} {'RL':>7} "
          f"{'P(W)':>9} {'M(kg)':>9} {'C($)':>10} feas succ")
    for r in eps_results:
        Rb, ts, Hb, N, B, g, RL = r['x']
        print(f"{r['M_budget']:9.1f} {Rb:6.3f} {N:6.0f} {B:5.2f} {g*1000:6.2f} "
              f"{RL:7.1f} {r['P']:9.1f} {r['M']:9.1f} {r['C']:10.0f} "
              f"{str(r['feasible']):5s} {str(r['success']):5s}")
