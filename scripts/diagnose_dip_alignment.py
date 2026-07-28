"""Ten targeted diagnostics for the rejected dip-constrained alignment model.

Each block answers one question from the failure analysis with a measured
number rather than an opinion. Everything is computed from an
``InferenceTask`` — no ``TVT`` label reaches any diagnostic that a model could
consume. The target is read **only** in the clearly marked post-prediction
error block, which exists to quantify the failure after the fact.

    python scripts/diagnose_dip_alignment.py --max-wells 120
    python scripts/diagnose_dip_alignment.py --protocol same_well_masked

Outputs (into REPORTS_DIR):

    dip_alignment_diagnostics_wells.csv    one row per well
    dip_alignment_diagnostics_summary.csv  one row per (protocol, question)

Questions, in the order of ``reports/dip_alignment_failure_analysis.md``:

    Q2  TVT + Z coordinate convention          tvt_plus_z_* / tvt_minus_z_*
    Q3  dip sign and gradient direction        dip_sign_*, gradient_*
    Q4  local apparent dip identifiability     xy_condition, perp_span_ft, ...
    Q5  GR amplitude calibration stability     gr_gain_*, gr_gain_clipped
    Q6  GR resolution compatibility            window_ft, tvt_per_window, ...
    Q7  alignment window length                dtvt_per_window vs search
    Q8  fallback dominance                     fallback_fraction, confidence
    Q9  systematic bias                        bias_*, track_move vs truth_move
    Q1/Q10 follow from the above.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from scripts._bootstrap import bootstrap
except ImportError:  # loose-file execution
    sys.path.insert(0, str(Path.cwd()))
    from scripts._bootstrap import bootstrap
bootstrap()

import numpy as np
import pandas as pd

from src.baselines import DipConstrainedGRTypewellAlignment
from src.data import discover_wells, load_well
from src.features import (
    DIP_ALIGNMENT_SEARCH,
    DIP_ALIGNMENT_WINDOW,
    TypewellReference,
    build_features,
    calibrate_gr_to_reference,
    gr_features,
)
from src.paths import ensure_reports_dir, require_competition_data
from src.tasks import TaskConstructionError, make_task
from src.validation import PROTOCOL_A, PROTOCOL_B, filter_blocked

MODES = {PROTOCOL_A: "masked", PROTOCOL_B: "real"}


def _dominant_wavelength(ref: TypewellReference) -> float:
    """Dominant TVT wavelength of the typewell GR log, in feet."""
    if not ref.ok or ref.gr_z.size < 8:
        return np.nan
    g = ref.gr_z - ref.gr_z.mean()
    spectrum = np.abs(np.fft.rfft(g))
    freq = np.fft.rfftfreq(g.size, d=ref.step)
    if spectrum.size < 2:
        return np.nan
    k = int(np.argmax(spectrum[1:])) + 1
    return float(1.0 / freq[k]) if freq[k] > 0 else np.nan


def _plane_fit(px, py, surface):
    """Return (coef, r2, residual_sigma) for the ridge-penalised plane fit."""
    A = np.column_stack([px, py, np.ones_like(px)])
    gram = A.T @ A
    try:
        coef = np.linalg.solve(gram + np.diag([1e-4, 1e-4, 0.0]), A.T @ surface)
    except np.linalg.LinAlgError:
        return None, np.nan, np.nan
    resid = surface - A @ coef
    denom = float(np.sum((surface - surface.mean()) ** 2))
    r2 = float(1.0 - resid @ resid / denom) if denom > 1e-12 else np.nan
    sigma = float(np.sqrt(resid @ resid / max(len(surface) - 3, 1)))
    return coef, r2, sigma


def diagnose_well(task, protocol: str) -> dict | None:
    """All target-free diagnostics for one well, plus post-hoc error metrics."""
    inp = task.inputs()
    inp.assert_no_target()
    s, row = inp.start, inp.anchor_row
    if s < 30 or row < 0:
        return None

    out: dict = {"protocol": protocol, "well_id": inp.well_id, "prefix_len": int(s),
                 "suffix_len": int(inp.n_predict)}

    known = np.isfinite(inp.tvt_known[:s])
    m = known & np.isfinite(inp.x[:s]) & np.isfinite(inp.y[:s]) & np.isfinite(inp.z[:s])
    if int(m.sum()) < 30:
        return None
    x0, y0 = float(inp.x[row]), float(inp.y[row])
    px = (inp.x[:s][m] - x0) / 1000.0
    py = (inp.y[:s][m] - y0) / 1000.0

    # -- Q2: which sign of Z flattens the structural surface? --------------
    # Both fits are reported. If the plane can absorb either sign with a
    # comparable R2 the convention is *not* determined by this test, and the
    # residual scatter of the raw sum is the tie-breaker.
    for label, sign in (("tvt_plus_z", +1.0), ("tvt_minus_z", -1.0)):
        surface = inp.tvt_known[:s][m] + sign * inp.z[:s][m]
        coef, r2, sigma = _plane_fit(px, py, surface)
        out[f"{label}_std"] = float(np.std(surface))
        out[f"{label}_r2"] = r2
        out[f"{label}_residual_sigma"] = sigma
    out["z_increases_with_md"] = bool(
        np.nanmean(np.diff(inp.z[:s][np.isfinite(inp.z[:s])])) > 0
    ) if int(np.isfinite(inp.z[:s]).sum()) > 1 else None

    # -- Q1: scale dominance of Z inside the fitted quantity TVT + Z -------
    # The plane is fitted to TVT + Z but only the TVT part is wanted. When Z
    # varies far more than TVT across the prefix, the fit is driven by the
    # wellbore's own vertical shape; the residual after subtracting Z back off
    # is then a *difference of large numbers*, and a small relative misfit in
    # the Z term becomes a large absolute TVT error.
    z_pref = inp.z[:s][m]
    t_pref = inp.tvt_known[:s][m]
    out["z_std_prefix"] = float(np.std(z_pref))
    out["tvt_std_prefix"] = float(np.std(t_pref))
    out["z_over_tvt_scale"] = (
        out["z_std_prefix"] / out["tvt_std_prefix"] if out["tvt_std_prefix"] > 1e-9 else np.inf
    )
    # Non-linearity of Z along the trajectory: the part of the Z term a planar
    # X/Y fit provably cannot reproduce, and therefore leaks into TVT 1:1.
    _, r2_z, sigma_z = _plane_fit(px, py, z_pref)
    out["z_plane_r2"] = r2_z
    out["z_nonplanar_sigma_ft"] = sigma_z
    # Error amplification: how large is the unmodelled Z wobble compared with
    # the whole TVT signal the model is trying to predict?
    out["z_nonplanarity_over_tvt_std"] = (
        sigma_z / out["tvt_std_prefix"] if out["tvt_std_prefix"] > 1e-9 else np.inf
    )

    # -- Q1b: is the hard-coded unit Z coefficient right? ------------------
    # ``dip_constrained_prediction`` evaluates
    #     pred = anchor + planar_XY_term - 1.0 * (Z - Z_anchor)
    # so it asserts dTVT/dZ = -1 exactly, after the planar trend. The
    # established GeometricProjection baseline instead *fits* that transfer
    # coefficient on fold-train wells. Measuring the empirical coefficient
    # shows how big an assumption the hard-coded -1 is.
    dz_pref = z_pref - float(inp.z[row])
    resid_tvt = t_pref - float(inp.anchor_tvt if np.isfinite(inp.anchor_tvt) else 0.0)
    if np.std(dz_pref) > 1e-9:
        out["empirical_dtvt_dz"] = float(np.polyfit(dz_pref, resid_tvt, 1)[0])
        out["assumed_dtvt_dz"] = -1.0
        out["dtvt_dz_error"] = out["empirical_dtvt_dz"] - (-1.0)
    # The lever arm the coefficient error is multiplied by: how much Z moves
    # across the region the model must predict. A masked boundary placed early
    # in the prefix can sit in the build section, where this is enormous.
    z_hidden = inp.z[inp.start : inp.stop]
    z_hidden = z_hidden[np.isfinite(z_hidden)]
    out["z_span_predicted_ft"] = float(np.ptp(z_hidden)) if z_hidden.size else np.nan
    out["z_step_from_anchor_ft"] = (
        float(np.nanmax(np.abs(z_hidden - float(inp.z[row])))) if z_hidden.size else np.nan
    )
    # Error implied by the coefficient being wrong, over that lever arm. This
    # is the single largest term when the predicted region is not flat-lying.
    out["implied_z_coefficient_error_ft"] = (
        abs(out.get("dtvt_dz_error", np.nan)) * out["z_step_from_anchor_ft"]
    )
    out["convention_flatter"] = (
        "TVT+Z" if out["tvt_plus_z_std"] < out["tvt_minus_z_std"] else "TVT-Z"
    )
    out["convention_r2_margin"] = out["tvt_plus_z_r2"] - out["tvt_minus_z_r2"]

    # -- Q4: is the X/Y plane identifiable from the visible prefix? --------
    P = np.column_stack([inp.x[:s][m], inp.y[:s][m]])
    Pc = P - P.mean(axis=0)
    _, sv, vt = np.linalg.svd(Pc, full_matrices=False)
    along, perp = vt[0], vt[1]
    out["xy_singular_ratio"] = float(sv[0] / max(sv[1], 1e-12))
    out["along_span_ft"] = float(np.ptp(Pc @ along))
    out["perp_span_ft"] = float(np.ptp(Pc @ perp))
    out["heading_deg"] = float(np.degrees(np.arctan2(along[1], along[0])))
    # Eigenvalues of the XY Gram block in (1,000 ft)^2 units set how hard the
    # 1e-4 ridge penalty bites on the cross-track direction.
    A = np.column_stack([px, py, np.ones_like(px)])
    eig = np.linalg.eigvalsh((A.T @ A)[:2, :2])
    lam_perp = float(max(eig[0], 1e-30))
    out["gram_lambda_perp"] = lam_perp
    out["gram_lambda_along"] = float(eig[1])
    out["ridge_shrinkage_perp"] = float(1e-4 / lam_perp)
    surface = inp.tvt_known[:s][m] + inp.z[:s][m]
    _, _, sigma = _plane_fit(px, py, surface)
    out["cross_track_dip_sigma"] = float(sigma / np.sqrt(lam_perp))
    hx = (inp.x[inp.start : inp.stop] - x0) / 1000.0
    hy = (inp.y[inp.start : inp.stop] - y0) / 1000.0
    lever = np.abs(hx * perp[0] + hy * perp[1])
    out["hidden_perp_lever_kft"] = float(np.nanmax(lever)) if lever.size else np.nan
    out["unidentified_dip_tvt_ft"] = out["cross_track_dip_sigma"] * out["hidden_perp_lever_kft"]
    # How far past the fitted support does the hidden region extrapolate?
    h_along = (np.column_stack([hx * 1000.0, hy * 1000.0]) - Pc.mean(axis=0)) @ along
    out["extrapolation_ratio"] = float(
        np.nanmax(np.abs(h_along)) / max(out["along_span_ft"], 1e-9)
    ) if h_along.size else np.nan

    # -- Q5/Q6/Q7: GR calibration and resolution --------------------------
    ref = TypewellReference(inp.tw_tvt, inp.tw_gr)
    grf = gr_features(inp)
    gr_z = grf["gr_z"]
    gr_missing = grf["gr_is_missing"] > 0.5
    out["has_typewell"] = bool(ref.ok)
    out["gr_missing_frac"] = float(np.mean(gr_missing))
    md_step = float(np.nanmedian(np.diff(inp.md))) if inp.n_rows > 1 else np.nan
    out["md_step_ft"] = md_step
    out["window_ft"] = DIP_ALIGNMENT_WINDOW * md_step
    dt = np.abs(np.diff(inp.tvt_known[:s][known]))
    tvt_rate = float(np.nanmedian(dt) / md_step) if dt.size and md_step else np.nan
    out["tvt_rate_per_ft"] = tvt_rate
    out["dtvt_per_window"] = tvt_rate * out["window_ft"] if np.isfinite(tvt_rate) else np.nan
    out["search_ft"] = float(DIP_ALIGNMENT_SEARCH)
    if ref.ok:
        out["typewell_span_ft"] = float(ref.tvt_max - ref.tvt_min)
        out["typewell_gr_wavelength_ft"] = _dominant_wavelength(ref)
        out["typewell_step_ft"] = float(ref.step)
        # Section traversed per window, as a fraction of one GR cycle: below
        # ~1 the window cannot see a full feature to lock onto.
        out["cycles_per_window"] = (
            out["dtvt_per_window"] / out["typewell_gr_wavelength_ft"]
            if np.isfinite(out["dtvt_per_window"]) and out["typewell_gr_wavelength_ft"]
            else np.nan
        )
        # Amplitude calibration: the raw robust gain and whether it was clipped.
        good = known & np.isfinite(gr_z[:s])
        if int(good.sum()) >= 50:
            obs = gr_z[:s][good]
            want = np.interp(inp.tvt_known[:s][good], ref.grid, ref.gr_z)
            o_scale = float(np.median(np.abs(obs - np.median(obs))))
            w_scale = float(np.median(np.abs(want - np.median(want))))
            gain = w_scale / o_scale if o_scale > 1e-6 else np.nan
            out["gr_gain_raw"] = gain
            out["gr_gain_clipped"] = bool(np.isfinite(gain) and not (0.2 <= gain <= 5.0))
            out["gr_gain_applied"] = float(np.clip(gain, 0.2, 5.0)) if np.isfinite(gain) else np.nan
            # Split-half stability of the gain across the prefix.
            half = int(good.sum()) // 2
            gains = []
            for sl in (slice(0, half), slice(half, None)):
                o, w = obs[sl], want[sl]
                if o.size < 20:
                    continue
                os_ = float(np.median(np.abs(o - np.median(o))))
                ws_ = float(np.median(np.abs(w - np.median(w))))
                if os_ > 1e-6:
                    gains.append(ws_ / os_)
            out["gr_gain_halves_ratio"] = (
                float(max(gains) / max(min(gains), 1e-9)) if len(gains) == 2 else np.nan
            )
        signal = calibrate_gr_to_reference(inp, ref, gr_z)
        expected = np.interp(inp.tvt_known[:s], ref.grid, ref.gr_z)
        cm = known & np.isfinite(signal[:s]) & np.isfinite(expected) & ~gr_missing[:s]
        if int(cm.sum()) >= 10:
            a, b = signal[:s][cm], expected[cm]
            if np.std(a) > 1e-9 and np.std(b) > 1e-9:
                out["prefix_gr_correlation"] = float(np.corrcoef(a, b)[0, 1])

    # -- Q3/Q8/Q9: the model's own behaviour ------------------------------
    feats = build_features(inp, alignment=False, dip_alignment=True)
    d = feats.dip_align
    model = DipConstrainedGRTypewellAlignment()
    pred = model.predict(inp, feats)
    diag = model.prediction_diagnostics(inp, feats, pred)
    out.update(
        {
            "alignment_ok": bool(d["ok"]),
            "alignment_failure_reason": str(d["failure_reason"]),
            "dip_fit_r2": float(d["dip_r2"]),
            "confidence_mean": diag.get("alignment_confidence_mean", np.nan),
            "confidence_p10": diag.get("alignment_confidence_p10", np.nan),
            "fallback_fraction": diag.get("fallback_fraction", np.nan),
        }
    )
    sl = slice(inp.start, inp.stop)
    expected_grad = np.asarray(d["expected_gradient"][sl], dtype="float64")
    prefix_grad = np.asarray(d["expected_gradient"][:s], dtype="float64")
    out["expected_gradient_mean"] = float(np.nanmean(expected_grad)) if expected_grad.size else np.nan
    out["prefix_gradient_mean"] = float(np.nanmean(prefix_grad)) if prefix_grad.size else np.nan
    # Observed prefix dTVT/dMD, the quantity the projected gradient should match
    # in both magnitude and sign.
    tv, md = inp.tvt_known[:s][known], inp.md[:s][known]
    if tv.size > 10 and np.ptp(md) > 1e-6:
        observed = float(np.polyfit(md, tv, 1)[0])
        out["observed_prefix_gradient"] = observed
        out["gradient_sign_agrees"] = bool(
            np.isfinite(out["prefix_gradient_mean"])
            and np.sign(observed) == np.sign(out["prefix_gradient_mean"])
        )
        out["gradient_ratio"] = (
            out["prefix_gradient_mean"] / observed if abs(observed) > 1e-12 else np.nan
        )
    track = np.asarray(d["track"], dtype="float64")
    if d["ok"] and inp.start < track.size and np.isfinite(track[inp.start]):
        move = track[sl] - track[inp.start]
        out["track_move_max"] = float(np.nanmax(np.abs(move)))
    tw = inp.tw_tvt[np.isfinite(inp.tw_tvt)] if inp.tw_tvt is not None else np.array([])
    out["typewell_clip_fraction"] = (
        float(np.mean((pred <= tw.min() + 1e-9) | (pred >= tw.max() - 1e-9))) if tw.size >= 2 else np.nan
    )

    # -- post-prediction validation diagnostics ONLY ----------------------
    # The target is read here and nowhere else. Nothing computed below is a
    # feature or reachable from a model.
    truth = np.asarray(task.scored(), dtype="float64") if task.target is not None else np.array([])
    if truth.size and truth.size == pred.size:
        err = pred - truth
        fin = np.isfinite(err)
        if fin.any():
            out["alignment_rmse"] = float(np.sqrt(np.nanmean(err[fin] ** 2)))
            out["alignment_bias"] = float(np.nanmean(err[fin]))
            out["alignment_bias_abs_share"] = (
                abs(out["alignment_bias"]) / out["alignment_rmse"] if out["alignment_rmse"] else np.nan
            )
        fb = np.asarray(d["dip_prediction"][sl], dtype="float64")
        fbe = fb - truth
        if np.isfinite(fbe).any():
            out["dip_fallback_rmse"] = float(np.sqrt(np.nanmean(fbe[np.isfinite(fbe)] ** 2)))
        anchor = model._anchor(inp)
        he = anchor - truth
        out["hold_last_rmse"] = float(np.sqrt(np.nanmean(he[np.isfinite(he)] ** 2)))
        out["truth_move_max"] = float(np.nanmax(np.abs(truth - truth[0]))) if truth.size else np.nan
    return out


def summarize(wells: pd.DataFrame) -> pd.DataFrame:
    """Per-protocol answers to the numbered questions."""
    if wells.empty:
        return pd.DataFrame()

    def q(name, question, fn):
        rows = []
        for protocol, g in wells.groupby("protocol", sort=False):
            try:
                value, detail = fn(g)
            except Exception as exc:  # a diagnostic must not kill the report
                value, detail = np.nan, f"unavailable: {type(exc).__name__}"
            rows.append(
                {"protocol": protocol, "question": name, "measures": question,
                 "value": value, "detail": detail}
            )
        return rows

    def frac(series) -> float:
        s = pd.to_numeric(series, errors="coerce")
        return float(np.nanmean(s.astype(float))) if len(s) else np.nan

    rows: list[dict] = []
    rows += q("Q1", "median std(Z) / std(TVT) over the visible prefix",
              lambda g: (float(np.nanmedian(g["z_over_tvt_scale"])),
                         f"median non-planar Z residual = {float(np.nanmedian(g['z_nonplanar_sigma_ft'])):.2f} ft, "
                         f"i.e. {float(np.nanmedian(g['z_nonplanarity_over_tvt_std'])):.2f}x the TVT signal's own "
                         "standard deviation — this leaks into TVT one-for-one"))
    rows += q("Q1b", "median empirical dTVT/dZ over the prefix (the code hard-codes -1)",
              lambda g: (float(np.nanmedian(g.get("empirical_dtvt_dz"))),
                         f"median |coefficient error| x Z lever arm = "
                         f"{float(np.nanmedian(g.get('implied_z_coefficient_error_ft'))):.2f} ft; "
                         f"median Z movement across the predicted region = "
                         f"{float(np.nanmedian(g['z_step_from_anchor_ft'])):.1f} ft"))
    rows += q("Q2", "share of wells where TVT+Z is the flatter surface",
              lambda g: (frac(g["convention_flatter"] == "TVT+Z"),
                         f"median |R2 margin| = {float(np.nanmedian(np.abs(g['convention_r2_margin']))):.4f} "
                         "(small margin means the plane absorbs either sign)"))
    rows += q("Q3", "share of wells whose projected gradient matches the observed prefix sign",
              lambda g: (frac(g.get("gradient_sign_agrees")),
                         f"median gradient ratio = {float(np.nanmedian(g.get('gradient_ratio'))):.3f}"))
    rows += q("Q4", "median cross-track/along-track singular ratio of the visible prefix",
              lambda g: (float(np.nanmedian(g["xy_singular_ratio"])),
                         f"median perpendicular span = {float(np.nanmedian(g['perp_span_ft'])):.0f} ft; "
                         f"median unidentified-dip TVT error = "
                         f"{float(np.nanmedian(g['unidentified_dip_tvt_ft'])):.3f} ft"))
    rows += q("Q4b", "median hidden-region extrapolation, as a multiple of the fitted span",
              lambda g: (float(np.nanmedian(g["extrapolation_ratio"])), ""))
    rows += q("Q5", "share of wells whose GR amplitude gain was clipped to [0.2, 5.0]",
              lambda g: (frac(g.get("gr_gain_clipped")),
                         f"median gain = {float(np.nanmedian(g.get('gr_gain_raw'))):.3f}; "
                         f"median split-half ratio = {float(np.nanmedian(g.get('gr_gain_halves_ratio'))):.3f}"))
    rows += q("Q6", "median GR cycles traversed per alignment window",
              lambda g: (float(np.nanmedian(g.get("cycles_per_window"))),
                         f"median typewell GR wavelength = "
                         f"{float(np.nanmedian(g.get('typewell_gr_wavelength_ft'))):.1f} ft TVT"))
    rows += q("Q7", "median TVT traversed per alignment window, in feet",
              lambda g: (float(np.nanmedian(g["dtvt_per_window"])),
                         f"search half-width = {float(np.nanmedian(g['search_ft'])):.1f} ft, i.e. the "
                         "search range exceeds the section a window actually crosses"))
    rows += q("Q8", "mean fallback fraction over predicted rows",
              lambda g: (float(np.nanmean(g["fallback_fraction"])),
                         f"mean confidence = {float(np.nanmean(g['confidence_mean'])):.4f}; "
                         f"alignment failed outright on {int((~g['alignment_ok'].astype(bool)).sum())} wells"))
    rows += q("Q9", "median |bias| as a share of that well's RMSE",
              lambda g: (float(np.nanmedian(g.get("alignment_bias_abs_share"))),
                         f"mean signed bias = {float(np.nanmean(g.get('alignment_bias'))):.3f} ft"))
    rows += q("Q9b", "median typewell-clip fraction of the prediction",
              lambda g: (float(np.nanmedian(g["typewell_clip_fraction"])),
                         "a high value means the prediction is pinned to the reference-section edge"))
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protocol", default=f"{PROTOCOL_A},{PROTOCOL_B}")
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    verbose = not args.quiet
    if args.reports_dir:
        os.environ["ROGII_REPORTS_DIR"] = args.reports_dir
        import importlib

        import src.paths

        importlib.reload(src.paths)
        reports_dir = src.paths.ensure_reports_dir()
    else:
        reports_dir = ensure_reports_dir()
    require_competition_data()

    protocols = [p.strip() for p in args.protocol.split(",") if p.strip()]
    bad = [p for p in protocols if p not in MODES]
    if bad:
        raise SystemExit(f"unknown protocol(s): {bad}; known: {list(MODES)}")

    files = discover_wells("train")
    ids = filter_blocked(sorted(files))
    if args.max_wells:
        rng = np.random.default_rng(args.seed)
        pick = rng.permutation(len(ids))[: args.max_wells]
        ids = sorted(ids[i] for i in pick)

    rows = []
    for protocol in protocols:
        mode = MODES[protocol]
        if verbose:
            print(f"[{protocol}] diagnosing {len(ids)} wells ...")
        for wid in ids:
            try:
                task = make_task(load_well(files[wid]), mode)
            except (TaskConstructionError, OSError, ValueError, KeyError):
                continue
            try:
                row = diagnose_well(task, protocol)
            except Exception as exc:
                rows.append({"protocol": protocol, "well_id": wid,
                             "diagnostic_error": f"{type(exc).__name__}: {exc}"})
                continue
            if row is not None:
                rows.append(row)

    wells = pd.DataFrame(rows)
    if wells.empty:
        raise SystemExit("No wells produced diagnostics.")
    wells_out = reports_dir / "dip_alignment_diagnostics_wells.csv"
    wells.to_csv(wells_out, index=False)
    summary = summarize(wells[wells.get("diagnostic_error").isna()] if "diagnostic_error" in wells else wells)
    summary_out = reports_dir / "dip_alignment_diagnostics_summary.csv"
    summary.to_csv(summary_out, index=False)

    if verbose:
        print()
        print(summary.to_string(index=False))
        print(f"\nWritten:\n  {wells_out}\n  {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
