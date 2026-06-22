#!/usr/bin/env python
"""Statistical comparison of Condition A vs B feature tables (experiment plan §5A).

Reads the per-episode feature CSVs from feature_extraction.py (one for A, one for B) and, for
each feature, tests whether the camera condition makes a difference:

  continuous features (speed / smoothness / accuracy)
      -> Welch independent-samples t-test (A/B are independently captured, not paired)
      -> Cohen's d effect size (how big the difference is, not just whether it's significant)
      -> Benjamini-Hochberg FDR correction across all features (controls false positives from
         testing many features at once)
  success / failure (binary)
      -> Fisher's exact test on the success counts (a proportion, not a continuous metric)
  task difficulty (if a `difficulty` column is present)
      -> one-way ANOVA across simple/medium/hard for each feature (plan's multi-difficulty test)

Output: a printed table + comparison.csv. (Excel export will be added separately.)

USAGE (run from the inner lerobot-main project dir):
    uv run --extra training python experiments/camera_ablation/features_compare.py \
        --features_a outputs/features_A.csv --features_b outputs/features_B.csv \
        --out outputs/comparison.csv
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats

# Columns that are identifiers/labels, not continuous features to test.
NON_FEATURE = {"episode", "n_frames", "success", "failure_type", "difficulty", "condition"}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference (pooled SD). |d|: 0.2 small, 0.5 medium, 0.8 large."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    pooled_sd = np.sqrt(pooled_var)
    return float((a.mean() - b.mean()) / pooled_sd) if pooled_sd > 0 else 0.0


def effect_label(d: float) -> str:
    ad = abs(d)
    if np.isnan(d):
        return "n/a"
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def continuous_comparison(df_a: pd.DataFrame, df_b: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    rows = []
    for f in feats:
        a = df_a[f].dropna().to_numpy(dtype=float)
        b = df_b[f].dropna().to_numpy(dtype=float)
        if len(a) < 2 or len(b) < 2:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)  # Welch
        d = cohens_d(a, b)
        rows.append({
            "feature": f,
            "A_mean": a.mean(), "A_std": a.std(ddof=1),
            "B_mean": b.mean(), "B_std": b.std(ddof=1),
            "diff_A_minus_B": a.mean() - b.mean(),
            "t": t, "p": p,
            "cohens_d": d, "effect": effect_label(d),
        })
    out = pd.DataFrame(rows)
    if len(out):
        # Benjamini-Hochberg FDR across all features tested together.
        out["p_fdr"] = stats.false_discovery_control(out["p"].to_numpy(), method="bh")
        out["significant"] = out["p_fdr"] < 0.05
        out = out.sort_values("p_fdr").reset_index(drop=True)
    return out


def success_comparison(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict | None:
    if "success" not in df_a.columns or "success" not in df_b.columns:
        return None
    sa, sb = df_a["success"].dropna(), df_b["success"].dropna()
    succ_a, n_a = int(sa.sum()), len(sa)
    succ_b, n_b = int(sb.sum()), len(sb)
    table = [[succ_a, n_a - succ_a], [succ_b, n_b - succ_b]]
    _, p = stats.fisher_exact(table)
    return {"SR_A": succ_a / n_a, "SR_B": succ_b / n_b, "n_a": n_a, "n_b": n_b, "p": p}


def difficulty_anova(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    if "difficulty" not in df.columns:
        return pd.DataFrame()
    groups_by = {g: sub for g, sub in df.groupby("difficulty")}
    if len(groups_by) < 2:
        return pd.DataFrame()
    rows = []
    for f in feats:
        samples = [sub[f].dropna().to_numpy(dtype=float) for sub in groups_by.values()]
        samples = [s for s in samples if len(s) >= 2]
        if len(samples) < 2:
            continue
        F, p = stats.f_oneway(*samples)
        rows.append({"feature": f, "F": F, "p": p, "levels": "/".join(map(str, groups_by))})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_fdr"] = stats.false_discovery_control(out["p"].to_numpy(), method="bh")
        out["significant"] = out["p_fdr"] < 0.05
    return out


def export_excel(path: str, comp: pd.DataFrame, sr: dict | None, anova: pd.DataFrame) -> None:
    """Write a multi-sheet .xlsx with significance/effect-size highlights."""
    from openpyxl.styles import Font, PatternFill

    sig_fill = PatternFill("solid", fgColor="C6EFCE")   # green: significant (p_fdr < 0.05)
    eff_fill = PatternFill("solid", fgColor="FFEB9C")   # yellow: large effect (|d| >= 0.8)
    bold = Font(bold=True)

    def _autosize(ws):
        for col in ws.columns:
            w = max((len(str(c.value)) for c in col if c.value is not None), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(w + 2, 32)

    def _highlight(ws, sig_col_name: str, extra_col: str | None = None):
        headers = {c.value: c.column for c in ws[1]}
        sig_c = headers.get(sig_col_name)
        extra_c = headers.get(extra_col) if extra_col else None
        for r in range(2, ws.max_row + 1):
            if sig_c and ws.cell(r, sig_c).value in (True, "True"):
                for c in range(1, ws.max_column + 1):
                    ws.cell(r, c).fill = sig_fill
            if extra_c:
                v = ws.cell(r, extra_c).value
                if isinstance(v, (int, float)) and abs(v) >= 0.8:
                    ws.cell(r, extra_c).fill = eff_fill
                    ws.cell(r, extra_c).font = bold
        _autosize(ws)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        comp.to_excel(writer, sheet_name="连续特征对比", index=False)
        if sr:
            pd.DataFrame([sr]).to_excel(writer, sheet_name="成功率", index=False)
        if len(anova):
            anova.to_excel(writer, sheet_name="多难度ANOVA", index=False)

        if len(comp):
            _highlight(writer.sheets["连续特征对比"], "significant", extra_col="cohens_d")
        if "成功率" in writer.sheets:
            _autosize(writer.sheets["成功率"])
        if len(anova):
            _highlight(writer.sheets["多难度ANOVA"], "significant")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features_a", required=True, help="Condition A feature CSV.")
    parser.add_argument("--features_b", required=True, help="Condition B feature CSV.")
    parser.add_argument("--out", default="outputs/comparison.csv", help="Output comparison CSV.")
    parser.add_argument("--excel", default=None,
                        help="Also write a multi-sheet .xlsx (significance/effect highlights).")
    args = parser.parse_args()

    df_a = pd.read_csv(args.features_a)
    df_b = pd.read_csv(args.features_b)

    feats = [c for c in df_a.columns
             if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(df_a[c]) and c in df_b.columns]
    print(f"Comparing {len(feats)} features | A: n={len(df_a)}, B: n={len(df_b)}")

    # 1) Continuous features: Welch t-test + Cohen's d + FDR
    comp = continuous_comparison(df_a, df_b, feats)
    print("\n=== Continuous features (Welch t-test, FDR-corrected) ===")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(comp.round(4).to_string(index=False) if len(comp) else "  (no testable features)")

    # 2) Success / failure rate: Fisher exact
    sr = success_comparison(df_a, df_b)
    if sr:
        print("\n=== Success rate (Fisher exact) ===")
        print(f"  SR_A={sr['SR_A']:.1%} (n={sr['n_a']})   SR_B={sr['SR_B']:.1%} (n={sr['n_b']})   "
              f"p={sr['p']:.4g}   {'significant' if sr['p'] < 0.05 else 'n.s.'}")

    # 3) Multi-difficulty ANOVA (pooled A+B)
    anova = difficulty_anova(pd.concat([df_a, df_b], ignore_index=True), feats)
    if len(anova):
        print("\n=== Multi-difficulty one-way ANOVA (pooled, FDR-corrected) ===")
        with pd.option_context("display.max_columns", None, "display.width", 240):
            print(anova.round(4).to_string(index=False))

    comp.to_csv(args.out, index=False)
    if len(anova):
        anova.to_csv(args.out.replace(".csv", "_anova.csv"), index=False)
    print(f"\nWrote comparison to {args.out}")

    if args.excel:
        export_excel(args.excel, comp, sr, anova)
        print(f"Wrote Excel workbook to {args.excel}")


if __name__ == "__main__":
    main()
