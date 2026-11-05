#!/usr/bin/env python3
"""
Diagnostic analysis of the NLQ x RAG-tier grid run by films_analyzer_inf.py (claude-sonnet-4-6, S1 flat_conjunctive,
--max_new_tokens 3000, --max_queries 2).

Inputs (all read from disk, nothing invented):
  claude_code_film_strategies/*/<cell>.json             non-RAG result records
  rag_claude_code_film_strategies/*/<cell>.json         RAG result records
  <cell dir>/raw_calls/*_q<n>_try1_events.jsonl         raw Claude Code stream: per-API-attempt stop_reason/output_tokens
  */analysis/eval_per_record.csv                        ground-truth scoring produced by the script's own analysis stage
  run_stdout.log                                        per-invocation exit codes

Outputs (diagnostic_analysis/):
  per_call_diagnostics.csv, cell_summary.csv, invocation_exit_codes.csv, diagnostic_report.txt, fig_*.png
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
ROOTS = [REPO / "claude_code_film_strategies", REPO / "rag_claude_code_film_strategies"]
MODES = ["nlq_general", "nlq_wikipedia", "nlq_cot_general", "nlq_cot_wikipedia"]
TIERS = ["no_evidence", "year_list_only", "titles_authors", "titles_authors_prize",
         "full_with_birthplace", "full_with_population", "full_with_distractors"]
TIER_LABEL = {"no_evidence": "RAG0\nnone", "year_list_only": "RAG1\ntitles", "titles_authors": "RAG2\ndirectors",
              "titles_authors_prize": "RAG3\naward", "full_with_birthplace": "RAG4\nbirthplace",
              "full_with_population": "RAG5\npopulation", "full_with_distractors": "RAG6\ndistractors"}
CLI_CAP_MSG = "exceeded the"   # text of the Claude Code CLI error returned instead of an answer

# palette (reference instance, light mode)
BLUE, ORANGE, AQUA, GREY, INK, INK2, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GREY, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.6, "axes.axisbelow": True})


def attempts_from_events(path: Path):
    """One entry per API message in the stream: (stop_reason, output_tokens) taken from message_delta events."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        e = ev.get("event") if ev.get("type") == "stream_event" else None
        if e and e.get("type") == "message_delta":
            out.append((e.get("delta", {}).get("stop_reason"), (e.get("usage") or {}).get("output_tokens")))
    return out


def load_records() -> pd.DataFrame:
    rows = []
    for root in ROOTS:
        for jf in sorted(root.glob("*/*.json")):
            if jf.parent.name in ("analysis", "raw_calls"):
                continue
            d = json.loads(jf.read_text(encoding="utf-8"))
            for r in d.get("results", []):
                ev = next((Path(REPO / p) for p in r.get("raw_files", []) if p.endswith("_events.jsonl")), None)
                att = attempts_from_events(ev) if ev else []
                text = str(r.get("Generated_Response", ""))
                u = r.get("usage") or {}
                rows.append({
                    "mode": r["mode"], "rag_tier": r["rag_tier"], "Query_ID": r["Query_ID"], "Film_Title": r["Film_Title"],
                    "release_year": r.get("Source_ReleaseYear"), "rag_rows_injected": r.get("rag_rows_injected"),
                    "est_prompt_tokens": r.get("est_prompt_tokens"), "script_status": r.get("generation_status"),
                    "cli_cap_error": CLI_CAP_MSG in text and text.startswith("API Error"),
                    "answered": not (CLI_CAP_MSG in text and text.startswith("API Error")),
                    "api_attempts": len(att), "attempts_max_tokens": sum(1 for s, _ in att if s == "max_tokens"),
                    "final_stop_reason": att[-1][0] if att else None,
                    "output_tokens_total": u.get("output_tokens"),
                    "thinking_tokens": (u.get("output_tokens_details") or {}).get("thinking_tokens"),
                    "input_tokens_total": (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
                                          + (u.get("cache_read_input_tokens") or 0),
                    "text_chars": r.get("text_chars"), "latency_s": r.get("generation_time_seconds"),
                    "cost_usd": r.get("cost_usd_reported"), "n_directors_parsed":
                        sum(1 for x in (r.get("Parsed_Rows") or []) if x.get("Director") not in ("", "NO RESULT")),
                })
    return pd.DataFrame(rows)


def load_gt() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in (r / "analysis" / "eval_per_record.csv" for r in ROOTS) if p.exists()]
    if not frames:
        return pd.DataFrame()
    g = pd.concat(frames, ignore_index=True)
    keep = ["mode", "rag_tier", "Query_ID", "tp", "n_pred", "n_gt", "birth_ok", "birth_n", "pop_5_n", "pop_cmp_n", "year_ok"]
    return g[[c for c in keep if c in g.columns]].drop_duplicates(["mode", "rag_tier", "Query_ID"], keep="last")


def load_exit_codes() -> pd.DataFrame:
    log = (REPO / "run_stdout.log").read_text(encoding="utf-8", errors="replace")
    pat = re.compile(r"^########## (FOLLOWUP )?END mode=(\S+) rag_tier=(\S+) exit=(\d+) end=(\S+)", re.M)
    rows = [{"invocation": "followup" if m.group(1) else "grid", "mode": m.group(2), "rag_tier": m.group(3),
             "exit_code": int(m.group(4)), "end": m.group(5)} for m in pat.finditer(log)]
    return pd.DataFrame(rows)


def main():
    df = load_records()
    gt = load_gt()
    if not gt.empty:
        df = df.merge(gt, on=["mode", "rag_tier", "Query_ID"], how="left")
        df["precision"] = np.where(df["n_pred"] > 0, df["tp"] / df["n_pred"].replace(0, np.nan), np.nan)
        df["recall"] = df["tp"] / df["n_gt"].replace(0, np.nan)
    df["mode"] = pd.Categorical(df["mode"], MODES, ordered=True)
    df["rag_tier"] = pd.Categorical(df["rag_tier"], TIERS, ordered=True)
    df = df.sort_values(["mode", "rag_tier", "Query_ID"]).reset_index(drop=True)
    ex = load_exit_codes()
    df.to_csv(OUT / "per_call_diagnostics.csv", index=False)
    ex.to_csv(OUT / "invocation_exit_codes.csv", index=False)

    cell = df.groupby(["mode", "rag_tier"], observed=True).agg(
        calls=("Query_ID", "size"), answered=("answered", "sum"), cli_cap_errors=("cli_cap_error", "sum"),
        mean_api_attempts=("api_attempts", "mean"), mean_output_tokens=("output_tokens_total", "mean"),
        mean_thinking_tokens=("thinking_tokens", "mean"), mean_latency_s=("latency_s", "mean"),
        cost_usd=("cost_usd", "sum"), rag_rows=("rag_rows_injected", "mean")).reset_index()
    cell.to_csv(OUT / "cell_summary.csv", index=False)

    # ---- Fig 1: outcome matrix (answered calls / calls) per mode x tier ----
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.grid(False)
    for i, m in enumerate(MODES):
        for j, t in enumerate(TIERS):
            sub = df[(df["mode"] == m) & (df["rag_tier"] == t)]
            e = ex[(ex["mode"] == m) & (ex["rag_tier"] == t)]
            if len(sub):
                a = int(sub["answered"].sum())
                col, lab = (AQUA if a == len(sub) else BLUE if a else ORANGE), f"{a}/{len(sub)} answered"
            elif len(e) and (e["exit_code"] != 0).all():
                col, lab = GREY, "startup fail\n(no ref. file)"
            else:
                col, lab = "#efeee9", "not run yet"
            ax.add_patch(plt.Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, color=col, lw=0))
            ax.text(j + 0.5, i + 0.5, lab, ha="center", va="center", fontsize=8,
                    color="white" if col in (ORANGE, BLUE, AQUA) else INK2)
    ax.set_xlim(0, len(TIERS)); ax.set_ylim(len(MODES), 0)
    ax.set_xticks(np.arange(len(TIERS)) + 0.5, [TIER_LABEL[t] for t in TIERS], fontsize=8)
    ax.set_yticks(np.arange(len(MODES)) + 0.5, MODES)
    ax.set_title("Calls that returned an answer (vs. Claude Code 'exceeded 3000 output token' error), per cell",
                 loc="left", fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig1_outcome_matrix.png", dpi=150); plt.close(fig)

    # ---- Fig 2: per-call API attempts that hit max_tokens vs. completed ----
    lab = [f"{m.replace('nlq_', '')}|{TIER_LABEL[t].split(chr(10))[0]}|q{q}" for m, t, q in
           zip(df["mode"], df["rag_tier"], df["Query_ID"])]
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(df) + 1.2))
    capped = df["attempts_max_tokens"].to_numpy()
    done = (df["api_attempts"] - df["attempts_max_tokens"]).to_numpy()
    ax.barh(y, capped, color=ORANGE, height=0.62, label="API attempt stopped at max_tokens (3000)")
    ax.barh(y, done, left=capped, color=AQUA, height=0.62, label="API attempt completed (end_turn)")
    ax.set_yticks(y, lab, fontsize=7); ax.invert_yaxis(); ax.set_xlabel("API attempts inside one `claude -p` call")
    ax.set_title("Claude Code internal retries per call", loc="left", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig2_attempts_per_call.png", dpi=150); plt.close(fig)

    # ---- Fig 3: output tokens billed vs visible text, per call ----
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(df) + 1.2))
    ax.barh(y, df["output_tokens_total"], color=BLUE, height=0.62, label="output tokens billed (thinking + text)")
    ax.axvline(3000, color=INK2, lw=1, ls="--"); ax.text(3050, -0.8, "per-attempt cap 3000", fontsize=7, color=INK2)
    for yi, (tok, tc) in enumerate(zip(df["output_tokens_total"], df["text_chars"])):
        ax.text(tok + 100, yi, f"{tc} text chars", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(y, lab, fontsize=7); ax.invert_yaxis(); ax.set_xlabel("output tokens per call")
    ax.set_title("Output tokens spent per call vs. visible text returned", loc="left", fontsize=10)
    ax.set_xlim(0, max(df["output_tokens_total"].max() * 1.18, 3500))
    fig.tight_layout(); fig.savefig(OUT / "fig3_tokens_per_call.png", dpi=150); plt.close(fig)

    # ---- Fig 4: cost and latency by RAG tier (pooled over modes), two panels, one axis each ----
    bt = df.groupby("rag_tier", observed=True).agg(cost=("cost_usd", "mean"), lat=("latency_s", "mean"),
                                                   rows=("rag_rows_injected", "mean")).reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    xl = [TIER_LABEL[t] for t in bt["rag_tier"]]
    axs[0].bar(xl, bt["cost"], color=BLUE, width=0.6); axs[0].set_title("Mean reported cost per call (USD)", loc="left", fontsize=10)
    axs[1].bar(xl, bt["lat"], color=BLUE, width=0.6); axs[1].set_title("Mean latency per call (s)", loc="left", fontsize=10)
    for a in axs:
        a.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig4_cost_latency_by_tier.png", dpi=150); plt.close(fig)

    # ---- Report ----
    n, a = len(df), int(df["answered"].sum())
    lines = ["NLQ x RAG-TIER DIAGNOSTIC REPORT  (claude-sonnet-4-6, S1 flat_conjunctive, max_new_tokens=3000, 2 films/cell)",
             "=" * 110, "",
             f"Invocations logged: {len(ex)}  (exit 0: {int((ex['exit_code'] == 0).sum())}, exit !=0: {int((ex['exit_code'] != 0).sum())})",
             f"Calls recorded: {n}   answered: {a}   Claude Code output-cap error: {int(df['cli_cap_error'].sum())}",
             f"Output tokens billed: {int(df['output_tokens_total'].sum())}   of which thinking: {int(df['thinking_tokens'].fillna(0).sum())}",
             f"Total reported cost: ${df['cost_usd'].sum():.4f}   mean latency/call: {df['latency_s'].mean():.1f}s", "",
             "INVOCATION EXIT CODES", ex.to_string(index=False), "",
             "CELL SUMMARY", cell.to_string(index=False, float_format=lambda v: f"{v:.3f}"), "",
             "PER CALL", df[["mode", "rag_tier", "Query_ID", "Film_Title", "answered", "api_attempts", "attempts_max_tokens",
                             "final_stop_reason", "output_tokens_total", "thinking_tokens", "text_chars", "latency_s",
                             "cost_usd", "n_directors_parsed"]].to_string(index=False), ""]
    if "tp" in df.columns:
        sc = df[df["answered"]]
        lines += ["GROUND-TRUTH SCORING OF ANSWERED CALLS (tp / n_pred / n_gt from the script's eval_per_record.csv)",
                  sc[["mode", "rag_tier", "Query_ID", "Film_Title", "tp", "n_pred", "n_gt", "precision", "recall",
                      "birth_ok", "birth_n", "pop_5_n", "pop_cmp_n", "year_ok"]].to_string(index=False), ""]
    (OUT / "diagnostic_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
