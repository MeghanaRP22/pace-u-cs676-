"""
evaluate.py — measure how good the scorer is.

    python evaluate.py                    # URL model only (fast, free, deterministic)
    python evaluate.py --extended         # also score the 24-URL extended held-out set
    python evaluate.py --baseline         # the original course baseline, for before/after
    python evaluate.py --variant structural   # ablation: model trained without published ratings
    python evaluate.py --net              # add page + OpenAlex/Crossref evidence (needs internet)
    python evaluate.py --llm              # add the Claude layer (needs ANTHROPIC_API_KEY)
    python evaluate.py --llm --tune-blend # re-estimate the LLM sigma used for precision weighting

Prints a per-URL table and summary numbers, and writes results/eval_<config>.csv.

THE LABELS ARE A STARTING POINT, NOT GROUND TRUTH. The first 24 are the
instructor's, unchanged, so numbers stay comparable with the course README.
EXTENDED_URLS were added for this submission: 24 more URLs whose domains appear
in neither the curated table nor training_data.py, labelled with the same
rubric before the model was scored on them.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from typing import Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

import credibility
from credibility import score_band, score_url_detailed

HERE = os.path.dirname(os.path.abspath(__file__))

# (url, expected_score, why_this_label) — instructor's labels, unchanged.
IN_TABLE_URLS: List[Tuple[str, float, str]] = [
    ("https://www.nature.com/articles/s41586-021-03819-2", 0.95, "peer-reviewed journal article"),
    ("https://www.nejm.org/doi/full/10.1056/NEJMoa2034577", 0.95, "peer-reviewed medical journal"),
    ("https://pubmed.ncbi.nlm.nih.gov/33301246/", 0.90, "indexed biomedical literature"),
    ("https://www.reuters.com/world/example-report-2024-01-01/", 0.85, "wire service, corrections policy"),
    ("https://www.census.gov/data/tables/2023/demo/income-poverty.html", 0.90, "primary government statistics"),
    ("https://apnews.com/article/example-story-12345", 0.85, "wire service"),
    ("https://arxiv.org/abs/1706.03762", 0.65, "influential BUT a preprint, not peer reviewed"),
    ("https://en.wikipedia.org/wiki/Statistical_learning_theory", 0.60, "well-sourced but tertiary and open-edit"),
    ("https://www.biorxiv.org/content/10.1101/2020.01.01.000001v1", 0.50, "preprint, no review"),
    ("https://scikit-learn.org/stable/modules/linear_model.html", 0.75, "authoritative docs for its own library"),
    ("https://medium.com/@someone/why-ai-is-magic-abc123", 0.25, "self-published, no editorial review"),
    ("https://randomblog.blogspot.com/2024/03/my-thoughts.html", 0.15, "personal blog"),
    ("https://www.reddit.com/r/MachineLearning/comments/abc123/", 0.20, "user-generated comment thread"),
    ("https://example.com/sponsored/miracle-supplement", 0.10, "sponsored commercial content"),
    ("https://www.theonion.com/study-finds-example-1849", 0.05, "satire, factually false by design"),
    ("http://totally-legit-news.xyz/shocking-truth", 0.05, "no provenance, insecure, throwaway TLD"),
]

# HELD OUT (instructor's): domains absent from the curated table.
HELD_OUT_URLS: List[Tuple[str, float, str]] = [
    ("https://www.pnas.org/doi/10.1073/pnas.2020123118", 0.92, "peer-reviewed academy journal"),
    ("https://jamanetwork.com/journals/jama/fullarticle/2762130", 0.93, "peer-reviewed medical journal"),
    ("https://www.who.int/news-room/fact-sheets/detail/example", 0.88, "international health authority"),
    ("https://www.propublica.org/article/example-investigation", 0.85, "investigative newsroom, fact-checked"),
    ("https://www.imf.org/en/Publications/WEO/example", 0.85, "primary economic data publisher"),
    ("https://stackoverflow.com/questions/12345/how-to-do-x", 0.45, "often correct, but unreviewed and unattributed"),
    ("https://seekingalpha.com/article/example-stock-analysis", 0.35, "contributor-submitted, light editorial review"),
    ("https://health-truth-daily.info/miracle-cure-doctors-hate", 0.05, "fabricated health claims, no provenance"),
]

LABELLED_URLS = IN_TABLE_URLS + HELD_OUT_URLS          # the original 24

# EXTENDED HELD OUT (added for this submission): not in the table, not in training.
EXTENDED_URLS: List[Tuple[str, float, str]] = [
    ("https://journals.lww.com/co-infectiousdiseases/fulltext/2021/02000/example.5.aspx", 0.88, "peer-reviewed journal"),
    ("https://www.cambridge.org/core/journals/psychological-medicine/article/example/ABC123", 0.90, "peer-reviewed journal"),
    ("https://www.techrxiv.org/articles/preprint/example/12345", 0.50, "preprint, no review"),
    ("https://www.nhs.uk/conditions/type-2-diabetes/", 0.88, "national health service guidance"),
    ("https://www.ecdc.europa.eu/en/publications-data/example-report", 0.88, "EU public-health agency report"),
    ("https://www.fao.org/publications/card/en/c/example", 0.85, "UN agency publication"),
    ("https://www.bea.gov/data/gdp/gross-domestic-product", 0.90, "primary government statistics"),
    ("https://www.nps.gov/yell/learn/nature/wolves.htm", 0.85, "government agency information page"),
    ("https://www.ipcc.ch/report/ar6/wg1/", 0.92, "intergovernmental scientific assessment"),
    ("https://www.aljazeera.com/news/2024/1/1/example-story", 0.75, "major international newsroom"),
    ("https://www.cnn.com/2024/01/01/politics/example/index.html", 0.75, "major newsroom"),
    ("https://www.snopes.com/fact-check/example-claim/", 0.80, "established fact-checker"),
    ("https://www.vice.com/en/article/example-story", 0.60, "digital outlet, mixed record"),
    ("https://www.verywellhealth.com/example-condition-overview", 0.60, "commercial health explainer, medically reviewed"),
    ("https://askubuntu.com/questions/12345/how-to-install-x", 0.45, "community Q&A"),
    ("https://www.cs.princeton.edu/~someone/teaching/notes.html", 0.55, "personal page on academic server"),
    ("https://www.forbes.com/sites/someone/2024/01/01/example-take/", 0.45, "contributor network, light editing"),
    ("https://someone.github.io/posts/my-take-on-transformers/", 0.25, "personal blog"),
    ("https://www.prweb.com/releases/example/prweb12345.htm", 0.25, "press release"),
    ("https://www.tripadvisor.com/ShowUserReviews-g1-d2-r3-Review.html", 0.15, "anonymous user review"),
    ("https://www.naturalhealth365.com/miracle-herb-cures-everything.html", 0.10, "health misinformation site"),
    ("http://cancer-cure-secrets.net/garlic-destroys-tumors", 0.03, "fabricated health claims"),
    ("https://breaking-news-now-24.top/celebrity-shocking-reveal", 0.05, "throwaway clickbait domain"),
    ("https://abcnews.com.co/obama-signs-order-example/", 0.02, "look-alike impersonation site"),
]

ALL_EVAL_URLS = LABELLED_URLS + EXTENDED_URLS


def _baseline_scorer():
    """Load the untouched course baseline from baseline/ for before/after runs."""
    path = os.path.join(HERE, "baseline", "credibility_baseline.py")
    spec = importlib.util.spec_from_file_location("credibility_baseline", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["credibility_baseline"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def run(urls: List[Tuple[str, float, str]], blocks: Dict[str, List[str]], use_llm: bool,
        use_net: bool, baseline: bool, tag: str, quiet: bool = False) -> Dict[str, float]:
    """Score every labelled URL and report error against the expected values."""
    base = _baseline_scorer() if baseline else None
    rows = []
    for url, expected, rationale in urls:
        if base is not None:
            r = base.score_url(url, use_llm=use_llm if use_llm else False)
            got, low, high, stype = r["score"], None, None, "baseline"
        else:
            r = score_url_detailed(url, use_llm=None if use_llm else False, use_network=use_net)
            got, low, high, stype = r["score"], r["low"], r["high"], r["source_type"]
        block = next((b for b, members in blocks.items() if url in members), "")
        rows.append({"url": url, "block": block, "expected": expected, "got": got, "low": low,
                     "high": high, "error": abs(got - expected), "source_type": stype,
                     "band_ok": score_band(got)[0] == score_band(expected)[0], "why": rationale})

    errors = [r["error"] for r in rows]
    worst = max(errors)
    if not quiet:
        print(f"\n{'expected':>9} {'got':>6} {'90% int.':>11} {'err':>6}  url")
        print("-" * 104)
        for r in sorted(rows, key=lambda r: -r["error"]):
            flag = "  <-- worst" if r["error"] == worst else ""
            disp = r["url"] if len(r["url"]) <= 58 else r["url"][:55] + "..."
            interval = f"{r['low']:.2f}-{r['high']:.2f}" if r["low"] is not None else "     -     "
            print(f"{r['expected']:>9.2f} {r['got']:>6.2f} {interval:>11} {r['error']:>6.2f}  {disp}{flag}")
            print(f"{'':>35}  [{r['block']}] ({r['why']}) -> {r['source_type']}")

    mae = sum(errors) / len(errors)
    band_acc = sum(r["band_ok"] for r in rows) / len(rows)
    summary = {"n": len(rows), "mae": mae, "band_accuracy": band_acc, "worst": worst}
    for b in blocks:
        errs = [r["error"] for r in rows if r["block"] == b]
        if errs:
            summary[f"mae_{b}"] = sum(errs) / len(errs)
            summary[f"band_{b}"] = sum(r["band_ok"] for r in rows if r["block"] == b) / len(errs)
    if rows[0]["low"] is not None:
        summary["coverage"] = sum(r["low"] <= r["expected"] <= r["high"] for r in rows) / len(rows)
        summary["mean_width"] = sum(r["high"] - r["low"] for r in rows) / len(rows)
    summary["brier_high"] = sum((r["got"] - (r["expected"] >= 0.7)) ** 2 for r in rows) / len(rows)

    if not quiet:
        print("-" * 104)
        print(f"  URLs evaluated     : {len(rows)}")
        print(f"  Mean absolute error: {mae:.3f}   (lower is better; 0.000 is perfect)")
        print(f"  Band accuracy      : {band_acc:.1%}   (HIGH/MEDIUM/LOW chip correct)")
        print(f"  Worst single error : {worst:.3f}")
        for b in blocks:
            if f"mae_{b}" in summary:
                print(f"    {b:<18}: MAE {summary[f'mae_{b}']:.3f}, band {summary[f'band_{b}']:.1%}")
        if "coverage" in summary:
            print(f"  90% interval cover : {summary['coverage']:.1%}   (mean width {summary['mean_width']:.2f})")
        print(f"  Brier (HIGH event) : {summary['brier_high']:.3f}")
        print(f"  Scorer             : {'course baseline' if baseline else 'new model (' + credibility.MODEL_VARIANT + ')'}")
        print(f"  Evidence layer     : {'on (page + OpenAlex/Crossref)' if use_net else 'off'}")
        print(f"  LLM layer          : {'on (' + credibility.JUDGE_MODEL + ')' if use_llm else 'off (rules only)'}\n")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"eval_{tag}.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    return summary


def tune_blend(urls, blocks, use_net: bool) -> None:
    """Grid-search the LLM sigma (precision weighting) using cached LLM answers."""
    print("Tuning LLM_SIGMA_RECOGNIZED on the evaluation set (LLM answers cached to disk).")
    print("Caution: tuning on the evaluation set flatters the result — report it as such.\n")
    best: Optional[Tuple[float, float]] = None
    for sigma in (0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0):
        credibility.LLM_SIGMA_RECOGNIZED = sigma
        credibility._CACHE.clear()
        s = run(urls, blocks, True, use_net, False, f"tune_{sigma}", quiet=True)
        print(f"  sigma={sigma:<6} MAE={s['mae']:.3f}  band={s['band_accuracy']:.1%}  worst={s['worst']:.3f}")
        if best is None or s["mae"] < best[1]:
            best = (sigma, s["mae"])
    print(f"\n  best sigma = {best[0]} (MAE {best[1]:.3f}). Set LLM_SIGMA_RECOGNIZED in credibility.py.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate credibility.score_url against labelled URLs.")
    parser.add_argument("--llm", action="store_true", help="include the Claude judgment layer (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--net", action="store_true", help="include page and scholarly-metadata evidence (needs internet)")
    parser.add_argument("--extended", action="store_true", help="also score the 24 extended held-out URLs")
    parser.add_argument("--baseline", action="store_true", help="score with the original course baseline")
    parser.add_argument("--variant", choices=["full", "structural"], default=None,
                        help="which fitted URL model to use (default: full)")
    parser.add_argument("--tune-blend", action="store_true", help="with --llm: grid-search the LLM sigma")
    args = parser.parse_args()

    if args.llm and not os.getenv("ANTHROPIC_API_KEY") and not credibility.LLM_CACHE_PATH:
        print(
            "\n  ERROR: --llm was requested but ANTHROPIC_API_KEY is not set.\n"
            "\n  Put ANTHROPIC_API_KEY=sk-... in .env (copy .env.example), or export it,\n"
            "  then re-run. To measure without the LLM layer, drop --llm.\n",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if args.variant:
        credibility.MODEL_VARIANT = args.variant
    if args.llm and not credibility.LLM_CACHE_PATH:
        credibility.LLM_CACHE_PATH = os.path.join(HERE, "results", "llm_cache.json")
        os.makedirs(os.path.dirname(credibility.LLM_CACHE_PATH), exist_ok=True)

    urls = ALL_EVAL_URLS if args.extended else LABELLED_URLS
    blocks = {"in-table": [u for u, _, _ in IN_TABLE_URLS],
              "held-out": [u for u, _, _ in HELD_OUT_URLS],
              "extended": [u for u, _, _ in EXTENDED_URLS]}
    tag = "_".join(filter(None, ["baseline" if args.baseline else credibility.MODEL_VARIANT,
                                 "net" if args.net else "", "llm" if args.llm else "",
                                 "ext" if args.extended else ""]))
    if args.tune_blend:
        if not args.llm:
            parser.error("--tune-blend needs --llm")
        tune_blend(urls, blocks, args.net)
    else:
        run(urls, blocks, args.llm, args.net, args.baseline, tag)
