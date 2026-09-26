"""
test_credibility.py — contract tests for score_url().

    python test_credibility.py

These check the SHAPE of your output, not its quality. They must keep passing
however much you rewrite the internals — the app, the grader, and evaluate.py
all rely on this contract. Use `evaluate.py` to measure quality.

Deliverable 1 asks for "initial testing to validate input/output handling".
This file is that, and adding your own cases here is part of the deliverable.

No pytest required, deliberately — one less thing to install.
"""

import os

# Contract tests must be deterministic and must not depend on the internet:
# score from the URL alone. Network behaviour is tested below with fakes.
os.environ["CREDIBILITY_NETWORK"] = "0"

import credibility  # noqa: E402
from credibility import score_band, score_url, score_url_detailed  # noqa: E402

PASSED = 0
FAILED = 0


def check(condition: bool, description: str) -> None:
    """Assert-with-a-label so one failure doesn't stop the whole run."""
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {description}")
    else:
        FAILED += 1
        print(f"  FAIL  {description}")


print("\nContract: return shape")
result = score_url("https://www.nature.com/articles/example", use_llm=False)
check(isinstance(result, dict), "returns a dict")
check(set(result.keys()) == {"score", "explanation"}, "has exactly the keys 'score' and 'explanation'")
check(isinstance(result["score"], float), "score is a float")
check(isinstance(result["explanation"], str), "explanation is a str")
check(len(result["explanation"]) > 0, "explanation is not empty")

print("\nContract: score range")
for url in [
    "https://www.nature.com/x",
    "https://medium.com/@a/b",
    "http://unknown-site.xyz/page",
    "https://en.wikipedia.org/wiki/X",
]:
    score = score_url(url, use_llm=False)["score"]
    check(0.0 <= score <= 1.0, f"{url[:40]:<42} -> {score:.2f} is within [0, 1]")

print("\nContract: malformed input is handled, not raised")
for bad in ["", "   ", "not a url", "ftp://files.example.com/x", "javascript:alert(1)", "//example.com"]:
    try:
        bad_result = score_url(bad, use_llm=False)
        ok = isinstance(bad_result, dict) and 0.0 <= bad_result["score"] <= 1.0
        check(ok, f"{bad!r:<28} -> {bad_result['score']:.2f} (no exception)")
    except Exception as e:
        check(False, f"{bad!r:<28} raised {type(e).__name__}")

print("\nContract: determinism")
a = score_url("https://arxiv.org/abs/1706.03762", use_llm=False)
b = score_url("https://arxiv.org/abs/1706.03762", use_llm=False)
check(a == b, "same URL scored twice gives the same result")

print("\nSanity: ordering the baseline should already get right")
journal = score_url("https://www.nature.com/articles/x", use_llm=False)["score"]
blog = score_url("https://randomblog.blogspot.com/x", use_llm=False)["score"]
check(journal > blog, f"a journal ({journal:.2f}) outranks a personal blog ({blog:.2f})")

gov = score_url("https://www.census.gov/data", use_llm=False)["score"]
throwaway = score_url("http://whatever.xyz/page", use_llm=False)["score"]
check(gov > throwaway, f"a .gov source ({gov:.2f}) outranks a throwaway domain ({throwaway:.2f})")

print("\nContract: score_band")
check(score_band(0.9)[0] == "HIGH", "0.90 -> HIGH")
check(score_band(0.5)[0] == "MEDIUM", "0.50 -> MEDIUM")
check(score_band(0.1)[0] == "LOW", "0.10 -> LOW")

# =============================================================================
# ADDED TESTS — beyond the provided contract checks
# =============================================================================
# Grouped by what they protect: odd inputs, the new model's behaviour on the
# weaknesses it claims to fix, the uncertainty interval, the explanation, and
# the network and LLM layers (exercised with fakes, so no internet or key).

print("\nAdded: odd and hostile inputs never raise")
for bad in [None, 12345, 3.14, ["https://x.com"], {"url": "x"}, b"https://nature.com",
            "https://a..b/x", "http://localhost/x", "https://" + "a" * 70 + ".com/",
            "https://example.com:99999/x", "https://" + "x" * 5000 + ".com",
            "http://[::1]:8080/x", "https://bücher.de/x", "https://xn--80ak6aa92e.com/",
            "mailto:someone@example.com", "data:text/html,<b>x</b>", "https://exa mple.com",
            "https://example.com/%zz%00\x00", "\n\thttps://www.nature.com/articles/x \n"]:
    try:
        r = score_url(bad, use_llm=False)
        ok = set(r) == {"score", "explanation"} and 0.0 <= r["score"] <= 1.0 and r["explanation"]
        check(bool(ok), f"{str(bad)[:34]!r:<38} -> {r['score']:.2f}")
    except Exception as e:  # pragma: no cover - this is the failure being tested
        check(False, f"{str(bad)[:34]!r:<38} raised {type(e).__name__}: {e}")

bare = score_url("www.nature.com/articles/x", use_llm=False)
check(bare["score"] > 0.7, f"a bare 'www.nature.com/...' is upgraded to https and scored ({bare['score']:.2f})")

print("\nAdded: held-out domains the lookup table never saw")
jama = score_url("https://jamanetwork.com/journals/jama/fullarticle/1", use_llm=False)["score"]
blogpost = score_url("https://medium.com/@x/y", use_llm=False)["score"]
check(jama > 0.6 and jama > blogpost + 0.3, f"a journal-article path ({jama:.2f}) beats a blog ({blogpost:.2f}) by a wide margin")
who = score_url("https://www.who.int/news-room/fact-sheets/detail/x", use_llm=False)["score"]
check(who >= 0.8, f".int treaty organisation scores HIGH ({who:.2f})")
junk = score_url("https://health-truth-daily.info/miracle-cure-doctors-hate", use_llm=False)["score"]
check(junk < 0.15, f"throwaway clickbait domain scores near zero ({junk:.2f})")
fake = score_url("https://abcnews.com.co/story", use_llm=False)["score"]
check(fake < 0.2, f"look-alike 'abcnews.com.co' is caught ({fake:.2f})")
ip = score_url("http://203.0.113.7/news/story.html", use_llm=False)["score"]
check(ip < 0.3, f"bare-IP host scores LOW ({ip:.2f})")

print("\nAdded: fixes for specific KNOWN WEAKNESSES")
inst = score_url("https://www.cs.cmu.edu/research/overview.html", use_llm=False)["score"]
pers = score_url("https://www.cs.cmu.edu/~someone/opinions.html", use_llm=False)["score"]
check(inst > pers, f"#5  institutional .edu ({inst:.2f}) > personal ~user page ({pers:.2f})")
news = score_url("https://www.nytimes.com/2024/01/01/us/story.html", use_llm=False)["score"]
opin = score_url("https://www.nytimes.com/2024/01/01/opinion/column.html", use_llm=False)["score"]
check(news > opin, f"#11 news report ({news:.2f}) > same outlet's opinion page ({opin:.2f})")
journal = score_url("https://www.nature.com/articles/x", use_llm=False)["score"]
preprint = score_url("https://www.biorxiv.org/content/10.1101/2020.01.01.000001v1", use_llm=False)["score"]
check(journal > preprint, f"#3  journal ({journal:.2f}) > preprint ({preprint:.2f})")
stacked = score_url("https://example.com/blog/sponsored/press-release/forum/comments/x", use_llm=False)["score"]
check(0.0 <= stacked <= 0.5, f"#12 five stacked penalties stay in range and saturate ({stacked:.2f})")

print("\nAdded: uncertainty interval (#8)")
for url in ["https://www.nature.com/articles/x", "https://unknown-site-abc.com/page",
            "https://health-truth-daily.info/x"]:
    d = score_url_detailed(url, use_llm=False)
    check(0.0 <= d["low"] <= d["score"] <= d["high"] <= 1.0,
          f"low <= score <= high for {url[:36]:<38} ({d['low']:.2f} <= {d['score']:.2f} <= {d['high']:.2f})")

print("\nAdded: explanation quality (#10)")
d = score_url_detailed("https://www.biorxiv.org/content/10.1101/2020.01.01.000001v1", use_llm=False)
check(d["explanation"].split(" (")[0] in ("Generally reliable", "Use with care", "Treat with skepticism"),
      "explanation opens with a plain-language verdict")
check("What to do:" in d["explanation"], "explanation ends with advice the reader can act on")
check("peer review" in d["explanation"].lower(), "a preprint's explanation says it lacks peer review")
check("; " not in d["explanation"], "explanation is prose, not semicolon-joined rule names")
check(d["source_type"] == "Preprint (not peer reviewed)", f"source type is labelled ({d['source_type']})")

print("\nAdded: caching returns copies, not shared state")
r1 = score_url("https://apnews.com/article/x", use_llm=False)
r1["score"] = 99.0
r2 = score_url("https://apnews.com/article/x", use_llm=False)
check(r2["score"] != 99.0, "mutating a returned result does not corrupt the cache")

print("\nAdded: fallback when model_weights.json is missing")
saved_model = credibility._MODEL
credibility._MODEL = {}
credibility._CACHE.clear()
fb = score_url("https://www.census.gov/data", use_llm=False)
check(0 <= fb["score"] <= 1 and fb["score"] >= 0.7, f"fallback scorer still works ({fb['score']:.2f})")
credibility._MODEL = saved_model
credibility._CACHE.clear()

# ---------------------------------------------------------------------------
# Network layer, tested with fakes. Each block patches the lowest-level I/O
# function, runs score_url with use_network=True, and restores the original.
# ---------------------------------------------------------------------------
print("\nAdded: network robustness (faked — no internet needed)")
REAL = {k: getattr(credibility, k) for k in ("_fetch", "_fetch_json", "_resolves", "_internet_ok")}


def with_fakes(url, fetch=None, fetch_json=None, resolves=True, internet=True):
    credibility._CACHE.clear()
    credibility._fetch = fetch or (lambda *a, **k: credibility.FetchResult(0, "timeout"))
    credibility._fetch_json = fetch_json or (lambda *a, **k: None)
    credibility._resolves = lambda *a, **k: resolves
    credibility._internet_ok = lambda: internet
    try:
        return score_url_detailed(url, use_llm=False, use_network=True)
    finally:
        for k, v in REAL.items():
            setattr(credibility, k, v)
        credibility._CACHE.clear()


offline = score_url("https://unknown-news-site.com/article/x", use_llm=False)["score"]
r = with_fakes("https://unknown-news-site.com/article/x")
check(r["score"] == offline and "timed out" in r["explanation"],
      f"timeout: same score as offline ({r['score']:.2f}) and says it timed out")
r = with_fakes("https://unknown-news-site.com/article/x", resolves=False, internet=True)
check(r["score"] < offline - 0.1 and "does not exist" in r["explanation"],
      f"dead domain is penalised ({offline:.2f} -> {r['score']:.2f}) and explained")
r = with_fakes("https://unknown-news-site.com/article/x", resolves=False, internet=False)
check(r["score"] == offline, "no internet at all is NOT mistaken for a dead domain")
r = with_fakes("https://unknown-news-site.com/article/x",
               fetch=lambda *a, **k: credibility.FetchResult(404, "http"))
check(r["score"] < offline and "404" in r["explanation"], f"HTTP 404 lowers the score a little ({r['score']:.2f})")
r = with_fakes("https://unknown-news-site.com/article/x",
               fetch=lambda *a, **k: credibility.FetchResult(403, "blocked"))
check(r["score"] == offline and "blocks" in r["explanation"], "HTTP 403 is a note, not a penalty")


def _boom(*a, **k):
    raise RuntimeError("simulated crash inside the fetcher")


r = with_fakes("https://unknown-news-site.com/article/x", fetch=_boom)
check(0 <= r["score"] <= 1, "an exception inside the fetcher is contained")

SCHOLARLY_HTML = """<html><head>
<meta name="citation_journal_title" content="Journal of Examples">
<meta name="citation_doi" content="10.1234/example.5678">
<meta name="citation_author" content="Ada Lovelace">
<meta name="citation_publication_date" content="2021/03/01">
<script type="application/ld+json">{"@type": "ScholarlyArticle", "author": {"name": "Ada"}}</script>
</head><body><a href="https://doi.org/10.1/a">a</a><a href="https://doi.org/10.1/b">b</a>
<a href="https://pubmed.ncbi.nlm.nih.gov/1">c</a></body></html>"""
page = lambda *a, **k: credibility.FetchResult(200, "", "text/html", SCHOLARLY_HTML, a[0] if a else "")
plain = score_url("https://unknown-press.com/view/123", use_llm=False)["score"]
r = with_fakes("https://unknown-press.com/view/123", fetch=page)
check(r["score"] > plain + 0.1 and "Journal of Examples" in r["explanation"],
      f"scholarly meta tags lift an unknown host ({plain:.2f} -> {r['score']:.2f})")

SPONSORED_HTML = "<html><body><article>This is Sponsored Content from BrandCo.</article></body></html>"
r = with_fakes("https://unknown-press.com/view/123",
               fetch=lambda *a, **k: credibility.FetchResult(200, "", "text/html", SPONSORED_HTML))
check(r["score"] < plain, f"a sponsorship disclosure lowers the score ({r['score']:.2f})")

BROKEN_HTML = "<html><head><script type='application/ld+json'>{not json</script><meta name=author" * 50
r = with_fakes("https://unknown-press.com/view/123",
               fetch=lambda *a, **k: credibility.FetchResult(200, "", "text/html", BROKEN_HTML))
check(0 <= r["score"] <= 1, "malformed HTML and JSON-LD are tolerated")

r = with_fakes("https://www.nejm.org/doi/full/10.1056/NEJMoa0000000",
               fetch_json=lambda *a, **k: {"is_retracted": True, "cited_by_count": 500})
check(r["score"] <= 0.10 and "RETRACTED" in r["explanation"] and r["source_type"] == "Retracted publication",
      f"#4  a retracted NEJM paper is capped ({r['score']:.2f})")

published = {"is_retracted": False, "cited_by_count": 120000, "publication_year": 2017, "type": "preprint",
             "locations": [{"source": {"type": "repository"}},
                           {"source": {"type": "journal", "display_name": "NeurIPS Proceedings"}}]}
arx_plain = score_url("https://arxiv.org/abs/1706.03762", use_llm=False)["score"]
r = with_fakes("https://arxiv.org/abs/1706.03762", fetch_json=lambda *a, **k: published)
check(r["score"] > arx_plain and "NeurIPS" in r["explanation"],
      f"#3  a preprint with a published, highly cited version rises ({arx_plain:.2f} -> {r['score']:.2f})")

check(credibility.extract_doi("https://x.org/doi/full/10.1056/NEJMoa2034577") == "10.1056/nejmoa2034577",
      "DOI extraction strips /full/ and lowercases")
check(credibility.extract_doi("https://www.biorxiv.org/content/10.1101/2020.01.01.000001v1") == "10.1101/2020.01.01.000001",
      "bioRxiv DOI drops the version suffix")
check(credibility.extract_arxiv_id("https://arxiv.org/pdf/1706.03762v5") == "1706.03762", "arXiv id extraction")

print("\nAdded: LLM blend (#9), with a fake model answer")
w_conf = credibility.precision_blend(0.5, 0.20, 0.9, 0.10)[1]
w_unk = credibility.precision_blend(0.5, 0.20, 0.9, 0.30)[1]
check(w_unk > w_conf, f"an LLM that does not recognise the source gets less weight ({1-w_unk:.0%} vs {1-w_conf:.0%})")
blend, _ = credibility.precision_blend(0.4, 0.1, 0.8, 0.1)
check(abs(blend - 0.6) < 1e-9, "equal precisions give the simple average")
real_llm = credibility.llm_opinion
credibility.llm_opinion = lambda url: credibility.LLMOpinion(0.95, True, "Test says it is a top journal.")
credibility._CACHE.clear()
r = score_url_detailed("https://unknown-press.com/view/123", use_llm=None, use_network=False)
credibility.llm_opinion = real_llm
credibility._CACHE.clear()
check(r["score"] > plain and "llm" in r["layers"] and "AI reviewer" in r["explanation"],
      f"a confident LLM answer moves an uncertain rule score ({plain:.2f} -> {r['score']:.2f})")


print(f"\n{'=' * 60}")
print(f"  {PASSED} passed, {FAILED} failed")
print(f"{'=' * 60}\n")
raise SystemExit(1 if FAILED else 0)
