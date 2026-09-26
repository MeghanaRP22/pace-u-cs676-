"""
credibility.py — source credibility scoring for the CS676 Project 1 chatbot.
===============================================================================

THE CONTRACT (unchanged from the baseline)
-------------------------------------------------------------------------------
    score_url("https://www.pnas.org/doi/10.1073/pnas.2020123118")
    -> {"score": 0.9, "explanation": "Generally reliable (0.90, likely 0.80–0.95). ..."}

    score:       float in [0.0, 1.0].  0 = not credible, 1 = highly credible.
    explanation: str. A reader-facing reason for the score.

`score_url_detailed()` returns the same two keys plus the extra fields the app
needs (interval, source type, evidence list). `score_url()` is a thin wrapper
that strips it back down to the contract, so nothing downstream can break.

WHAT CHANGED FROM THE BASELINE (the algorithm, in one screen)
-------------------------------------------------------------------------------
The baseline added hand-picked numbers from three lookup tables. This version
is a small statistical model with optional evidence layers on top:

  Layer 1  URL MODEL (always on, offline, deterministic)
           `extract_features()` turns a URL into ~30 *structural* features that
           describe what KIND of source it is — a DOI, a journal-article path,
           a treaty-organisation TLD (.int), a Q&A thread, a self-hosted blog
           subdomain, a clickbait slug, a throwaway domain name... Features like
           these generalise to domains nobody put in a table.
           A fractional-logit regression (Papke & Wooldridge 1996) with an L2
           penalty maps features to a score. Its weights are LEARNED by
           `train.py` from ~190 labelled URLs in `training_data.py`, which share
           no URL and no held-out domain with the evaluation set.
           A published, citable domain-quality dataset (Lin et al. 2023, PNAS
           Nexus) enters as one feature among many, not as an override.
           200 bootstrap refits give a 90% interval around every score.

  Layer 2  EVIDENCE (on in the app, off in evaluate.py unless --net)
           Reads the page (scholarly citation_* meta tags, schema.org type, a
           named author, a date, outbound citations, corrections/ethics policy,
           sponsorship disclosures) and, when a DOI or arXiv id is found, asks
           OpenAlex (Crossref as fallback) for retraction status, venue and
           citation count. Dead domains, dead pages and timeouts are handled and
           explained, never raised.

  Layer 3  LLM JUDGMENT (only with ANTHROPIC_API_KEY)
           One Claude call, as before, now also reporting whether it actually
           recognises the source. Blended by PRECISION WEIGHTING (inverse
           variance) instead of the untested constant 0.6/0.4: a confident
           rule score keeps most of the weight, an uncertain one defers to the
           model, and an LLM that admits it does not know the source barely
           moves anything.

The KNOWN WEAKNESSES list at the bottom of the file records, item by item,
what was done about each of the twelve defects in the baseline.
"""

from __future__ import annotations

import concurrent.futures
import csv
import html
import json
import math
import os
import re
import socket
import ssl
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(HERE, "model_weights.json")
RATINGS_PATH = os.path.join(HERE, "data", "domain_pc1.csv")

# Which fitted variant of the URL model to use. "full" includes the published
# domain-ratings feature; "structural" is trained without it and exists so the
# report can show how much of the gain is the dataset and how much the features.
MODEL_VARIANT = os.getenv("CREDIBILITY_MODEL_VARIANT", "full")

# Model used for the Layer 3 judgment. Report which model produced your numbers.
# "claude-haiku-4-5" is ~5x cheaper while developing (see README).
JUDGE_MODEL = "claude-opus-5"

# Precision weighting needs a standard deviation for the LLM's error. 0.10 is
# roughly the per-URL error the course README reports for the LLM layer; a model
# that says it does NOT recognise the source gets a much wider 0.30, so it moves
# the blend very little. `evaluate.py --llm --tune-blend` re-estimates these
# from your own cached LLM answers.
LLM_SIGMA_RECOGNIZED = 0.10
LLM_SIGMA_UNRECOGNIZED = 0.30

# Network evidence is on by default in the app. Set CREDIBILITY_NETWORK=0 to
# score from the URL alone (evaluate.py and the tests do this for determinism).
NETWORK_TIMEOUT = 4.0          # seconds per HTTP request
NETWORK_BUDGET = 7.0           # seconds for all evidence on one URL
MAX_PAGE_BYTES = 1_000_000     # never read more than ~1 MB of HTML
USER_AGENT = "Mozilla/5.0 (compatible; CS676-CredibilityScorer/2.0; research use)"

# Neutral starting point used only by the fallback path when the model file
# is missing or unreadable.
NEUTRAL_SCORE = 0.5


# =============================================================================
# SECTION 1 — DOMAIN KNOWLEDGE (small, categorical, and cited)
# =============================================================================
# The curated table is kept from the baseline, but it is no longer the scorer:
# it is one input feature whose weight the regression learns. Each entry now
# also names a source CATEGORY, which drives the explanation and the advice the
# reader sees. Everything else in this section is a list of *types* of thing
# (hosting platforms, abused TLDs), not a list of reputations.

DOMAIN_SCORES: Dict[str, float] = {
    "nature.com": 0.95, "science.org": 0.95, "nejm.org": 0.95, "thelancet.com": 0.95,
    "pubmed.ncbi.nlm.nih.gov": 0.92,
    "arxiv.org": 0.75, "biorxiv.org": 0.70,
    "wikipedia.org": 0.65, "britannica.com": 0.80,
    "reuters.com": 0.85, "apnews.com": 0.85, "bbc.com": 0.82, "nytimes.com": 0.80, "wsj.com": 0.80,
    "medium.com": 0.35, "substack.com": 0.35, "blogspot.com": 0.25, "wordpress.com": 0.25,
    "reddit.com": 0.25, "quora.com": 0.20, "x.com": 0.15, "twitter.com": 0.15,
    "theonion.com": 0.05, "clickhole.com": 0.05, "babylonbee.com": 0.05,
}

DOMAIN_CATEGORY: Dict[str, str] = {
    "nature.com": "journal", "science.org": "journal", "nejm.org": "journal",
    "thelancet.com": "journal", "pubmed.ncbi.nlm.nih.gov": "index",
    "arxiv.org": "preprint", "biorxiv.org": "preprint",
    "wikipedia.org": "reference", "britannica.com": "reference",
    "reuters.com": "news", "apnews.com": "news", "bbc.com": "news",
    "nytimes.com": "news", "wsj.com": "news",
    "medium.com": "blog", "substack.com": "blog", "blogspot.com": "blog", "wordpress.com": "blog",
    "reddit.com": "forum", "quora.com": "forum", "x.com": "social", "twitter.com": "social",
    "theonion.com": "satire", "clickhole.com": "satire", "babylonbee.com": "satire",
}

# Platforms where *anyone* can publish under a subdomain. user.blogspot.com is a
# personal blog regardless of Blogger's own reputation (fixes weakness 11).
HOSTING_PLATFORMS = {
    "blogspot.com", "blogger.com", "wordpress.com", "medium.com", "substack.com",
    "tumblr.com", "wixsite.com", "weebly.com", "github.io", "gitlab.io", "netlify.app",
    "vercel.app", "pages.dev", "squarespace.com", "ghost.io", "livejournal.com",
    "typepad.com", "neocities.org", "webflow.io", "herokuapp.com", "glitch.me",
    "hashnode.dev", "carrd.co", "jimdosite.com", "site123.me", "mystrikingly.com",
}

# Multi-label public suffixes we need to find the registrable domain. A full
# Public Suffix List would be better; this covers the common cases offline.
MULTI_SUFFIXES = {
    "co.uk", "ac.uk", "gov.uk", "org.uk", "nhs.uk", "police.uk", "com.au", "gov.au",
    "edu.au", "org.au", "net.au", "ac.nz", "govt.nz", "co.nz", "org.nz", "gc.ca",
    "co.jp", "ac.jp", "go.jp", "or.jp", "com.br", "gov.br", "edu.br", "co.in", "gov.in",
    "ac.in", "nic.in", "edu.cn", "gov.cn", "com.cn", "ac.kr", "go.kr", "co.kr",
    "com.mx", "gob.mx", "edu.mx", "com.co", "gov.co", "edu.co", "co.za", "gov.za",
    "ac.za", "com.sg", "gov.sg", "edu.sg", "com.tr", "gov.tr", "edu.tr", "ac.il",
    "gov.il", "co.il", "com.ar", "gob.ar", "edu.ar", "ac.at", "gv.at", "europa.eu",
}

# Government / public-sector second-level labels used by many countries.
GOV_SECOND_LEVEL = {"gov", "gob", "go", "govt", "gouv", "gc", "gv", "nic", "mil", "nhs"}
ACADEMIC_SECOND_LEVEL = {"ac", "edu"}

# TLDs that appear disproportionately in spam, phishing and throwaway sites
# (Spamhaus "most abused TLDs" reports). A cheap TLD is not proof of anything,
# which is why this is one weighted feature and not a verdict.
ABUSED_TLDS = {
    "xyz", "top", "info", "biz", "click", "buzz", "online", "site", "club", "live",
    "icu", "cyou", "rest", "work", "loan", "win", "bid", "gq", "tk", "ml", "cf", "ga",
    "shop", "vip", "fun", "space", "website", "today", "news", "press",
}
COMMERCIAL_TLDS = {"com", "net", "io", "co", "ai", "app", "dev", "me", "tv"}

# Words that sensational or fabricated sites like to put in their NAME.
DOMAIN_SENSATIONAL = {
    "truth", "legit", "patriot", "freedom", "exposed", "secret", "secrets", "cure",
    "cures", "miracle", "awake", "alert", "shocking", "viral", "uncensored", "insider",
    "conspiracy", "hoax", "liberty", "real", "natural", "wake", "red-pill", "redpill",
}

# Words that clickbait or promotional content puts in its SLUG.
CLICKBAIT_TERMS = {
    "miracle", "cure", "cures", "cured", "secret", "secrets", "shocking", "shock",
    "hate", "hates", "trick", "tricks", "detox", "exposed", "expose", "truth", "hoax",
    "banned", "conspiracy", "unbelievable", "insane", "believe", "weight-loss",
    "big-pharma", "cover-up", "coverup", "jaw-dropping", "mind-blowing", "must-see",
    "bombshell", "destroys", "slams", "hidden", "revealed", "reveal", "instantly",
    "overnight", "melt", "amazing", "wonder",
}


# =============================================================================
# SECTION 2 — URL PARSING HELPERS
# =============================================================================
# Everything here is pure string work and must never raise: the scorer is fed
# whatever a search engine returns, including IDNs, IP addresses, ports, and
# percent-encoded junk. Each helper returns a safe empty value on bad input.

def _normalize_domain(url: str) -> str:
    """Lowercase host without userinfo, port, trailing dot, or leading 'www.'."""
    try:
        host = (urlparse(url).netloc or "").lower()
    except Exception:
        return ""
    host = host.split("@")[-1]
    if host.startswith("["):                     # IPv6 literal
        return host.split("]")[0] + "]"
    host = host.split(":")[0].rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def _registered_domain(host: str) -> str:
    """'blogs.cs.ox.ac.uk' -> 'ox.ac.uk'; 'en.wikipedia.org' -> 'wikipedia.org'."""
    labels = [p for p in host.split(".") if p]
    if len(labels) <= 2:
        return ".".join(labels)
    if ".".join(labels[-2:]) in MULTI_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _public_suffix(host: str) -> str:
    """The suffix part of the host: 'ac.uk', 'gov', 'com', ..."""
    labels = [p for p in host.split(".") if p]
    if len(labels) >= 2 and ".".join(labels[-2:]) in MULTI_SUFFIXES:
        return ".".join(labels[-2:])
    return labels[-1] if labels else ""


def _is_ip(host: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host)) or host.startswith("[")


def _match_known_domain(domain: str) -> Optional[Tuple[str, float]]:
    """Exact match first, then suffix match (so 'en.wikipedia.org' finds 'wikipedia.org')."""
    if domain in DOMAIN_SCORES:
        return domain, DOMAIN_SCORES[domain]
    for known, score in DOMAIN_SCORES.items():
        if domain.endswith("." + known):
            return known, score
    return None


def _slug_tokens(path: str) -> List[str]:
    """Split a URL path into lowercase word tokens, keeping hyphenated bigrams."""
    words = [w for w in re.split(r"[^a-z0-9]+", unquote(path).lower()) if w]
    bigrams = [f"{a}-{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


def _logit(p: float) -> float:
    p = min(0.98, max(0.02, p))
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


# =============================================================================
# SECTION 3 — PUBLISHED DOMAIN RATINGS (Lin et al., 2023)
# =============================================================================
# 11,520 news domains rated by aggregating six expert rating systems (NewsGuard,
# MBFC, Lasser et al., and others) into one principal component in [0, 1].
# Citable and reproducible, unlike the hand-typed table. It rates NEWS quality,
# so it is fed to the regression as a feature, not trusted as an answer —
# e.g. it rates medium.com 0.72 because some outlets publish there.

_RATINGS: Optional[Dict[str, float]] = None
_RATINGS_LOCK = threading.Lock()


def _load_ratings() -> Dict[str, float]:
    """Load the ratings CSV once; an absent file just disables the feature."""
    global _RATINGS
    if _RATINGS is not None:
        return _RATINGS
    with _RATINGS_LOCK:
        if _RATINGS is None:
            table: Dict[str, float] = {}
            try:
                with open(RATINGS_PATH, newline="", encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        try:
                            table[row["domain"].strip().lower()] = float(row["pc1"])
                        except (KeyError, ValueError):
                            continue
            except OSError:
                _warn_once("ratings", f"{RATINGS_PATH} not found; ratings feature disabled.")
            _RATINGS = table
    return _RATINGS


_BRANDS: Optional[set] = None


def _brand_labels() -> set:
    """First labels of well-rated outlets ('abcnews', 'cnn', 'reuters') for look-alike checks."""
    global _BRANDS
    if _BRANDS is None:
        names = {d.split(".")[0] for d in DOMAIN_SCORES if DOMAIN_SCORES[d] >= 0.6}
        for d, v in _load_ratings().items():
            if v >= 0.6 and not re.search(r"\.(com|org|net)\.[a-z]{2}$", d):
                names.add(d.split(".")[0])
        _BRANDS = {n for n in names if len(n) >= 3}
    return _BRANDS


def domain_rating(host: str) -> Optional[float]:
    """Published quality rating for this host or its registered domain, if any."""
    ratings = _load_ratings()
    if host in ratings:
        return ratings[host]
    return ratings.get(_registered_domain(host))


# =============================================================================
# SECTION 4 — FEATURE EXTRACTION (Layer 1 inputs)
# =============================================================================
# Each feature is a number in roughly [0, 1] with a fixed name. The ORDER of
# FEATURE_NAMES is the order of the coefficients in model_weights.json, so add
# new features at the end and re-run train.py. Features describe the type of
# source, which is what lets the model score a domain it has never seen.

FEATURE_NAMES: List[str] = [
    "curated_logit",      # logit of the curated table score (0 if absent)
    "has_curated",        # 1 if the domain is in the curated table
    "ratings_logit",      # logit of the Lin et al. rating (0 if absent or curated)
    "has_ratings",        # 1 if a published rating was used
    "tld_gov",            # .gov / .mil / gov.uk / gc.ca / europa.eu ...
    "tld_int",            # .int — registrable only by treaty organisations
    "tld_edu",            # .edu / ac.uk / edu.au ...
    "tld_org",            # .org
    "tld_commercial",     # .com / .net / .io ...
    "tld_abused",         # TLDs over-represented in spam and phishing
    "doi",                # a DOI in the path or query string
    "scholarly_path",     # /doi/, /fullarticle/, /journals/, /pmc/articles/ ...
    "scholarly_host",     # host name says journal/academic/proceedings/pubs
    "preprint",           # preprint server or versioned preprint path
    "primary_publication",# /publications/, /data/, /statistics/, /reports/, fact sheets
    "news_path",          # dated article path or /news/, /article/, /story/
    "docs_path",          # official documentation (/docs/, /stable/, docs.*)
    "reference_path",     # encyclopaedia / dictionary style (/wiki/, /entries/)
    "self_hosted",        # subdomain of a publish-anything platform
    "ugc_path",           # /@user, /u/, /r/x/comments, /status/, /threads/ ...
    "qa_path",            # /questions/ — community Q&A
    "opinion",            # /opinion/, /commentisfree/, opinion.* subdomain
    "blog",               # /blog/ or blog.* subdomain
    "promotional",        # sponsored, advertorial, press release, shop
    "clickbait",          # sensational words in the slug (0, 0.5, 1)
    "suspicious_domain",  # hyphen-heavy, digit-laden, sensational domain name
    "impersonation",      # look-alike hosts such as abcnews.com.co
    "ip_host",            # bare IP address instead of a name
    "https",              # served over HTTPS
    "personal_page",      # ~user, /people/x on academic hosts
]

_SCHOLARLY_PATH = re.compile(
    r"/doi/|/fullarticle/|/full[-_]?text|/journals?/|/pmc/articles/|/abstract/|"
    r"/science/article/|/article/pii/|/content/\d+/|/articles/10\.|/papers/|"
    r"/paper/|/proceedings/|/document/\d+|/cdsr/"
)
_SCHOLARLY_HOST_TOKENS = ("journal", "academic", "proceedings", "pubs.", "onlinelibrary",
                          "scholar", "aclanthology", "ieeexplore", "dl.acm", "springer",
                          "sciencedirect", "tandfonline", "sagepub", "wiley", "elifesciences",
                          "royalsocietypublishing", "annualreviews", "cochranelibrary")
_PREPRINT_HOST = re.compile(r"rxiv|preprint|ssrn|researchsquare|authorea|osf\.io|zenodo")
_PRIMARY_PUB_PATH = re.compile(
    r"/publications?(/|-|_|$)|/data(/|$)|/statistics|/stats/|/reports?(/|$)|/fact-?sheets?|"
    r"/indicators?/|/releases?/|/news\.release|/research/|/publ/|/datasets?/|/surveys?/"
)
_NEWS_PATH = re.compile(r"/(19|20)\d{2}/\d{1,2}/|/(19|20)\d{2}-\d{2}-\d{2}|/news/|/article/|"
                        r"/articles/|/story/|/stories/|/politics/|/world/|/business/")
_DOCS_PATH = re.compile(r"/docs?/|/documentation/|/stable/|/latest/|/manual/|/api_docs/|"
                        r"/reference/|/modules/|/library/|/tutorials?/|/guide/")
_REFERENCE_PATH = re.compile(r"/wiki/|/entries/|/dictionary/|/topic/|/terms/|/encyclopedia/")
_UGC_PATH = re.compile(r"/@[\w.-]+|/u/|/user/|/users/|/r/\w+/comments/|/comments/|/status/|"
                       r"/threads?/|/forums?/|/t/|/profile/|/community/|/discussions?/|"
                       r"/posts?/\d|/pulse/|/video/|/watch\b|/p/[\w-]+/?$|/item\?|/showthread")
_QA_PATH = re.compile(r"/questions?/|/answers?/|/ask/")
_OPINION_PATH = re.compile(r"/opinions?(/|-)|/commentisfree/|/editorials?/|/columns?/|/op-?ed/|"
                           r"/perspectives?/|/analysis/opinion")
_BLOG_PATH = re.compile(r"/blogs?(/|$)|/diary/|/my-thoughts")
_PROMO_PATH = re.compile(r"/sponsored|/advertorial|/partner-content|/paid-?post|/press-?releases?|"
                         r"/news-releases?|/news-release/|/pr/|/products?/|/shop/|/store/|/offer|"
                         r"/deals?/|/buy[-/]|/brandvoice|/promo")
_PERSONAL_PATH = re.compile(r"/~|/people/[\w.-]+|/users?/~?[\w.-]+/|/home/[\w.-]+|/staff/[\w.-]+/")


def extract_features(url: str, use_ratings: bool = True) -> Dict[str, float]:
    """
    Turn a URL into the named feature dict the regression consumes.

    Pure function of the URL string (plus the static ratings file), so it is
    deterministic, needs no network, and is shared by train.py and the scorer.
    `use_ratings=False` zeroes the published-ratings features for the ablation.
    """
    f = {name: 0.0 for name in FEATURE_NAMES}
    parsed = urlparse(url.strip())
    host = _normalize_domain(url)
    reg = _registered_domain(host)
    suffix = _public_suffix(host)
    tld = suffix.split(".")[-1] if suffix else ""
    second = suffix.split(".")[0] if "." in suffix else ""
    path = unquote(parsed.path or "").lower()
    query = unquote(parsed.query or "").lower()
    full_path = path + ("?" + query if query else "")
    sub = host[: -len(reg)].rstrip(".") if reg and host.endswith(reg) else ""

    # --- Reputation priors ------------------------------------------------
    match = _match_known_domain(host)
    if match:
        f["curated_logit"] = _logit(match[1])
        f["has_curated"] = 1.0
    elif use_ratings:
        rating = domain_rating(host)
        if rating is not None:
            f["ratings_logit"] = _logit(rating)
            f["has_ratings"] = 1.0

    # --- Registry type of the domain --------------------------------------
    if tld in ("gov", "mil") or second in GOV_SECOND_LEVEL or suffix == "europa.eu":
        f["tld_gov"] = 1.0
    elif tld == "int":
        f["tld_int"] = 1.0
    elif tld == "edu" or second in ACADEMIC_SECOND_LEVEL:
        f["tld_edu"] = 1.0
    elif tld == "org":
        f["tld_org"] = 1.0
    elif tld in COMMERCIAL_TLDS and not second:
        f["tld_commercial"] = 1.0
    if tld in ABUSED_TLDS:
        f["tld_abused"] = 1.0

    # --- What kind of document the path points at -------------------------
    if re.search(r"(^|/|=|doi:)10\.\d{4,9}/", full_path):
        f["doi"] = 1.0
    if _SCHOLARLY_PATH.search(full_path):
        f["scholarly_path"] = 1.0
    if any(tok in host for tok in _SCHOLARLY_HOST_TOKENS):
        f["scholarly_host"] = 1.0
    if _PREPRINT_HOST.search(host) or re.search(r"/10\.1101/.*v\d+$", path):
        f["preprint"] = 1.0
    if _PRIMARY_PUB_PATH.search(full_path):
        f["primary_publication"] = 1.0
    if _NEWS_PATH.search(full_path):
        f["news_path"] = 1.0
    if _DOCS_PATH.search(full_path) or re.match(r"(docs|developer|learn|devdocs)\.", host) \
            or reg in ("readthedocs.io",):
        f["docs_path"] = 1.0
    if _REFERENCE_PATH.search(full_path):
        f["reference_path"] = 1.0

    # --- Who is speaking: institution, crowd, or one person ----------------
    if reg in HOSTING_PLATFORMS and sub and sub not in ("www", "blog", "en"):
        f["self_hosted"] = 1.0
    if host == "sites.google.com":
        f["self_hosted"] = 1.0
    if _UGC_PATH.search(full_path):
        f["ugc_path"] = 1.0
    if _QA_PATH.search(full_path) or re.match(r"(answers|ask)\.", host):
        f["qa_path"] = 1.0
    sub_labels = set(sub.split(".")) if sub else set()
    if _OPINION_PATH.search(full_path) or sub_labels & {"opinion", "opinions", "op-ed"}:
        f["opinion"] = 1.0
    if _BLOG_PATH.search(full_path) or sub_labels & {"blog", "blogs"}:
        f["blog"] = 1.0
    if _PROMO_PATH.search(full_path) or sub_labels & {"shop", "store", "promo"}:
        f["promotional"] = 1.0
    academic_host = f["tld_edu"] or sub_labels & {"people", "users", "students", "homes",
                                                  "personal", "web", "home", "staff"}
    if _PERSONAL_PATH.search(path) and (academic_host or "~" in path):
        f["personal_page"] = 1.0
    elif f["tld_edu"] and sub_labels & {"people", "users", "students", "homes", "personal"}:
        f["personal_page"] = 1.0

    # --- Tell-tales of fabricated or throwaway sites -----------------------
    tokens = set(_slug_tokens(path))
    hits = len(tokens & CLICKBAIT_TERMS)
    f["clickbait"] = 0.0 if hits == 0 else (0.5 if hits == 1 else 1.0)

    label = reg[: -len(suffix) - 1] if suffix and reg.endswith("." + suffix) else reg.split(".")[0]
    sus = 0.0
    if label.startswith("xn--"):          # punycode (internationalised) names are
        label = ""                         # legitimate; don't read its hyphens as spam
    if label.count("-") >= 2:
        sus += 0.4
    elif label.count("-") == 1:
        sus += 0.15
    if re.search(r"\d", label):
        sus += 0.2
    if len(label) > 18:
        sus += 0.15
    parts = set(label.split("-"))
    for word in DOMAIN_SENSATIONAL:
        if word in parts or (len(word) >= 5 and word in label):
            sus += 0.35
    f["suspicious_domain"] = min(1.0, sus) if not (f["has_curated"] or f["tld_gov"]) else 0.0

    # abcnews.com.co, cnn.com-breaking.info: a reputable outlet's name placed
    # in front of a doubled suffix ("com.co") or glued to ".com-" is the classic
    # look-alike trick. Brand names come from the curated and published ratings.
    doubled = bool(re.search(r"\.(com|org|net)\.[a-z]{2}$", reg)) or bool(re.search(r"\.(com|org|net|gov)-", host))
    if doubled and not f["has_curated"] and (set(label.split("-")) | {label}) & _brand_labels():
        f["impersonation"] = 1.0
    if _is_ip(host):
        f["ip_host"] = 1.0
    if parsed.scheme == "https":
        f["https"] = 1.0
    return f


# =============================================================================
# SECTION 5 — THE LEARNED URL MODEL (Layer 1)
# =============================================================================
# model_weights.json is written by train.py. It holds, per variant, the mean
# coefficient vector, 200 bootstrap coefficient vectors, and out-of-bag
# residuals split by whether the domain was known. Inference is a dot product
# and a sigmoid — pure Python, microseconds, no numpy needed at run time.

_MODEL: Optional[Dict[str, Any]] = None


def _load_model() -> Optional[Dict[str, Any]]:
    """Load (once) the fitted weights for MODEL_VARIANT, or None if unavailable."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL.get(MODEL_VARIANT) or None
    try:
        with open(WEIGHTS_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("feature_names") != FEATURE_NAMES:
            _warn_once("weights-schema", "model_weights.json does not match FEATURE_NAMES; "
                       "re-run `python train.py`. Using the fallback scorer.")
            _MODEL = {}
        else:
            _MODEL = data.get("variants", {})
    except (OSError, ValueError) as exc:
        _warn_once("weights", f"Could not load model_weights.json ({exc}); using fallback scorer.")
        _MODEL = {}
    return _MODEL.get(MODEL_VARIANT) or None


def _fallback_score(feats: Dict[str, float]) -> float:
    """Baseline-style score used only when the weights file is missing."""
    if feats["has_curated"]:
        return _sigmoid(feats["curated_logit"])
    base = NEUTRAL_SCORE
    if feats["tld_gov"] or feats["tld_int"]:
        base = 0.85
    elif feats["tld_edu"]:
        base = 0.75
    elif feats["tld_abused"]:
        base = 0.3
    return base


@dataclass
class ModelResult:
    score: float
    low: float
    high: float
    contributions: Dict[str, float] = field(default_factory=dict)  # logit units


def model_score(url: str, use_ratings: Optional[bool] = None) -> ModelResult:
    """
    Layer 1: learned score plus a 90% bootstrap interval.

    The interval pairs each bootstrap model's prediction with an out-of-bag
    residual from the same "known domain / unknown domain" group, so it widens
    for sources the training data says we are bad at — the heteroscedastic
    residual bootstrap of Efron & Tibshirani (1993), ch. 9.
    """
    model = _load_model()
    if use_ratings is None:
        use_ratings = MODEL_VARIANT != "structural"
    feats = extract_features(url, use_ratings=use_ratings)
    x = [feats[n] for n in FEATURE_NAMES]
    if not model:
        s = _fallback_score(feats)
        return ModelResult(s, max(0.0, s - 0.2), min(1.0, s + 0.2), {})

    coef = model["coef"]
    z = model["intercept"] + sum(w * v for w, v in zip(coef, x))
    contributions = {n: w * v for n, w, v in zip(FEATURE_NAMES, coef, x) if w * v != 0}
    point = _sigmoid(z)

    known = feats["has_curated"] or feats["has_ratings"]
    residuals = model["residuals_known"] if known else model["residuals_unknown"]
    draws = []
    for i, (b0, bw) in enumerate(zip(model["boot_intercept"], model["boot_coef"])):
        zb = b0 + sum(w * v for w, v in zip(bw, x))
        r = residuals[(i * 7919) % len(residuals)] if residuals else 0.0
        draws.append(min(1.0, max(0.0, _sigmoid(zb) + r)))
    draws.sort()
    low = draws[int(0.05 * (len(draws) - 1))]
    high = draws[int(math.ceil(0.95 * (len(draws) - 1)))]
    return ModelResult(point, min(low, point), max(high, point), contributions)


# =============================================================================
# SECTION 6 — NETWORK EVIDENCE (Layer 2): fetching safely
# =============================================================================
# Every network call has a timeout, a byte cap, and a catch-all. The key subtlety
# is telling "this domain does not exist" (strong evidence against the source)
# from "this machine has no internet" (no evidence at all): a DNS failure only
# counts against a URL if a control lookup of a known-good host succeeds.

@dataclass
class Evidence:
    """One observation from the page or a metadata API, in logit units."""
    name: str
    delta: float
    reason: str


@dataclass
class FetchResult:
    status: int = 0                 # HTTP status, 0 if no response
    error: str = ""                 # "", "dns", "timeout", "ssl", "blocked", "http", "error"
    content_type: str = ""
    text: str = ""
    final_url: str = ""


_CONNECTIVITY: Dict[str, bool] = {}


def _resolves(host: str, timeout: float = 3.0) -> Optional[bool]:
    """True/False if DNS answered in time, None if we could not tell."""
    result: List[Optional[bool]] = [None]

    def _lookup() -> None:
        try:
            socket.getaddrinfo(host, 443)
            result[0] = True
        except socket.gaierror:
            result[0] = False
        except Exception:
            result[0] = None

    t = threading.Thread(target=_lookup, daemon=True)
    t.start()
    t.join(timeout)
    return result[0]


def _internet_ok() -> bool:
    """Can this machine resolve public names at all? Cached for the process."""
    if "ok" not in _CONNECTIVITY:
        _CONNECTIVITY["ok"] = bool(_resolves("www.iana.org") or _resolves("example.com"))
    return _CONNECTIVITY["ok"]


def _fetch(url: str, timeout: float = NETWORK_TIMEOUT, max_bytes: int = MAX_PAGE_BYTES,
           accept: str = "text/html,application/xhtml+xml") -> FetchResult:
    """GET a URL with a timeout and byte cap. Never raises."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "")
            raw = resp.read(max_bytes)
            charset = resp.headers.get_content_charset() or "utf-8"
            return FetchResult(resp.status, "", ctype, raw.decode(charset, errors="replace"),
                               resp.geturl())
    except urllib.error.HTTPError as exc:
        kind = "blocked" if exc.code in (401, 403, 429, 451, 999) else "http"
        return FetchResult(exc.code, kind)
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, socket.gaierror):
            return FetchResult(0, "dns")
        if isinstance(reason, (socket.timeout, TimeoutError)):
            return FetchResult(0, "timeout")
        if isinstance(reason, ssl.SSLError):
            return FetchResult(0, "ssl")
        return FetchResult(0, "error")
    except (socket.timeout, TimeoutError):
        return FetchResult(0, "timeout")
    except ssl.SSLError:
        return FetchResult(0, "ssl")
    except Exception:
        return FetchResult(0, "error")


def _fetch_json(url: str) -> Optional[Dict[str, Any]]:
    res = _fetch(url, accept="application/json", max_bytes=2_000_000)
    if res.status != 200 or not res.text:
        return None
    try:
        return json.loads(res.text)
    except ValueError:
        return None


# =============================================================================
# SECTION 7 — NETWORK EVIDENCE (Layer 2): reading the page
# =============================================================================
# The parser collects only what the credibility literature says matters and is
# cheap to detect: machine-readable scholarly metadata (the Highwire citation_*
# tags Google Scholar indexes), schema.org article type, a named author, a date,
# outbound citations, and the transparency links the Trust Project and
# NewsGuard criteria look for (corrections, ethics, ownership).

class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: Dict[str, List[str]] = {}
        self.jsonld: List[str] = []
        self.links: List[Tuple[str, str]] = []
        self._in_ld = False
        self._in_a: Optional[str] = None
        self._a_text: List[str] = []
        self.text_chunks = 0

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "meta":
            key = (a.get("name") or a.get("property") or a.get("itemprop") or "").lower()
            if key and a.get("content"):
                self.meta.setdefault(key, []).append(a["content"].strip())
        elif tag == "script" and "ld+json" in a.get("type", "").lower():
            self._in_ld = True
            self.jsonld.append("")
        elif tag == "a" and a.get("href"):
            self._in_a = a["href"]
            self._a_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_ld = False
        elif tag == "a" and self._in_a is not None:
            self.links.append((self._in_a, " ".join(self._a_text).strip().lower()))
            self._in_a = None

    def handle_data(self, data: str) -> None:
        if self._in_ld and self.jsonld:
            self.jsonld[-1] += data
        elif self._in_a is not None:
            self._a_text.append(data)
        self.text_chunks += 1


def _jsonld_items(blobs: List[str]) -> List[Dict[str, Any]]:
    """Flatten JSON-LD blocks (including @graph) into a list of dicts."""
    out: List[Dict[str, Any]] = []
    for blob in blobs[:10]:
        try:
            data = json.loads(blob.strip())
        except ValueError:
            continue
        stack = data if isinstance(data, list) else [data]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                out.append(item)
                if isinstance(item.get("@graph"), list):
                    stack.extend(item["@graph"])
            elif isinstance(item, list):
                stack.extend(item)
    return out


_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>#?&]+", re.I)


def extract_doi(text: str) -> Optional[str]:
    """First plausible DOI in a URL or string, cleaned of trailing path noise."""
    m = _DOI_RE.search(unquote(text or ""))
    if not m:
        return None
    doi = m.group(0).rstrip(".,;)/")
    doi = re.sub(r"/(full|abstract|pdf|epdf|html|fulltext|meta)$", "", doi, flags=re.I)
    doi = re.sub(r"v\d+(\.full(\.pdf)?)?$", "", doi) if doi.startswith("10.1101/") else doi
    return doi.lower()


def extract_arxiv_id(url: str) -> Optional[str]:
    m = re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})(?:v\d+)?", url, re.I)
    return m.group(1) if m else None


def page_evidence(html_text: str, url: str) -> Tuple[List[Evidence], Optional[str]]:
    """
    Read one HTML page and return (evidence list, DOI found in its metadata).

    Deltas are in logit units and deliberately modest: each one was set by hand
    (there is no fetched-page training corpus yet — see the report), so no
    single tag is allowed to dominate, and the total is clamped by the caller.
    """
    parser = _PageParser()
    try:
        parser.feed(html_text[:MAX_PAGE_BYTES])
    except Exception:
        return [], None
    meta = parser.meta
    items = _jsonld_items(parser.jsonld)
    types = set()
    for it in items:
        t = it.get("@type")
        for v in (t if isinstance(t, list) else [t]):
            if isinstance(v, str):
                types.add(v)
    ev: List[Evidence] = []

    # Machine-readable scholarly metadata: the strongest page-level signal.
    journal = (meta.get("citation_journal_title") or meta.get("prism.publicationname") or [""])[0]
    doi = extract_doi((meta.get("citation_doi") or meta.get("dc.identifier") or
                       meta.get("prism.doi") or [""])[0])
    if journal or "ScholarlyArticle" in types or "MedicalScholarlyArticle" in types:
        name = f" in {html.unescape(journal)}" if journal else ""
        ev.append(Evidence("scholarly_meta", 0.6, f"the page carries scholarly citation metadata{name}"))
    elif types & {"NewsArticle", "ReportageNewsArticle", "AnalysisNewsArticle"}:
        ev.append(Evidence("news_schema", 0.15, "the page declares itself a news report"))
    if types & {"OpinionNewsArticle", "BlogPosting", "SocialMediaPosting", "DiscussionForumPosting"}:
        ev.append(Evidence("opinion_schema", -0.25, "the page declares itself an opinion piece, blog or forum post"))

    # A named, accountable author.
    author = (meta.get("citation_author") or meta.get("author") or meta.get("article:author")
              or meta.get("parsely-author") or [])
    if not author:
        for it in items:
            a = it.get("author")
            if isinstance(a, dict) and a.get("name"):
                author = [a["name"]]
            elif isinstance(a, list) and a and isinstance(a[0], dict) and a[0].get("name"):
                author = [a[0]["name"]]
            if author:
                break
    article_like = bool(types & {"Article", "NewsArticle", "BlogPosting", "ScholarlyArticle",
                                 "ReportageNewsArticle", "OpinionNewsArticle"}) or bool(journal)
    if author and author[0] and not re.search(r"admin|staff|team|editor$", str(author[0]), re.I):
        ev.append(Evidence("author", 0.15, "it names an author"))
    elif article_like:
        ev.append(Evidence("no_author", -0.2, "it is an article with no named author"))

    date = (meta.get("citation_publication_date") or meta.get("article:published_time")
            or meta.get("citation_date") or meta.get("dc.date") or meta.get("date") or [])
    if date or any(it.get("datePublished") for it in items):
        ev.append(Evidence("date", 0.05, "it is dated"))

    # Outbound citations: links to DOIs, journals, or government sources.
    host = _normalize_domain(url)
    cites = 0
    policy = set()
    for href, text in parser.links[:3000]:
        h = href.lower()
        if "doi.org/10." in h or re.search(r"pubmed|ncbi\.nlm|\.gov/|jstor|arxiv\.org/abs", h):
            if host not in h:
                cites += 1
        blob = h + " " + text
        for key, pat in (("corrections", r"correction"), ("ethics", r"ethic|standards|editorial[- ]polic|principles"),
                         ("ownership", r"masthead|ownership|who we are|funding|our funders"),
                         ("factcheck", r"fact[- ]?check")):
            if re.search(pat, blob):
                policy.add(key)
    if cites >= 3:
        ev.append(Evidence("citations", 0.25, f"it links to {cites} scholarly or official sources"))
    if len(policy & {"corrections", "ethics", "ownership"}) >= 2:
        ev.append(Evidence("transparency", 0.3, "the site publishes " +
                           " and ".join(sorted(policy & {"corrections", "ethics", "ownership"})) +
                           " information, a marker of editorial accountability"))

    # Sponsorship disclosures.
    lowered = html_text[:MAX_PAGE_BYTES].lower()
    if re.search(r"sponsored content|paid post|advertorial|paid partnership|this post contains affiliate", lowered):
        ev.append(Evidence("sponsored", -0.4, "it discloses sponsorship or affiliate links"))
    return ev, doi


# =============================================================================
# SECTION 8 — NETWORK EVIDENCE (Layer 2): scholarly metadata APIs
# =============================================================================
# OpenAlex (Priem et al., 2022) answers the two questions a URL cannot: has this
# work been retracted, and has a preprint since been published in a journal?
# Crossref is used as a fallback for retraction status. Both are free and need
# no key; set OPENALEX_MAILTO to join OpenAlex's faster "polite pool".

def scholarly_evidence(doi: Optional[str], arxiv_id: Optional[str], is_preprint: bool
                       ) -> Tuple[List[Evidence], bool]:
    """Return (evidence, retracted?) for a work identified by DOI or arXiv id."""
    if arxiv_id and not doi:
        doi = f"10.48550/arxiv.{arxiv_id}"
    if not doi:
        return [], False
    ev: List[Evidence] = []
    mailto = os.getenv("OPENALEX_MAILTO", "")
    q = f"?mailto={quote(mailto)}" if mailto else ""
    work = _fetch_json(f"https://api.openalex.org/works/doi:{quote(doi, safe='/.:')}{q}")
    retracted = False
    if work:
        retracted = bool(work.get("is_retracted"))
        cites = int(work.get("cited_by_count") or 0)
        year = work.get("publication_year") or 0
        venue_src = ((work.get("primary_location") or {}).get("source") or {})
        journal_locs = [loc for loc in (work.get("locations") or [])
                        if ((loc or {}).get("source") or {}).get("type") == "journal"]
        if is_preprint and journal_locs:
            name = ((journal_locs[0].get("source") or {}).get("display_name") or "a journal")
            ev.append(Evidence("published_version", 0.6,
                               f"a peer-reviewed version has since appeared in {name}"))
        elif not is_preprint and venue_src.get("type") == "journal":
            ev.append(Evidence("indexed_journal", 0.3,
                               f"OpenAlex indexes it as a journal article in {venue_src.get('display_name', 'a journal')}"))
        if cites >= 1000:
            ev.append(Evidence("highly_cited", 0.45, f"it has been cited {cites:,} times"))
        elif cites >= 100:
            ev.append(Evidence("well_cited", 0.25, f"it has been cited {cites:,} times"))
        elif year and year <= 2023 and cites == 0:
            ev.append(Evidence("uncited", -0.2, "it has not been cited by any indexed work"))
    else:
        cr = _fetch_json(f"https://api.crossref.org/works/{quote(doi, safe='/.:')}")
        msg = (cr or {}).get("message") or {}
        for upd in msg.get("updated-by", []) or []:
            if str(upd.get("type", "")).lower() == "retraction":
                retracted = True
        title = " ".join(msg.get("title") or [])
        if title.upper().startswith("RETRACTED"):
            retracted = True
        if msg.get("type") == "journal-article" and not is_preprint:
            container = " ".join(msg.get("container-title") or []) or "a journal"
            ev.append(Evidence("indexed_journal", 0.3, f"Crossref registers it as a journal article in {container}"))
    if retracted:
        ev.append(Evidence("retracted", -3.0, "the work has been RETRACTED"))
    return ev, retracted


def gather_evidence(url: str, feats: Dict[str, float]) -> Tuple[List[Evidence], bool, List[str]]:
    """
    Run all network evidence for one URL inside a hard time budget.

    Returns (evidence, retracted?, notes). Notes are non-scoring facts the
    reader should know, e.g. "the page could not be fetched (timed out)".
    """
    notes: List[str] = []
    host = _normalize_domain(url)
    ev: List[Evidence] = []

    # 1. Does the domain exist at all? Only meaningful if the internet works.
    if not _is_ip(host):
        resolved = _resolves(host)
        if resolved is False and _internet_ok():
            return ([Evidence("dns", -1.2, "the domain does not exist or no longer resolves — "
                              "the site is gone or was never real")], False, notes)

    # 2. Page fetch and metadata lookup run in parallel under one budget.
    url_doi = extract_doi(url)
    arxiv_id = extract_arxiv_id(url)
    is_preprint = bool(feats.get("preprint")) or bool(arxiv_id)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    try:
        page_f = pool.submit(_fetch, url)
        meta_f = pool.submit(scholarly_evidence, url_doi, arxiv_id, is_preprint) \
            if (url_doi or arxiv_id) else None
        page_doi = None
        try:
            page = page_f.result(timeout=NETWORK_BUDGET)
        except concurrent.futures.TimeoutError:
            page = FetchResult(0, "timeout")
        if page.status == 200 and "html" in page.content_type.lower():
            page_ev, page_doi = page_evidence(page.text, url)
            ev.extend(page_ev)
        elif page.error == "http" and page.status in (404, 410):
            ev.append(Evidence("dead_page", -0.25, f"the page returns HTTP {page.status} (not found), "
                               "so its content cannot be checked"))
        elif page.error == "blocked":
            notes.append("the publisher blocks automated reading, so the page itself was not checked")
        elif page.error == "timeout":
            notes.append("the page timed out, so it was scored from its address alone")
        elif page.error == "ssl":
            ev.append(Evidence("bad_tls", -0.3, "its HTTPS certificate is invalid"))
        elif page.error:
            notes.append("the page could not be fetched, so it was scored from its address alone")

        retracted = False
        if meta_f is None and page_doi:
            meta_f = pool.submit(scholarly_evidence, page_doi, None, is_preprint)
        if meta_f is not None:
            try:
                meta_ev, retracted = meta_f.result(timeout=NETWORK_BUDGET)
                ev.extend(meta_ev)
            except concurrent.futures.TimeoutError:
                notes.append("the scholarly-metadata lookup timed out")
    finally:
        pool.shutdown(wait=False)

    # Don't double-count what the URL model already knew (weakness 12):
    # scholarly metadata on a page whose URL already looked scholarly adds half.
    for e in ev:
        if e.name == "scholarly_meta" and (feats.get("scholarly_path") or feats.get("doi")):
            e.delta *= 0.5
    return ev, retracted, notes


# =============================================================================
# SECTION 9 — LLM JUDGMENT (Layer 3) — needs ANTHROPIC_API_KEY
# =============================================================================
# Unchanged in spirit from the baseline (structured output, effort-parameter
# retry, refusal handling, warn-once diagnostics). Two additions: the schema
# asks whether the model RECOGNISES the source, which sets how much weight its
# answer gets; and answers can be persisted to a JSON cache so evaluate.py can
# tune the blend without paying for the same 24 calls twice.

_JUDGE_SYSTEM = """You assess the credibility of web sources for a research assistant.

Given a URL, judge how much a careful reader should trust content published there.
Consider: the publisher's editorial standards and reputation, whether the content is
peer reviewed, whether it is self-published, and whether the outlet is satirical.

Score 0.0 (not credible at all) to 1.0 (highly credible). Be skeptical of
self-published platforms and satire. Judge the SOURCE, not the topic. Set
"recognized" to false if you do not actually know this publisher; in that case
score near 0.5 rather than guessing confidently."""

_JUDGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "description": "Credibility from 0.0 to 1.0"},
        "recognized": {"type": "boolean", "description": "Do you know this publisher?"},
        "reason": {"type": "string", "description": "One sentence justifying the score"},
    },
    "required": ["score", "recognized", "reason"],
    "additionalProperties": False,
}

_NO_EFFORT_SUPPORT: set = set()
_WARNED: set = set()
LLM_CACHE_PATH = os.getenv("CREDIBILITY_LLM_CACHE", "")   # e.g. ".llm_cache.json"
_LLM_DISK: Optional[Dict[str, Any]] = None
_LLM_LOCK = threading.Lock()


def _warn_once(key: str, message: str) -> None:
    """Print a diagnostic to stderr the first time `key` is seen."""
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"  [credibility] {message}", file=sys.stderr)


@dataclass
class LLMOpinion:
    score: float
    recognized: bool
    reason: str


def _llm_disk_cache() -> Dict[str, Any]:
    global _LLM_DISK
    if _LLM_DISK is None:
        _LLM_DISK = {}
        if LLM_CACHE_PATH and os.path.exists(LLM_CACHE_PATH):
            try:
                with open(LLM_CACHE_PATH, encoding="utf-8") as fh:
                    _LLM_DISK = json.load(fh)
            except (OSError, ValueError):
                _LLM_DISK = {}
    return _LLM_DISK


def llm_opinion(url: str) -> Optional[LLMOpinion]:
    """Ask Claude to judge the URL. Returns None whenever the call cannot be made."""
    cache_key = f"{JUDGE_MODEL}|{url}"
    disk = _llm_disk_cache()
    if cache_key in disk:
        d = disk[cache_key]
        return LLMOpinion(d["score"], d["recognized"], d["reason"])
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None

    output_config: Dict[str, Any] = {"format": {"type": "json_schema", "schema": _JUDGE_SCHEMA}}
    if JUDGE_MODEL not in _NO_EFFORT_SUPPORT:
        output_config["effort"] = "low"
    try:
        import anthropic

        client = anthropic.Anthropic(timeout=30.0, max_retries=1)
        kwargs = dict(model=JUDGE_MODEL, max_tokens=1024, system=_JUDGE_SYSTEM,
                      messages=[{"role": "user", "content": f"Rate the credibility of this source: {url}"}])
        try:
            response = client.messages.create(**kwargs, output_config=output_config)
        except anthropic.BadRequestError as exc:
            if "effort" not in str(exc) or "effort" not in output_config:
                raise
            _NO_EFFORT_SUPPORT.add(JUDGE_MODEL)
            _warn_once(f"effort:{JUDGE_MODEL}", f"{JUDGE_MODEL} does not accept output_config.effort; "
                       "retrying without it. The LLM layer is still on.")
            output_config.pop("effort")
            response = client.messages.create(**kwargs, output_config=output_config)

        if response.stop_reason == "refusal":
            _warn_once(f"refusal:{JUDGE_MODEL}", f"{JUDGE_MODEL} declined to score a URL; using rules for it.")
            return None
        text = next((b.text for b in response.content if b.type == "text"), "")
        data = json.loads(text)
        op = LLMOpinion(max(0.0, min(1.0, float(data["score"]))), bool(data.get("recognized", True)),
                        str(data["reason"]))
        if LLM_CACHE_PATH:
            with _LLM_LOCK:
                disk[cache_key] = {"score": op.score, "recognized": op.recognized, "reason": op.reason}
                try:
                    with open(LLM_CACHE_PATH, "w", encoding="utf-8") as fh:
                        json.dump(disk, fh, indent=1)
                except OSError:
                    pass
        return op
    except Exception as exc:
        _warn_once(f"{type(exc).__name__}:{JUDGE_MODEL}",
                   f"LLM layer unavailable ({type(exc).__name__}: {str(exc)[:160]}). Scoring without it.")
        return None


def precision_blend(rule: float, rule_sigma: float, llm: float, llm_sigma: float) -> Tuple[float, float]:
    """
    Inverse-variance weighting of two noisy estimates of the same quantity.

    Returns (blended score, weight given to the rule layer). This is the
    minimum-variance unbiased combination when the two errors are independent,
    and it replaces the baseline's untested RULE_WEIGHT = 0.6 (weakness 9).
    """
    pr = 1.0 / max(rule_sigma, 1e-3) ** 2
    pl = 1.0 / max(llm_sigma, 1e-3) ** 2
    w = pr / (pr + pl)
    return w * rule + (1 - w) * llm, w


# =============================================================================
# SECTION 10 — EXPLANATIONS
# =============================================================================
# The baseline joined rule names with semicolons. Here the explanation answers
# three questions in order: how much should I trust this, what kind of source
# is it, and what should I check before relying on it. The "why" sentences come
# from the features that actually moved the score most, so they are faithful to
# the model rather than decoration.

SOURCE_TYPES: Dict[str, Tuple[str, str]] = {
    # key: (label shown to reader, what to do about it)
    "retracted": ("Retracted publication", "Do not rely on it; its findings were withdrawn."),
    "satire": ("Satire", "It is written to be funny, not true — do not cite it as fact."),
    "journal": ("Peer-reviewed journal article",
                "Peer review is strong but not infallible; check for later corrections or replications."),
    "index": ("Scholarly index / database", "Follow the record through to the original article."),
    "preprint": ("Preprint (not peer reviewed)",
                 "Check whether a peer-reviewed version exists before relying on its conclusions."),
    "government": ("Government / official source",
                   "Official data is usually reliable for facts and figures; official statements can still be one-sided."),
    "intergovernmental": ("International organisation",
                          "Authoritative for its own data and guidance; note the publication date."),
    "news": ("News report", "Reputable outlets correct errors; check the date and look for a second outlet."),
    "opinion": ("Opinion / commentary", "It argues a position — separate the facts it cites from the view it pushes."),
    "reference": ("Reference work", "Good for orientation; cite the primary sources it points to instead."),
    "docs": ("Official documentation", "Authoritative for how the software behaves; check the version."),
    "academic": ("University / academic site", "Institutional pages are reliable; personal pages reflect one person's view."),
    "personal_academic": ("Personal page on an academic server",
                          "Written by one person, not reviewed by the institution."),
    "qa": ("Community Q&A", "Often right but unreviewed — test answers yourself and prefer highly voted ones."),
    "forum": ("Forum / user-generated", "Anyone can post; treat claims as leads, not evidence."),
    "social": ("Social media post", "Unvetted; find the original source before sharing or citing."),
    "blog": ("Self-published blog", "No editor checked it; verify every factual claim elsewhere."),
    "promotional": ("Promotional / sponsored", "Written to sell something; look for independent sources."),
    "suspicious": ("Unverified, suspicious site",
                   "Shows the hallmarks of fabricated or throwaway sites; do not rely on it."),
    "unknown": ("Unrecognised source", "Look for an about page, named authors and cited evidence."),
}

# How each feature reads to a person, when it pushes the score up / down.
FEATURE_PHRASES: Dict[str, Tuple[str, str]] = {
    "curated_logit": ("the publisher has a strong track record", "the platform or outlet has a poor track record for accuracy"),
    "ratings_logit": ("independent news-quality raters score this domain well (Lin et al. 2023)",
                      "independent news-quality raters score this domain poorly (Lin et al. 2023)"),
    "tld_gov": ("it is a government domain, which only public bodies can register", ""),
    "tld_int": ("it uses .int, which only treaty-based international organisations can register", ""),
    "tld_edu": ("it is hosted by an accredited academic institution", ""),
    "tld_org": ("it is a .org domain, typical of non-profits and institutions", ""),
    "tld_commercial": ("", "it is an ordinary commercial domain that anyone can register"),
    "tld_abused": ("", "it uses a cheap top-level domain that is common among spam and throwaway sites"),
    "doi": ("the address contains a DOI, so it is a formally registered publication", ""),
    "scholarly_path": ("the address has the shape of a journal article", ""),
    "scholarly_host": ("the host is a scholarly publishing platform", ""),
    "preprint": ("", "it is a preprint, which has not been peer reviewed"),
    "primary_publication": ("it points at an organisation's own publications or data", ""),
    "news_path": ("it is a dated news article", ""),
    "docs_path": ("it is official documentation", ""),
    "reference_path": ("it is a reference entry", ""),
    "self_hosted": ("", "it is hosted on a platform where anyone can publish"),
    "ugc_path": ("", "the address points at user-posted content"),
    "qa_path": ("", "it is a community Q&A thread, which nobody edits"),
    "opinion": ("", "it is in an opinion section"),
    "blog": ("", "it is a blog post"),
    "promotional": ("", "the address marks it as sponsored, promotional or a press release"),
    "clickbait": ("", "the headline uses sensational words typical of clickbait"),
    "suspicious_domain": ("", "the domain name itself looks like a throwaway or sensational site"),
    "impersonation": ("", "the domain imitates a well-known site's address"),
    "ip_host": ("", "it is served from a bare IP address rather than a named site"),
    "https": ("", ""),
    "personal_page": ("", "it is one person's page, not an institutional publication"),
}


def _source_type(feats: Dict[str, float], host: str, retracted: bool) -> str:
    """Pick the single most informative source-type label for the reader."""
    if retracted:
        return "retracted"
    known = _match_known_domain(host)
    cat = DOMAIN_CATEGORY.get(known[0]) if known else None
    if cat == "satire":
        return "satire"
    if feats["impersonation"] or feats["ip_host"] or (feats["suspicious_domain"] >= 0.5 and feats["clickbait"]):
        return "suspicious"
    if feats["promotional"]:
        return "promotional"
    if feats["opinion"]:
        return "opinion"
    if cat in ("preprint",) or feats["preprint"]:
        return "preprint"
    if cat in ("journal", "index") or feats["doi"] or (feats["scholarly_path"] and (feats["scholarly_host"] or feats["has_ratings"] or feats["tld_org"])):
        return cat if cat in ("journal", "index") else "journal"
    if feats["personal_page"]:
        return "personal_academic"
    if feats["self_hosted"] or cat == "blog" or feats["blog"]:
        return "blog"
    if feats["qa_path"]:
        return "qa"
    if cat in ("social",):
        return "social"
    if cat == "forum" or feats["ugc_path"]:
        return "forum"
    if feats["tld_gov"]:
        return "government"
    if feats["tld_int"]:
        return "intergovernmental"
    if cat == "news" or (feats["news_path"] and (feats["has_ratings"] or feats["has_curated"])):
        return "news"
    if cat == "reference" or feats["reference_path"]:
        return "reference"
    if feats["docs_path"]:
        return "docs"
    if feats["tld_edu"]:
        return "academic"
    if feats["primary_publication"] and feats["tld_org"]:
        return "intergovernmental"
    if feats["suspicious_domain"] >= 0.5 or feats["clickbait"]:
        return "suspicious"
    if feats["news_path"]:
        return "news"
    return "unknown"


def _verdict(score: float) -> str:
    if score >= 0.70:
        return "Generally reliable"
    if score >= 0.40:
        return "Use with care"
    return "Treat with skepticism"


def _join(items: List[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def build_explanation(score: float, low: float, high: float, stype: str,
                      contributions: Dict[str, float], evidence: List[Evidence],
                      notes: List[str], llm: Optional[LLMOpinion], llm_weight: Optional[float]
                      ) -> Tuple[str, List[Dict[str, str]]]:
    """Compose the reader-facing explanation and a structured evidence list for the UI."""
    label, advice = SOURCE_TYPES.get(stype, SOURCE_TYPES["unknown"])
    bullets: List[Dict[str, str]] = []

    ranked = sorted(contributions.items(), key=lambda kv: -abs(kv[1]))
    ups, downs = [], []
    for name, c in ranked:
        up, down = FEATURE_PHRASES.get(name, ("", ""))
        if c >= 0.25 and up and len(ups) < 3:
            ups.append(up)
        elif c <= -0.25 and down and len(downs) < 3:
            downs.append(down)
    for e in sorted(evidence, key=lambda e: -abs(e.delta)):
        (ups if e.delta > 0 else downs).append(e.reason)
    for u in ups:
        bullets.append({"direction": "+", "text": u})
    for d in downs:
        bullets.append({"direction": "-", "text": d})
    for n in notes:
        bullets.append({"direction": "i", "text": n})

    parts = [f"{_verdict(score)} ({score:.2f}, likely range {low:.2f}–{high:.2f}). {label}."]
    if ups:
        parts.append(f"In its favour: {_join(ups[:3])}.")
    if downs:
        parts.append(f"Against it: {_join(downs[:3])}.")
    if llm is not None:
        who = "recognises" if llm.recognized else "does not recognise"
        parts.append(f"An AI reviewer ({JUDGE_MODEL}) {who} the source and rates it {llm.score:.2f}: "
                     f"{llm.reason.rstrip('.')}.")
        bullets.append({"direction": "i", "text": f"AI reviewer {llm.score:.2f} (weight {1 - (llm_weight or 0):.0%}): {llm.reason}"})
    if notes:
        parts.append(notes[0][0].upper() + notes[0][1:] + ".")
    parts.append(f"What to do: {advice}")
    return " ".join(parts), bullets


# =============================================================================
# SECTION 11 — THE FUNCTION YOU ARE GRADED ON
# =============================================================================
# score_url_detailed() runs the three layers in order and records everything
# the UI needs; score_url() keeps the original two-key contract. Results are
# memoised per (url, llm, network) setting, and every failure path returns a
# valid result instead of raising.

_CACHE: Dict[Tuple[str, Optional[bool], bool], Dict[str, Any]] = {}


def _network_default() -> bool:
    return os.getenv("CREDIBILITY_NETWORK", "1").strip().lower() not in ("0", "false", "no", "off")


def _clean_url(url: Any) -> Optional[str]:
    """Return a usable http(s) URL, adding https:// to bare 'example.com/x', else None."""
    if not isinstance(url, str):
        return None
    u = url.strip().strip("<>\"'")
    if not u or len(u) > 4096 or any(c in u for c in " \t\n\r"):
        return None
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", u) and not u.startswith("//") \
            and re.match(r"^[\w-]+(\.[\w-]+)+(/|$)", u):
        u = "https://" + u
    try:
        parsed = urlparse(u)
    except ValueError:
        return None
    host = _normalize_domain(u)
    if parsed.scheme.lower() not in ("http", "https") or not host:
        return None
    if not host.startswith("["):
        try:
            host = host.encode("idna").decode("ascii")   # bücher.de -> xn--bcher-kva.de
        except (UnicodeError, ValueError):
            return None
        labels = host.split(".")
        if len(labels) < 2 or any(not re.fullmatch(r"[a-z0-9_]([a-z0-9_-]{0,61}[a-z0-9_])?", lab)
                                  for lab in labels):
            return None
    return u


def _invalid(url: Any) -> Dict[str, Any]:
    shown = url if isinstance(url, str) else type(url).__name__
    msg = (f"Expected a URL string but got {type(url).__name__}." if not isinstance(url, str) and url is not None
           else "No URL was provided." if not (isinstance(url, str) and url.strip())
           else f"'{str(shown)[:80]}' is not a valid web address (http or https), so it cannot be scored.")
    return {"score": 0.0, "explanation": msg, "low": 0.0, "high": 0.0, "band": "LOW",
            "source_type": "Invalid URL", "evidence": [], "layers": []}


def score_url_detailed(url: str, use_llm: Optional[bool] = None,
                       use_network: Optional[bool] = None) -> Dict[str, Any]:
    """
    Score a URL and return the full record used by the app.

    Keys: score, explanation (the contract) plus low, high (90% interval),
    band, source_type, evidence (list of {direction, text}), layers (which of
    "model", "evidence", "llm" contributed).
    """
    net = _network_default() if use_network is None else bool(use_network)
    clean = _clean_url(url)
    if clean is None:
        return _invalid(url)
    key = (clean, use_llm, net)
    if key in _CACHE:
        return json.loads(json.dumps(_CACHE[key]))

    try:
        feats = extract_features(clean, use_ratings=MODEL_VARIANT != "structural")
        host = _normalize_domain(clean)
        m = model_score(clean)
        layers = ["model"]
        score, low, high = m.score, m.low, m.high

        # Layer 2: evidence shifts the score (and its interval) in logit space.
        evidence: List[Evidence] = []
        notes: List[str] = []
        retracted = False
        if net:
            try:
                evidence, retracted, notes = gather_evidence(clean, feats)
            except Exception as exc:          # belt and braces: never crash the app
                _warn_once(f"evidence:{type(exc).__name__}", f"evidence layer failed: {exc}")
                notes.append("the evidence check failed, so it was scored from its address alone")
            if evidence:
                layers.append("evidence")
                delta = max(-1.5, min(1.0, sum(e.delta for e in evidence if e.name != "retracted")))
                score, low, high = (_sigmoid(_logit(v) + delta) for v in (score, low, high))
            if retracted:
                score, low, high = min(score, 0.10), 0.0, min(high, 0.20)

        # Layer 3: precision-weighted blend with the LLM judgment.
        llm = llm_opinion(clean) if use_llm is not False else None
        llm_weight = None
        if llm is not None and not retracted:
            layers.append("llm")
            rule_sigma = max(0.03, (high - low) / 3.29)
            llm_sigma = LLM_SIGMA_RECOGNIZED if llm.recognized else LLM_SIGMA_UNRECOGNIZED
            score, llm_weight = precision_blend(score, rule_sigma, llm.score, llm_sigma)
            post_sigma = (1.0 / (1 / rule_sigma ** 2 + 1 / llm_sigma ** 2)) ** 0.5
            low, high = max(0.0, score - 1.645 * post_sigma), min(1.0, score + 1.645 * post_sigma)

        score = round(max(0.0, min(1.0, score)), 2)
        low = round(max(0.0, min(score, low)), 2)
        high = round(min(1.0, max(score, high)), 2)
        stype = _source_type(feats, host, retracted)
        explanation, bullets = build_explanation(score, low, high, stype, m.contributions,
                                                 evidence, notes, llm, llm_weight)
        result = {"score": float(score), "explanation": explanation, "low": float(low),
                  "high": float(high), "band": score_band(score)[0],
                  "source_type": SOURCE_TYPES.get(stype, SOURCE_TYPES["unknown"])[0],
                  "evidence": bullets, "layers": layers}
    except Exception as exc:
        # The contract promises a dict, never an exception. Log and degrade.
        _warn_once(f"score:{type(exc).__name__}", f"unexpected scoring error: {exc}")
        result = {"score": NEUTRAL_SCORE, "explanation": "The scorer hit an internal error on this "
                  "address, so it shows a neutral score. Verify the source manually.",
                  "low": 0.0, "high": 1.0, "band": "MEDIUM", "source_type": "Unscored",
                  "evidence": [], "layers": []}

    _CACHE[key] = json.loads(json.dumps(result))
    return result


def score_url(url: str, use_llm: Optional[bool] = None,
              use_network: Optional[bool] = None) -> Dict[str, Any]:
    """
    Score the credibility of a source URL.

    :param url:         The URL to evaluate.
    :param use_llm:     True/None use the Claude judgment when a key exists; False forces it off.
    :param use_network: Read the page and metadata APIs. None = CREDIBILITY_NETWORK env (default on).
    :return:            {"score": float in [0,1], "explanation": str}
    """
    r = score_url_detailed(url, use_llm=use_llm, use_network=use_network)
    return {"score": float(r["score"]), "explanation": str(r["explanation"])}


def score_band(score: float) -> Tuple[str, str]:
    """Map a score onto a display band: (label, streamlit colour)."""
    if score >= 0.70:
        return "HIGH", "green"
    if score >= 0.40:
        return "MEDIUM", "orange"
    return "LOW", "red"


# =============================================================================
# KNOWN WEAKNESSES — WHAT WAS DONE ABOUT EACH
# =============================================================================
#  1. NEVER READS THE PAGE → FIXED (Layer 2). Scholarly meta tags, schema.org
#     type, author, date, outbound citations, transparency links, sponsorship.
#     Page deltas are hand-set, not fitted: there is no fetched-page corpus yet.
#  2. HAND-WRITTEN DOMAIN TABLE → REDUCED. The table survives as one learned
#     feature; a published 11.5k-domain rating set (Lin et al. 2023) is another.
#  3. PREPRINT vs PEER REVIEW → FIXED. A preprint feature on any preprint host,
#     plus OpenAlex tells us when a preprint has a published journal version.
#  4. RETRACTION → FIXED when online. OpenAlex is_retracted / Crossref
#     updated-by caps the score at 0.10.
#  5. ANY .edu SCORES HIGH → FIXED. personal_page feature (~user, /people/x).
#  6. ARITHMETIC AGGREGATION → FIXED. Weights are fitted (fractional logit, L2),
#     and train.py reports a lasso path of which features carry signal.
#  7. NOT CALIBRATED → MEASURED. train.py reports cross-validated reliability
#     bins and calibration slope; evaluate.py reports interval coverage.
#  8. NO UNCERTAINTY → FIXED. 90% bootstrap interval on every score, shown in UI.
#  9. CONSTANT BLEND → FIXED. Precision (inverse-variance) weighting.
# 10. FRAGMENT EXPLANATIONS → FIXED. Verdict, source type, top reasons drawn
#     from the actual model contributions, and advice.
# 11. SUBDOMAIN INHERITANCE → FIXED for hosting platforms (self_hosted) and
#     opinion/blog subdomains; still inherited for other subdomains.
# 12. STACKING PENALTIES → FIXED. Features are indicators, not counts; the
#     logistic link saturates; evidence deltas are clamped to [-1.5, +1.0].
# =============================================================================
