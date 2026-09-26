// Builds report/CS676_Project1_Technique_Report.docx
// Run from the project root:  node report/build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, AlignmentType, ImageRun, LevelFormat, BorderStyle, Footer, PageNumber,
} = require("docx");

const S = JSON.parse(fs.readFileSync("results/summary.json", "utf8"));
const T = JSON.parse(fs.readFileSync("results/train_report.json", "utf8"));
const f3 = (x) => x.toFixed(3);
const pc = (x) => (100 * x).toFixed(1) + "%";

const FONT = "Calibri";
const P = (text, opts = {}) => new Paragraph({
  spacing: { after: 120, line: 276 }, alignment: opts.align,
  children: (Array.isArray(text) ? text : [text]).map((t) =>
    typeof t === "string" ? new TextRun({ text: t, font: FONT, size: 21 }) : t),
});
const B = (t) => new TextRun({ text: t, bold: true, font: FONT, size: 21 });
const I = (t) => new TextRun({ text: t, italics: true, font: FONT, size: 21 });
const C = (t) => new TextRun({ text: t, font: "Consolas", size: 19 });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 240, after: 120 }, children: [new TextRun({ text: t, font: FONT })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 180, after: 80 }, children: [new TextRun({ text: t, font: FONT })] });
const bullet = (parts) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60, line: 264 },
  children: (Array.isArray(parts) ? parts : [parts]).map((t) => typeof t === "string" ? new TextRun({ text: t, font: FONT, size: 21 }) : t) });
const caption = (t) => new Paragraph({ spacing: { after: 160 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: t, italics: true, font: FONT, size: 18, color: "52514E" })] });

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
function table(header, rows, widths, boldRow = -1) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, w, head, bold) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    borders: { top: border, bottom: border, left: border, right: border },
    shading: head ? { type: ShadingType.CLEAR, color: "auto", fill: "E8EEF7" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ children: [new TextRun({ text: String(t), bold: head || bold, font: FONT, size: 18 })] })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, widths[i], true)) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((v, i) => cell(v, widths[i], false, ri === boldRow)) }))],
  });
}
const img = (file, w, h) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
  children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: w, height: h } })] });

const b24 = S.baseline_orig24, s24 = S.structural_orig24, f24 = S.full_orig24;
const b48 = S.baseline_all48, s48 = S.structural_all48, f48 = S.full_all48, n48 = S.full_net_all48;

const children = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: "Learning to Score Source Credibility from URL Structure", bold: true, font: FONT, size: 34 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: "CS676 Algorithms for Data Science · Project 1 · Technique Report", font: FONT, size: 22, color: "52514E" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: "Harish · Pace University", font: FONT, size: 22, color: "52514E" })] }),

  H1("1. Summary"),
  P([
    "The course baseline scored a URL by adding hand-typed numbers from three lookup tables. It did well on the 16 domains in its table and collapsed on the 8 it had never seen (held-out MAE ",
    B(f3(b24["mae_held-out"])), "). I replaced the additive rules with a ", B("fractional-logit regression over 30 structural URL features"),
    ", whose weights are learned from 230 labelled URLs that share no domain with any held-out evaluation URL. A published, citable domain-quality dataset enters as one feature among many. Two hundred bootstrap refits put a 90% interval around every score; an optional evidence layer reads the page and queries OpenAlex/Crossref for retractions; the LLM layer is blended by inverse-variance weighting instead of a fixed 0.6/0.4 split; and the explanation is rebuilt from the features that actually moved the score.",
  ]),
  P([
    "On the instructor's 24 URLs, rules-only MAE fell from ", B(f3(b24.mae)), " to ", B(f3(f24.mae)),
    ", band accuracy rose from ", B(pc(b24.band_accuracy)), " to ", B(pc(f24.band_accuracy)), ", and the worst error fell from ",
    B(f3(b24.worst)), " to ", B(f3(f24.worst)), ". The gain is driven by the held-out block, where MAE fell from ",
    B(f3(b24["mae_held-out"])), " to ", B(f3(f24["mae_held-out"])), ". Without any API call, the new rules layer already beats the ",
    I("baseline plus Claude Opus 5"), " (MAE 0.086, 83.3%, as reported in the course README). On 24 additional held-out URLs I labelled, MAE fell from ",
    B(f3(b48.mae_extended)), " to ", B(f3(f48.mae_extended)), ".",
  ]),
  table(["Configuration (instructor's 24 URLs, rules only)", "MAE", "Band acc.", "Worst", "Held-out MAE", "Brier (HIGH)"], [
    ["Course baseline", f3(b24.mae), pc(b24.band_accuracy), f3(b24.worst), f3(b24["mae_held-out"]), f3(b24.brier_high)],
    ["Course baseline + LLM (from README, Opus 5)", "0.086", "83.3%", "0.230", "—", "—"],
    ["URL model, structural features only", f3(s24.mae), pc(s24.band_accuracy), f3(s24.worst), f3(s24["mae_held-out"]), f3(s24.brier_high)],
    ["URL model + published ratings (submitted)", f3(f24.mae), pc(f24.band_accuracy), f3(f24.worst), f3(f24["mae_held-out"]), f3(f24.brier_high)],
    ["Submitted model + LLM (evaluate.py --llm)", "run with key", "", "", "", ""],
  ], [3700, 900, 1000, 900, 1300, 1200], 3),
  caption("Table 1. Before/after on the unchanged course evaluation set. Lower is better except band accuracy. Brier score treats the score as the probability that the label is HIGH (≥ 0.70)."),

  H1("2. The algorithm and why I chose it"),
  H2("2.1 Diagnosis: why the baseline could not generalise"),
  P("A domain table is a memorised answer, not a model. When a domain is missing the baseline falls back to its TLD, so every .com scores 0.50 and a JAMA article scores 0.52. The fix is not a longer table but inputs that describe the kind of source, because kinds recur across domains the table will never contain: a DOI and a /journals/…/fullarticle/ path mean a journal article whoever publishes it, .int can only be registered by treaty organisations, /questions/ is community Q&A, a subdomain on blogspot.com is one person's blog, and a hyphenated .info domain with “miracle-cure” in the slug is a pattern, not a name."),
  H2("2.2 Features (Layer 1, offline and deterministic)"),
  P([C("extract_features()"), " maps a URL to 30 named values in [0, 1], in six groups:"]),
  bullet([B("Reputation priors: "), "logit of the curated table score (kept from the baseline, now a feature rather than the answer) and the logit of the Lin et al. (2023) news-domain quality rating when the domain is not curated, each with a presence indicator."]),
  bullet([B("Registry type: "), ".gov/.mil and foreign equivalents (gov.uk, gc.ca, europa.eu, nhs.uk), .int, academic (.edu, ac.xx), .org, ordinary commercial TLDs, and TLDs over-represented in abuse reports (Spamhaus)."]),
  bullet([B("Document type: "), "DOI, journal-article path, scholarly host, preprint, primary publication/data section, dated news path, official documentation, reference entry."]),
  bullet([B("Who is speaking: "), "self-hosted platform subdomain, user-content path, Q&A, opinion section, blog, promotional/sponsored/press-release, personal page on an academic server (weakness 5)."]),
  bullet([B("Fabrication tell-tales: "), "clickbait vocabulary in the slug; a lexical suspicious-domain score (hyphens, digits, length, sensational words in the name), in the spirit of lexical URL classifiers (Ma et al., 2009); look-alike hosts such as abcnews.com.co; bare IP hosts."]),
  bullet([B("Transport: "), "HTTPS."]),
  H2("2.3 Model: fractional logit with learned weights"),
  P(["The label is a proportion in [0, 1], so I use the fractional logit of Papke & Wooldridge (1996): ", I("E[y | x] = σ(b + w·x)"),
    ", fitted by minimising binomial cross-entropy with fractional targets plus an L2 penalty, using Newton's method (", C("train.py"),
    "). I chose it over (a) ordinary least squares, which predicts outside [0, 1] and lets penalties stack without a floor (weakness 12); (b) a tree ensemble, which with 230 rows would overfit and would not give per-feature contributions for the explanation; and (c) classifying into bands, which throws away the ordering the labels carry. The logistic link also makes signals combine sensibly: a second negative feature matters less once the score is already near zero. The penalty λ is chosen by 10-fold cross-validation (λ = ",
    String(T.full.ridge_lambda), "). Inference is a dot product in pure Python, so the app gains no dependency."]),
  P(["Training data (", C("training_data.py"), ") is 230 URLs across 17 source categories, labelled on the same rubric as the course set. ",
    C("train.py"), " refuses to run if any training URL, or any domain from either held-out block, appears in the evaluation set. Cross-validated MAE on the training set is ",
    B(f3(T.full.cv_mae)), " (full) and ", B(f3(T.structural.cv_mae)), " (structural only)."]),
  H2("2.4 Uncertainty and calibration (weaknesses 7 and 8)"),
  P(["Each of 200 bootstrap resamples is refitted. At scoring time every bootstrap model predicts, and each prediction is paired with an out-of-bag residual drawn from the same group — domains with a curated or published rating versus unknown domains — so the interval is wider where the training data says the model is weaker (a heteroscedastic residual bootstrap; Efron & Tibshirani, 1993). The 5th and 95th percentiles are shown in the app as “likely 0.44–0.88”. On all 48 evaluation URLs the 90% interval covers the label ",
    B(pc(f48.coverage)), " of the time (mean width ", f3(f48.mean_width), "), close to its nominal level. Cross-validated reliability bins (Figure 1) have a calibration slope of ",
    B(String(T.full.calibration.slope)), "; the one visible bias is that predictions of 0.4–0.6 average a label of 0.40, i.e. the model is slightly generous to middling sources."]),
  img("report/fig_reliability.png", 260, 245),
  caption("Figure 1. Reliability of out-of-fold predictions on the 230 training URLs (10-fold CV). Points near the diagonal mean a score of 0.7 corresponds to sources labelled about 0.7."),
  H2("2.5 Evidence layer: reading the page (weaknesses 1, 3, 4)"),
  P(["When online, the scorer resolves the domain, fetches the page (4 s timeout, 1 MB cap) and, if it finds a DOI or arXiv id in the URL or the page's metadata, queries OpenAlex (Priem et al., 2022) with Crossref as fallback. Page evidence follows what the web-credibility literature and journalism-transparency standards identify as cheap, strong signals: Highwire citation_* meta tags (what Google Scholar indexes), schema.org article type, a named author, a date, outbound links to DOIs and official sources, and links to corrections, ethics and ownership pages (Trust Project; NewsGuard criteria), plus sponsorship disclosures. Metadata evidence covers retraction (score capped at 0.10), whether a preprint has since appeared in a journal, and citation counts. Each observation shifts the score in logit space; the total is clamped to [−1.5, +1.0]. Robustness was designed in: a DNS failure counts against a site only if a control lookup succeeds (so an offline laptop is not mistaken for a dead web), 403/429 responses are notes rather than penalties, and every I/O path is behind a timeout and a catch-all."]),
  H2("2.6 LLM blend by precision weighting (weakness 9)"),
  P(["Instead of a constant 0.6/0.4, the rule score and the Claude score are combined by inverse-variance weighting — the minimum-variance combination of two independent unbiased estimates. The rule variance comes from the bootstrap interval; the LLM variance is 0.10 when the model reports that it recognises the publisher and 0.30 when it does not (the schema now asks). A confident rule score (NEJM) therefore keeps most of the weight, an uncertain one (an unseen journal) defers to the model, and an LLM that admits ignorance barely moves anything. ",
    C("evaluate.py --llm --tune-blend"), " re-estimates the LLM sigma from cached answers."]),
  H2("2.7 Explanations (weakness 10)"),
  P(["The explanation now answers three questions: how much to trust the source, what kind of source it is, and what to check. For example: ",
    I("“Use with care (0.68, likely range 0.44–0.88). Preprint (not peer reviewed). In its favour: … the publisher has a strong track record. Against it: it is a preprint, which has not been peer reviewed. What to do: Check whether a peer-reviewed version exists before relying on its conclusions.”"),
    " The “in its favour/against it” reasons are the features with the largest contributions w·x for this URL, so the text is faithful to the model rather than decorative. The app shows the score, range and source type on one line and puts the reasons in a collapsed expander."]),

  H1("3. Related work"),
  P([B("Human credibility judgments. "), "Fogg et al. (2001) and Fogg (2003) showed that users judge web credibility largely from surface cues — design, source identity, advertising — which motivates scoring cues that are visible without reading the content. Metzger (2007) reviews checklist-based credibility evaluation (authority, accuracy, currency, coverage) and finds users rarely apply it; my features automate the authority and provenance parts of that checklist, and the page layer the currency and accountability parts."]),
  P([B("Automatic web credibility prediction. "), "Olteanu et al. (2013) predicted expert credibility ratings of web pages with supervised learning over URL, content, appearance and social features, finding URL and content features among the most predictive. Wawer, Nielek & Wierzbicki (2014) added linguistic features. Popat et al. (2016) assess claims by aggregating the reliability of the sources that report them — the setting a research chatbot is in. My approach is closest to Olteanu et al. but restricted to features available cheaply at chat time."]),
  P([B("Source-level factuality of news media. "), "Baly et al. (2018) predict factuality and bias of news outlets from Wikipedia, Twitter, URL and article features, arguing that source-level labels scale better than claim-level fact-checking. Lin et al. (2023) show that six independent expert rating systems (including NewsGuard and Media Bias/Fact Check) agree strongly and release a combined principal-component score for 11,520 domains; I use it as a feature because it is published and reproducible, unlike a hand-typed table. Pennycook & Rand (2019) show that even crowd ratings of source quality correlate well with experts, supporting domain-level priors."]),
  P([B("URL-only classification. "), "Ma et al. (2009) detect malicious sites from lexical and host features of the URL alone, the precedent for my suspicious-domain, abused-TLD and look-alike features. Zhou & Zafarani (2020) survey fake-news detection and distinguish knowledge-, style-, propagation- and source-based methods; this project is source-based, supplemented by style cues in the headline slug."]),
  P([B("Scholarly metadata. "), "OpenAlex (Priem et al., 2022) and Crossref (Hendricks et al., 2020) expose venue, citation counts and retraction status openly, allowing the preprint-versus-published and retraction checks the baseline lacked."]),
  P([B("Statistical method. "), "Fractional logit (Papke & Wooldridge, 1996) for proportion outcomes; the lasso (Tibshirani, 1996) for feature selection; the bootstrap (Efron & Tibshirani, 1993) for intervals; and reliability diagrams and Brier scores (Brier, 1950; Niculescu-Mizil & Caruana, 2005) for calibration."]),

  H1("4. Results"),
  H2("4.1 Before and after"),
  table(["Scorer (rules only)", "All 48 MAE", "Band acc.", "Worst", "In-table (16)", "Held-out (8)", "Extended (24)", "90% cover"], [
    ["Course baseline", f3(b48.mae), pc(b48.band_accuracy), f3(b48.worst), f3(b48["mae_in-table"]), f3(b48["mae_held-out"]), f3(b48.mae_extended), "—"],
    ["Structural only", f3(s48.mae), pc(s48.band_accuracy), f3(s48.worst), f3(s48["mae_in-table"]), f3(s48["mae_held-out"]), f3(s48.mae_extended), pc(s48.coverage)],
    ["Structural + ratings", f3(f48.mae), pc(f48.band_accuracy), f3(f48.worst), f3(f48["mae_in-table"]), f3(f48["mae_held-out"]), f3(f48.mae_extended), pc(f48.coverage)],
    ["+ evidence (DNS only*)", f3(n48.mae), pc(n48.band_accuracy), f3(n48.worst), f3(n48["mae_in-table"]), f3(n48["mae_held-out"]), f3(n48.mae_extended), pc(n48.coverage)],
  ], [1900, 1000, 900, 750, 1100, 1100, 1150, 900], 2),
  caption("Table 2. MAE by block on all 48 evaluation URLs. *The development machine had DNS but no general HTTP egress, so the evidence row reflects only dead-domain detection; run evaluate.py --net online for the full layer."),
  img("report/fig_mae_blocks.png", 560, 256),
  caption("Figure 2. Mean absolute error by evaluation block. The largest gains are on domains the curated table has never seen."),
  P(["The structural features alone carry most of the improvement: held-out MAE falls from ", f3(b24["mae_held-out"]), " to ", f3(s24["mae_held-out"]),
    " without the ratings dataset. Adding published ratings lowers MAE everywhere (all-48 MAE ", f3(s48.mae), " → ", f3(f48.mae),
    ") and narrows the intervals (mean width ", f3(s48.mean_width), " → ", f3(f48.mean_width), "), but on the 8 held-out URLs it trades one band: the ratings give seekingalpha.com 0.69, which pulls it into MEDIUM. I submit the full model because it has the lower error on every block and set, and report the structural model so the contribution of the dataset is visible rather than hidden."]),
  H2("4.2 Which features carry signal (lasso path)"),
  P(["Refitting with an L1 penalty along a grid (", C("results/lasso_path_*.csv"), ") shows which features survive as λ grows. In the structural model, at λ = 0.02 only seven remain: the curated prior, commercial TLD (negative), .gov, .org, journal-article path, scholarly host and the suspicious-domain score; at λ = 0.05 only the curated prior and commercial TLD survive. In the full model the published rating is the last feature standing. Features the ridge fit gives near-zero weight — the DOI indicator (collinear with journal-article path) and the suspicious-domain score once the abused-TLD and clickbait features are present — are the ones the lasso drops first, so the model is telling us that several hand-designed features are redundant, not independently useful. The widest bootstrap spreads belong to rare features (impersonation ±", String(T.full.coef_boot_sd.impersonation), ", bare-IP ±", String(T.full.coef_boot_sd.ip_host), "), which have only a handful of training examples."]),

  H1("5. What still fails, and why"),
  bullet([B("Press releases and review sites on unknown .com domains. "), "prweb.com/releases/… scores 0.67 against a label of 0.25 (worst error, 0.42): /releases/ fires the primary-publication feature, which the model learned from government data releases. A linear model cannot learn that “releases” means data on .gov and marketing on .com; an interaction term would fix it, but I found this on the evaluation set and did not patch it, to keep the extended numbers honest. TripAdvisor reviews (0.49 vs 0.15) have no user-content path the features recognise."]),
  bullet([B("Reputable sources without structural cues. "), "ipcc.ch (0.66 vs 0.92) is a treaty body on a country-code domain with no .int; JAMA (0.69 vs 0.93) has a journal path but a .com host and a middling news-quality rating (0.70), because Lin et al. rate news quality, not scholarly quality. Both are exactly where the evidence layer (citation_* tags on JAMA pages) or the LLM should take over, which is why they are blended rather than replaced."]),
  bullet([B("Contributor networks inside reputable domains. "), "forbes.com/sites/… (0.68 vs 0.45) and seekingalpha.com (0.55 vs 0.35) inherit their domain's reputation, weakness 11 in a new form. A /sites/ contributor feature would need several contributor platforms in training."]),
  bullet([B("Satire is not structurally detectable. "), "The Onion is right only because it is in the curated table; a new satire site looks like a news site."]),
  bullet([B("Labels are one person's judgment. "), "The 230 training labels and the 24 extended labels are mine, on the instructor's rubric. The features were designed after reading the course evaluation URLs, so the held-out block is less “blind” than its name suggests; the extended block, labelled before scoring and never used for tuning, is the more honest test, and its MAE (", f3(f48.mae_extended), ") is correspondingly higher than the held-out block's (", f3(f48["mae_held-out"]), ")."]),
  bullet([B("Evidence deltas are hand-set. "), "No labelled corpus of fetched pages exists yet, so page-level weights are conservative guesses, not fits. The next step is to cache pages for the training URLs and fit these deltas the same way as the URL weights."]),
  bullet([B("Not measured here. "), "I had no API key and no general HTTP egress while developing, so the LLM-blend and full evidence-layer rows must be produced with evaluate.py --llm and --net; the code paths are covered by tests with fakes (77 tests pass)."]),
  H2("5.1 Where I disagree with the labels"),
  P(["Two instructor labels look harsher than the evidence supports. seekingalpha.com (0.35) is rated 0.69 by the aggregated expert systems of Lin et al. (2023) and publishes both staff news and contributor analysis; 0.40–0.45 (low MEDIUM) better reflects “read with care” than “treat as unreliable”. bioRxiv (0.50) is labelled 0.20 below arXiv (0.65) even though both are unrefereed preprint servers with light moderation; the arXiv label seems to reward the fame of one paper (“Attention Is All You Need”) rather than the source, which is precisely the source-versus-content confusion the judge prompt warns against. Neither disagreement was used to change the evaluation labels; the numbers above use them unchanged."]),

  H1("6. Reproducing these numbers"),
  P([C("uv run python train.py"), " (writes model_weights.json and results/), ", C("uv run python test_credibility.py"), " (77 tests), ",
    C("uv run python evaluate.py"), " (course 24), ", C("--extended"), " (all 48), ", C("--baseline"), " (the untouched course scorer, kept in baseline/), ",
    C("--variant structural"), " (ablation), ", C("--net"), " and ", C("--llm"), ". All numbers in this report are rules-only; no model produced them."]),

  H1("References"),
  ...[
    "Baly, R., Karadzhov, G., Alexandrov, D., Glass, J., & Nakov, P. (2018). Predicting factuality of reporting and bias of news media sources. Proceedings of EMNLP 2018, 3528–3539.",
    "Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. Monthly Weather Review, 78(1), 1–3.",
    "Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall/CRC.",
    "Fogg, B. J. (2003). Prominence-interpretation theory: Explaining how people assess credibility online. CHI '03 Extended Abstracts, 722–723.",
    "Fogg, B. J., Marshall, J., Laraki, O., et al. (2001). What makes web sites credible? A report on a large quantitative study. Proceedings of CHI 2001, 61–68.",
    "Hendricks, G., Tkaczyk, D., Lin, J., & Feeney, P. (2020). Crossref: The sustainable source of community-owned scholarly metadata. Quantitative Science Studies, 1(1), 414–427.",
    "Lin, H., Lasser, J., Lewandowsky, S., Cole, R., Gully, A., Rand, D. G., & Pennycook, G. (2023). High level of correspondence across different news domain quality rating sets. PNAS Nexus, 2(9), pgad286.",
    "Ma, J., Saul, L. K., Savage, S., & Voelker, G. M. (2009). Beyond blacklists: Learning to detect malicious web sites from suspicious URLs. Proceedings of KDD 2009, 1245–1254.",
    "Metzger, M. J. (2007). Making sense of credibility on the Web: Models for evaluating online information and recommendations for future research. JASIST, 58(13), 2078–2091.",
    "Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. Proceedings of ICML 2005, 625–632.",
    "Olteanu, A., Peshterliev, S., Liu, X., & Aberer, K. (2013). Web credibility: Features exploration and credibility prediction. Proceedings of ECIR 2013, 557–568.",
    "Papke, L. E., & Wooldridge, J. M. (1996). Econometric methods for fractional response variables with an application to 401(k) plan participation rates. Journal of Applied Econometrics, 11(6), 619–632.",
    "Pennycook, G., & Rand, D. G. (2019). Fighting misinformation on social media using crowdsourced judgments of news source quality. PNAS, 116(7), 2521–2526.",
    "Popat, K., Mukherjee, S., Strötgen, J., & Weikum, G. (2016). Credibility assessment of textual claims on the web. Proceedings of CIKM 2016, 2173–2178.",
    "Priem, J., Piwowar, H., & Orr, R. (2022). OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts. arXiv:2205.01833.",
    "Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society B, 58(1), 267–288.",
    "Wawer, A., Nielek, R., & Wierzbicki, A. (2014). Predicting webpage credibility using linguistic features. Proceedings of WWW '14 Companion, 1135–1140.",
    "Zhou, X., & Zafarani, R. (2020). A survey of fake news: Fundamental theories, detection methods, and opportunities. ACM Computing Surveys, 53(5), 1–40.",
    "The Trust Project. Trust Indicators. https://thetrustproject.org/ · NewsGuard. Rating process and criteria. https://www.newsguardtech.com/ · Spamhaus. The world's most abused TLDs. https://www.spamhaus.org/",
  ].map((r) => new Paragraph({ spacing: { after: 80 }, indent: { left: 360, hanging: 360 }, children: [new TextRun({ text: r, font: FONT, size: 18 })] })),
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 27, bold: true, color: "1F3864", font: FONT } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 23, bold: true, color: "2E5496", font: FONT } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 400, hanging: 260 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1080, left: 1150, right: 1150 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: "7F7F7F" })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(path.join("report", "CS676_Project1_Technique_Report.docx"), buf);
  console.log("wrote report/CS676_Project1_Technique_Report.docx");
});
