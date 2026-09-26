"""
training_data.py — labelled URLs used to FIT the URL model (train.py).

Kept strictly separate from evaluate.py:
  * no URL here appears in evaluate.py;
  * no domain from either HELD-OUT block of evaluate.py appears here
    (train.py checks this and refuses to run if it is violated).

Labels follow the same rubric as evaluate.py, so the two sets are comparable:

  0.90–0.95  peer-reviewed journals, primary government statistics, systematic reviews
  0.80–0.90  wire services, major newsrooms with corrections policies,
             intergovernmental organisations, curated references, official health bodies
  0.70–0.80  official software documentation, university institutional pages,
             reputable secondary reference
  0.45–0.65  preprints, open-edit encyclopaedias, opinion sections of good outlets,
             moderated community Q&A, personal academic pages
  0.20–0.40  self-published blogs, contributor platforms, press releases, forums
  0.05–0.20  social media posts, sponsored/promotional content, hyper-partisan sites
  0.00–0.05  satire, fabricated or throwaway sites

The labels are one student's judgment in one sitting, like the instructor's.
Many paths are illustrative ("example-..."): the URL model only ever sees the
string, so a realistic string is all it needs. Page-level evidence (Layer 2) is
not trained on this set.
"""

from typing import List, Tuple

# (url, label, category)
TRAINING_URLS: List[Tuple[str, float, str]] = [
    # ---------------- Peer-reviewed journals ---------------------------------
    ("https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0230000", 0.88, "journal"),
    ("https://www.bmj.com/content/372/bmj.n71", 0.93, "journal"),
    ("https://www.cell.com/cell/fulltext/S0092-8674(20)30000-1", 0.93, "journal"),
    ("https://academic.oup.com/nar/article/49/D1/D10/6000000", 0.92, "journal"),
    ("https://onlinelibrary.wiley.com/doi/10.1002/sim.1234", 0.90, "journal"),
    ("https://link.springer.com/article/10.1007/s10994-020-05900-1", 0.88, "journal"),
    ("https://www.sciencedirect.com/science/article/pii/S0140673620301835", 0.90, "journal"),
    ("https://www.acpjournals.org/doi/10.7326/M20-1234", 0.92, "journal"),
    ("https://www.science.org/doi/10.1126/science.abc1234", 0.95, "journal"),
    ("https://www.nature.com/articles/s41591-020-0820-9", 0.95, "journal"),
    ("https://www.nejm.org/doi/full/10.1056/NEJMoa2001017", 0.95, "journal"),
    ("https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30183-5/fulltext", 0.95, "journal"),
    ("https://ieeexplore.ieee.org/document/8000000", 0.85, "journal"),
    ("https://dl.acm.org/doi/10.1145/3292500.3330701", 0.88, "journal"),
    ("https://www.jmlr.org/papers/v12/pedregosa11a.html", 0.88, "journal"),
    ("https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html", 0.87, "journal"),
    ("https://aclanthology.org/2020.acl-main.1/", 0.86, "journal"),
    ("https://pubs.acs.org/doi/10.1021/jacs.0c00001", 0.90, "journal"),
    ("https://www.frontiersin.org/articles/10.3389/fpsyg.2020.01234/full", 0.80, "journal"),
    ("https://www.mdpi.com/2072-6643/12/2/334", 0.70, "journal"),
    ("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7000000/", 0.90, "journal"),
    ("https://journals.sagepub.com/doi/10.1177/0956797620900000", 0.88, "journal"),
    ("https://www.tandfonline.com/doi/full/10.1080/01621459.2020.1000000", 0.90, "journal"),
    ("https://royalsocietypublishing.org/doi/10.1098/rspb.2020.0001", 0.90, "journal"),
    ("https://elifesciences.org/articles/12345", 0.90, "journal"),
    ("https://www.jci.org/articles/view/12345", 0.90, "journal"),
    ("https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.120.050000", 0.92, "journal"),
    ("https://www.annualreviews.org/doi/10.1146/annurev-statistics-010814-020120", 0.92, "journal"),
    ("https://www.cochranelibrary.com/cdsr/doi/10.1002/14651858.CD000000.pub2/full", 0.95, "journal"),
    ("https://pubmed.ncbi.nlm.nih.gov/32000000/", 0.90, "index"),
    ("https://www.jstor.org/stable/2345678", 0.88, "journal"),
    ("https://doi.org/10.1016/j.cell.2020.02.052", 0.90, "journal"),
    ("https://www.biorxiv.org/content/10.1101/2021.03.01.433000v2", 0.50, "preprint"),

    # ---------------- Preprints ---------------------------------------------
    ("https://arxiv.org/abs/2005.14165", 0.65, "preprint"),
    ("https://arxiv.org/pdf/2301.00001", 0.55, "preprint"),
    ("https://arxiv.org/abs/1810.04805", 0.65, "preprint"),
    ("https://www.medrxiv.org/content/10.1101/2020.04.14.20062463v1", 0.50, "preprint"),
    ("https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3500000", 0.50, "preprint"),
    ("https://osf.io/preprints/psyarxiv/abcd1", 0.50, "preprint"),
    ("https://www.researchsquare.com/article/rs-12345/v1", 0.45, "preprint"),
    ("https://www.preprints.org/manuscript/202001.0001/v1", 0.45, "preprint"),
    ("https://chemrxiv.org/engage/chemrxiv/article-details/60c7400000000", 0.50, "preprint"),
    ("https://eartharxiv.org/repository/view/1234/", 0.50, "preprint"),

    # ---------------- Government & public bodies ----------------------------
    ("https://www.cdc.gov/flu/about/index.html", 0.90, "government"),
    ("https://www.nih.gov/news-events/news-releases/example-release", 0.88, "government"),
    ("https://www.bls.gov/news.release/empsit.nr0.htm", 0.92, "government"),
    ("https://www.fda.gov/drugs/drug-safety-and-availability/example", 0.88, "government"),
    ("https://catalog.data.gov/dataset/example-dataset", 0.85, "government"),
    ("https://www.nasa.gov/missions/example-mission/", 0.88, "government"),
    ("https://www.gov.uk/government/statistics/example-statistics", 0.88, "government"),
    ("https://www.ons.gov.uk/economy/inflationandpriceindices/bulletins/example", 0.92, "government"),
    ("https://www.abs.gov.au/statistics/people/population/example", 0.90, "government"),
    ("https://www.federalreserve.gov/releases/h15/", 0.92, "government"),
    ("https://www.sec.gov/files/example-report.pdf", 0.85, "government"),
    ("https://www.noaa.gov/news/example-climate-report", 0.88, "government"),
    ("https://www.epa.gov/climate-indicators/example", 0.87, "government"),
    ("https://www.usda.gov/media/press-releases/2024/01/01/example", 0.80, "government"),
    ("https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679", 0.90, "government"),
    ("https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Example", 0.90, "government"),
    ("https://www.army.mil/article/123456/example", 0.78, "government"),
    ("https://medlineplus.gov/diabetes.html", 0.90, "government"),
    ("https://www.whitehouse.gov/briefing-room/statements-releases/2024/01/01/example/", 0.70, "government"),
    ("https://www.canada.ca/en/public-health/services/diseases/example.html", 0.87, "government"),
    ("https://www150.statcan.gc.ca/n1/daily-quotidien/240101/dq240101a-eng.htm", 0.90, "government"),
    ("https://www.nist.gov/publications/example-standard", 0.92, "government"),
    ("https://www.cancer.gov/about-cancer/treatment/types/example", 0.90, "government"),

    # ---------------- Intergovernmental / international orgs -----------------
    ("https://www.nato.int/cps/en/natohq/topics_1234.htm", 0.80, "intergovernmental"),
    ("https://www.itu.int/en/ITU-D/Statistics/Pages/stat/default.aspx", 0.88, "intergovernmental"),
    ("https://www.wipo.int/publications/en/details.jsp?id=1234", 0.87, "intergovernmental"),
    ("https://public.wmo.int/en/media/press-release/example", 0.87, "intergovernmental"),
    ("https://www.coe.int/en/web/human-rights-convention/example", 0.82, "intergovernmental"),
    ("https://www.iom.int/resources/example-report", 0.83, "intergovernmental"),
    ("https://www.esa.int/Science_Exploration/Space_Science/example", 0.88, "intergovernmental"),
    ("https://www.ecmwf.int/en/forecasts/documentation-and-support/example", 0.88, "intergovernmental"),
    ("https://www.worldbank.org/en/publication/wdr2024", 0.87, "intergovernmental"),
    ("https://data.worldbank.org/indicator/NY.GDP.MKTP.CD", 0.90, "intergovernmental"),
    ("https://www.oecd.org/en/publications/example-outlook_12345.html", 0.87, "intergovernmental"),
    ("https://www.un.org/en/climatechange/reports", 0.85, "intergovernmental"),
    ("https://www.unicef.org/reports/state-of-worlds-children-2023", 0.85, "intergovernmental"),
    ("https://www.bis.org/publ/qtrpdf/r_qt2312.htm", 0.88, "intergovernmental"),
    ("https://www.wto.org/english/res_e/publications_e/wtr23_e.htm", 0.86, "intergovernmental"),
    ("https://www.ecb.europa.eu/pub/economic-bulletin/html/eb202401.en.html", 0.90, "intergovernmental"),
    ("https://www.unhcr.org/refugee-statistics/", 0.87, "intergovernmental"),

    # ---------------- Universities -----------------------------------------
    ("https://news.mit.edu/2021/example-research-0101", 0.80, "academic"),
    ("https://www.ox.ac.uk/research/research-impact/example", 0.80, "academic"),
    ("https://www.hsph.harvard.edu/nutritionsource/healthy-eating-plate/", 0.85, "academic"),
    ("https://www.stanford.edu/about/", 0.78, "academic"),
    ("https://plato.stanford.edu/entries/statistics/", 0.88, "reference"),
    ("https://www.cam.ac.uk/research/news/example", 0.80, "academic"),
    ("https://engineering.berkeley.edu/news/2024/01/example/", 0.78, "academic"),
    ("https://www.jhsph.edu/research/centers-and-institutes/example/", 0.82, "academic"),

    # ---------------- Personal pages on academic hosts ----------------------
    ("https://www.cs.cmu.edu/~someone/notes/lecture1.html", 0.55, "personal_academic"),
    ("https://people.csail.mit.edu/someone/blog/my-opinion.html", 0.40, "personal_academic"),
    ("https://web.stanford.edu/~someone/random-thoughts.html", 0.45, "personal_academic"),
    ("https://www.math.ucla.edu/~someone/teaching/notes.pdf", 0.55, "personal_academic"),
    ("https://homes.cs.washington.edu/~someone/", 0.50, "personal_academic"),
    ("https://www.eecs.harvard.edu/~someone/essays/why-x.html", 0.45, "personal_academic"),
    ("https://sites.google.com/view/someone-research/home", 0.35, "personal_academic"),

    # ---------------- Mainstream news --------------------------------------
    ("https://www.theguardian.com/world/2024/jan/01/example-story", 0.80, "news"),
    ("https://www.npr.org/2024/01/01/1234567/example-story", 0.82, "news"),
    ("https://www.washingtonpost.com/politics/2024/01/01/example-story/", 0.80, "news"),
    ("https://www.economist.com/finance-and-economics/2024/01/01/example", 0.82, "news"),
    ("https://www.ft.com/content/12345678-aaaa-bbbb-cccc-1234567890ab", 0.82, "news"),
    ("https://www.bloomberg.com/news/articles/2024-01-01/example-story", 0.82, "news"),
    ("https://www.latimes.com/california/story/2024-01-01/example", 0.78, "news"),
    ("https://www.cbsnews.com/news/example-story/", 0.75, "news"),
    ("https://abcnews.go.com/Politics/example-story/story?id=123456", 0.75, "news"),
    ("https://www.pbs.org/newshour/politics/example-story", 0.82, "news"),
    ("https://www.axios.com/2024/01/01/example-story", 0.75, "news"),
    ("https://www.theatlantic.com/science/archive/2024/01/example/677000/", 0.75, "news"),
    ("https://www.nytimes.com/2024/01/01/us/example-story.html", 0.80, "news"),
    ("https://www.bbc.com/news/world-12345678", 0.82, "news"),
    ("https://www.reuters.com/business/example-story-2024-01-01/", 0.85, "news"),
    ("https://apnews.com/article/example-story-abcdef123", 0.85, "news"),
    ("https://www.wsj.com/articles/example-story-123456", 0.80, "news"),
    ("https://themarkup.org/privacy/2024/01/01/example-investigation", 0.83, "news"),
    ("https://calmatters.org/politics/2024/01/example/", 0.80, "news"),
    ("https://www.icij.org/investigations/example-papers/", 0.84, "news"),
    ("https://www.texastribune.org/2024/01/01/example/", 0.80, "news"),
    ("https://revealnews.org/article/example-investigation/", 0.82, "news"),
    ("https://www.thebureauinvestigates.com/stories/2024-01-01/example", 0.82, "news"),
    ("https://www.usatoday.com/story/news/2024/01/01/example/12345/", 0.72, "news"),
    ("https://www.csmonitor.com/World/2024/0101/example", 0.80, "news"),
    ("https://www.scientificamerican.com/article/example-science-story/", 0.82, "news"),
    ("https://www.statnews.com/2024/01/01/example-health-story/", 0.82, "news"),

    # ---------------- Opinion sections -------------------------------------
    ("https://www.nytimes.com/2024/01/01/opinion/example-column.html", 0.58, "opinion"),
    ("https://www.washingtonpost.com/opinions/2024/01/01/example/", 0.58, "opinion"),
    ("https://www.theguardian.com/commentisfree/2024/jan/01/example", 0.55, "opinion"),
    ("https://www.foxnews.com/opinion/example-column", 0.40, "opinion"),
    ("https://www.wsj.com/articles/example-opinion-11600000000", 0.78, "news"),
    ("https://thehill.com/opinion/campaign/123456-example/", 0.45, "opinion"),

    # ---------------- Partisan / tabloid / low-quality news ------------------
    ("https://www.foxnews.com/politics/example-story", 0.55, "news"),
    ("https://www.breitbart.com/politics/2024/01/01/example/", 0.25, "partisan"),
    ("https://www.dailymail.co.uk/news/article-1234567/example.html", 0.40, "tabloid"),
    ("https://nypost.com/2024/01/01/news/example/", 0.50, "tabloid"),
    ("https://www.huffpost.com/entry/example-story_n_123456", 0.55, "news"),
    ("https://www.thesun.co.uk/news/1234567/example/", 0.35, "tabloid"),
    ("https://www.infowars.com/posts/example-story", 0.05, "fabricated"),
    ("https://www.naturalnews.com/2024-01-01-example-story.html", 0.05, "fabricated"),
    ("https://www.zerohedge.com/markets/example-story", 0.20, "partisan"),
    ("https://www.thegatewaypundit.com/2024/01/example-story/", 0.08, "fabricated"),
    ("https://www.dailykos.com/stories/2024/1/1/123456/-example", 0.30, "partisan"),
    ("https://www.occupydemocrats.com/2024/01/01/example/", 0.15, "partisan"),

    # ---------------- Reference -------------------------------------------
    ("https://en.wikipedia.org/wiki/Linear_regression", 0.60, "reference"),
    ("https://de.wikipedia.org/wiki/Statistik", 0.60, "reference"),
    ("https://www.britannica.com/topic/statistics", 0.80, "reference"),
    ("https://www.merriam-webster.com/dictionary/credibility", 0.78, "reference"),
    ("https://www.investopedia.com/terms/r/regression.asp", 0.60, "reference"),
    ("https://www.mayoclinic.org/diseases-conditions/diabetes/symptoms-causes/syc-20371444", 0.85, "reference"),
    ("https://www.webmd.com/diabetes/example-article", 0.60, "reference"),
    ("https://www.healthline.com/nutrition/example-article", 0.55, "reference"),
    ("https://my.clevelandclinic.org/health/diseases/7104-diabetes", 0.83, "reference"),
    ("https://www.khanacademy.org/math/statistics-probability/example", 0.72, "reference"),

    # ---------------- Official documentation ------------------------------
    ("https://docs.python.org/3/library/statistics.html", 0.80, "docs"),
    ("https://pytorch.org/docs/stable/nn.html", 0.78, "docs"),
    ("https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html", 0.78, "docs"),
    ("https://pandas.pydata.org/docs/user_guide/groupby.html", 0.78, "docs"),
    ("https://developer.mozilla.org/en-US/docs/Web/HTTP/Status", 0.80, "docs"),
    ("https://learn.microsoft.com/en-us/azure/example-service/overview", 0.75, "docs"),
    ("https://docs.aws.amazon.com/s3/latest/userguide/example.html", 0.75, "docs"),
    ("https://www.tensorflow.org/api_docs/python/tf/keras", 0.76, "docs"),
    ("https://statsmodels.readthedocs.io/en/latest/regression.html", 0.70, "docs"),
    ("https://github.com/someone/some-repo", 0.40, "code"),
    ("https://gist.github.com/someone/abc123", 0.30, "code"),

    # ---------------- Community Q&A and forums ----------------------------
    ("https://superuser.com/questions/123456/how-to-fix-x", 0.45, "qa"),
    ("https://math.stackexchange.com/questions/123456/why-is-x-true", 0.50, "qa"),
    ("https://serverfault.com/questions/123456/example-question", 0.45, "qa"),
    ("https://stats.stackexchange.com/questions/123456/example", 0.50, "qa"),
    ("https://www.quora.com/Is-example-true", 0.20, "forum"),
    ("https://answers.yahoo.com/question/index?qid=20100101000000AAxxxx", 0.12, "forum"),
    ("https://www.reddit.com/r/science/comments/abc123/example_title/", 0.25, "forum"),
    ("https://old.reddit.com/r/conspiracy/comments/xyz789/example/", 0.12, "forum"),
    ("https://news.ycombinator.com/item?id=12345678", 0.25, "forum"),
    ("https://www.physicsforums.com/threads/example-question.123456/", 0.35, "forum"),
    ("https://discuss.pytorch.org/t/example-question/12345", 0.40, "forum"),
    ("https://community.spiceworks.com/t/example-topic/123456", 0.30, "forum"),

    # ---------------- Social media ----------------------------------------
    ("https://twitter.com/someone/status/1234567890", 0.15, "social"),
    ("https://x.com/someone/status/1234567890", 0.15, "social"),
    ("https://www.facebook.com/someone/posts/1234567890", 0.10, "social"),
    ("https://www.tiktok.com/@someone/video/1234567890", 0.10, "social"),
    ("https://www.youtube.com/watch?v=abc123xyz", 0.25, "social"),
    ("https://www.instagram.com/p/AbC123xyz/", 0.10, "social"),
    ("https://www.linkedin.com/pulse/example-post-someone/", 0.25, "social"),
    ("https://www.threads.net/@someone/post/AbC123", 0.12, "social"),

    # ---------------- Blogs and contributor platforms ---------------------
    ("https://someone.wordpress.com/2023/01/01/my-thoughts-on-x/", 0.20, "blog"),
    ("https://someone.substack.com/p/why-x-is-wrong", 0.30, "blog"),
    ("https://medium.com/@someone/understanding-transformers-abc123", 0.28, "blog"),
    ("https://towardsdatascience.com/example-post-abc123", 0.35, "blog"),
    ("https://someone.tumblr.com/post/123456/example", 0.12, "blog"),
    ("https://someone.wixsite.com/mysite/post/example", 0.15, "blog"),
    ("https://someone.weebly.com/blog/example", 0.15, "blog"),
    ("https://example-personal-site.com/blog/my-thoughts", 0.28, "blog"),
    ("https://dev.to/someone/example-post-1abc", 0.30, "blog"),
    ("https://hackernoon.com/example-post-abc123", 0.30, "blog"),
    ("https://someone.blogspot.com/2023/05/example-post.html", 0.18, "blog"),
    ("https://blog.someproduct.com/why-our-approach-wins", 0.25, "blog"),
    ("https://someone.netlify.app/posts/example/", 0.22, "blog"),

    # ---------------- Commercial / promotional ----------------------------
    ("https://www.prnewswire.com/news-releases/example-company-announces-301234567.html", 0.28, "promotional"),
    ("https://www.businesswire.com/news/home/20240101000001/en/Example-Announces", 0.28, "promotional"),
    ("https://www.globenewswire.com/news-release/2024/01/01/1234567/0/en/example.html", 0.28, "promotional"),
    ("https://shop.example-brand.com/products/super-greens-powder", 0.12, "promotional"),
    ("https://www.examplevitamins.com/blog/why-our-supplement-works", 0.15, "promotional"),
    ("https://www.goop.com/wellness/health/example-article/", 0.15, "promotional"),
    ("https://articles.mercola.com/sites/articles/archive/2024/01/01/example.aspx", 0.05, "fabricated"),
    ("https://www.examplenews.com/sponsored/best-credit-cards", 0.12, "promotional"),
    ("https://www.examplemagazine.com/partner-content/example-brand-story", 0.15, "promotional"),
    ("https://www.amazon.com/Example-Product/dp/B000000000", 0.15, "promotional"),

    # ---------------- Satire --------------------------------------------
    ("https://www.theonion.com/example-headline-1850000000", 0.05, "satire"),
    ("https://babylonbee.com/news/example-headline", 0.05, "satire"),
    ("https://www.newsthump.com/2024/01/01/example-headline/", 0.05, "satire"),
    ("https://www.thebeaverton.com/2024/01/example-headline/", 0.05, "satire"),
    ("https://www.dailymash.co.uk/news/example-headline-2024010112345", 0.05, "satire"),

    # ---------------- Fabricated / throwaway / suspicious ------------------
    ("https://natural-cures-now.info/doctors-hate-this-trick", 0.03, "fabricated"),
    ("http://real-patriot-news.xyz/shocking-election-truth", 0.03, "fabricated"),
    ("https://best-health-secrets.biz/detox-miracle-revealed", 0.03, "fabricated"),
    ("https://www.healthnutnews.com/example-vaccine-story/", 0.05, "fabricated"),
    ("http://247-breaking-news.online/celebrity-exposed", 0.04, "fabricated"),
    ("https://cnn-politics.com.co/breaking-story", 0.02, "fabricated"),
    ("http://192.168.10.20/news/article.html", 0.10, "fabricated"),
    ("https://truth-uncensored-daily.club/big-pharma-cover-up", 0.03, "fabricated"),
    ("https://freedom-alert-news.top/they-dont-want-you-to-know", 0.03, "fabricated"),
    ("http://the-real-story-247.site/miracle-weight-loss", 0.03, "fabricated"),
    ("https://superfoodhealing.net/amazing-cure-for-diabetes", 0.08, "fabricated"),
    ("https://www.worldtruth.tv/example-hidden-history/", 0.05, "fabricated"),
    ("http://yournewswire.com/example-story/", 0.05, "fabricated"),

    # ---------------- Unremarkable commercial sites -------------------------
    ("https://www.examplecompany.com/about-us", 0.45, "commercial"),
    ("https://www.acme-widgets.com/support/faq", 0.45, "commercial"),
    ("http://www.localbakery.net/menu.html", 0.40, "commercial"),
    ("https://www.nerdwallet.com/article/finance/example", 0.55, "commercial"),
    ("https://www.consumerreports.org/cars/example-review/", 0.80, "reference"),
]
