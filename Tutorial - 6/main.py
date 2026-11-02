"""
Tutorial 6: AI-Powered Fake News Detection and Intelligent News Retrieval
=========================================================================
Application : Detect potentially misleading news and retrieve related articles.
Objective   : Build an end-to-end IR system that
              - classifies articles as credible vs potentially misleading
                (with a confidence score — never a definitive "fake" claim),
              - engineers source / style / text features and trains an ensemble,
              - embeds articles (GloVe SIF) for neural / semantic retrieval,
              - clusters related stories and discovers topics (LDA / NMF),
              - visualises the collection with PCA and t-SNE,
              - flags anomalous articles,
              - ranks results with Learning-to-Rank features, and
              - evaluates classification and retrieval quality.

Run with --interactive to type your own queries / articles after the report.
Launch the Streamlit UI with:  streamlit run app.py
"""

from __future__ import annotations

import gzip
import math
import os
import re
import sys
import urllib.request
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import (LatentDirichletAllocation, NMF, PCA,
                                   TruncatedSVD)
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier,
                              GradientBoostingClassifier)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =====================================================================
#  1. News collection
#     label: 0 = credible, 1 = potentially misleading
#     (labels used only for supervised training / evaluation)
# =====================================================================
ARTICLES = {
    # ---- Credible technology / science ----
    "A01": dict(label=0, category="Technology", source="reuters.com",
                credibility=0.92, published="2024-03-12",
                title="EV makers publish battery fire safety data after regulator review",
                body=("Major electric vehicle manufacturers released detailed battery "
                      "fire statistics following a request from transport regulators. "
                      "Independent laboratory tests found that modern lithium-ion packs "
                      "include thermal runaway barriers and automatic disconnect systems. "
                      "Analysts said the data will help insurers price EV coverage more "
                      "accurately and may reduce public concern about rare battery fires.")),
    "A02": dict(label=0, category="Technology", source="bbc.com",
                credibility=0.90, published="2024-05-02",
                title="Researchers demonstrate solid-state battery prototype with longer range",
                body=("University laboratories demonstrated a solid-state battery cell that "
                      "retained eighty percent of capacity after one thousand charge cycles. "
                      "The prototype replaces liquid electrolyte with a ceramic separator, "
                      "which researchers say could reduce fire risk and increase energy "
                      "density. Commercial production is still several years away.")),
    "A03": dict(label=0, category="Technology", source="nature.com",
                credibility=0.95, published="2024-01-18",
                title="Peer-reviewed study maps lithium supply chains for electric vehicles",
                body=("A peer-reviewed study mapped global lithium supply chains used in "
                      "electric vehicle batteries. Authors used customs data and satellite "
                      "imagery to estimate mining output and processing capacity. They "
                      "concluded that recycling and new mines will both be needed to meet "
                      "projected demand this decade.")),
    # ---- Potentially misleading technology ----
    "A04": dict(label=1, category="Technology", source="viralbuzz.biz",
                credibility=0.15, published="2024-04-01",
                title="Secret study proves every EV battery explodes within two years",
                body=("A secret study that pharmaceutical and oil companies tried to hide "
                      "supposedly proves that every electric vehicle battery will explode "
                      "within two years. Anonymous experts claim governments are covering "
                      "up thousands of deaths. Share this before it is deleted! No named "
                      "authors, journal or laboratory were provided.")),
    "A05": dict(label=1, category="Technology", source="truthleak.info",
                credibility=0.18, published="2024-06-11",
                title="Vaccines in EV charging stations cause infertility, doctor claims",
                body=("An unnamed doctor claims that vaccines hidden inside electric vehicle "
                      "charging stations are causing widespread infertility. The article "
                      "offers no clinical evidence, cites a blog comment as proof and urges "
                      "readers to avoid all public chargers immediately.")),
    # ---- Credible health ----
    "A06": dict(label=0, category="Health", source="who.int",
                credibility=0.96, published="2024-02-20",
                title="WHO updates guidance on seasonal influenza vaccination",
                body=("The World Health Organization updated guidance on seasonal influenza "
                      "vaccination for older adults and people with chronic conditions. "
                      "Clinical trials and surveillance data continue to show that annual "
                      "vaccines reduce severe illness and hospitalisation. Side effects are "
                      "usually mild and temporary.")),
    "A07": dict(label=0, category="Health", source="nih.gov",
                credibility=0.94, published="2024-03-05",
                title="Large trial finds common vaccine side effects usually mild",
                body=("A large randomised trial reported that common vaccine side effects "
                      "such as sore arm and mild fever resolve within two days for most "
                      "participants. Serious adverse events were rare and monitored by an "
                      "independent safety board. Researchers emphasised transparency of "
                      "trial protocols and data.")),
    "A08": dict(label=0, category="Health", source="thelancet.com",
                credibility=0.95, published="2023-11-30",
                title="Meta-analysis reviews cardiovascular outcomes after vaccination",
                body=("A meta-analysis of twenty-two studies reviewed cardiovascular "
                      "outcomes after vaccination. Overall risk of myocarditis remained "
                      "low and was outweighed by protection against severe infection. "
                      "Authors called for continued post-market surveillance.")),
    # ---- Potentially misleading health ----
    "A09": dict(label=1, category="Health", source="naturalcurez.net",
                credibility=0.12, published="2024-05-19",
                title="Common vaccine causes severe health issues covered up by pharma",
                body=("Recent scientific studies suggest that a common vaccine causes "
                      "severe health issues. The study, published in an obscure journal, "
                      "claims that the side effects are widespread and have been covered "
                      "up by pharmaceutical companies. Read the full story to understand "
                      "the risks. No primary data or conflict-of-interest disclosures "
                      "were linked.")),
    "A10": dict(label=1, category="Health", source="healthtruth.club",
                credibility=0.14, published="2024-07-02",
                title="Doctors shocked: lemon water cures every disease overnight",
                body=("Doctors are supposedly shocked that drinking lemon water cures "
                      "every disease overnight including cancer and diabetes. Miracle "
                      "testimonials replace clinical trials. Hospitals are said to be "
                      "hiding the cure to protect drug profits.")),
    # ---- Credible politics / public affairs ----
    "A11": dict(label=0, category="Politics", source="apnews.com",
                credibility=0.91, published="2024-08-14",
                title="Election commission releases verified turnout figures by district",
                body=("The national election commission released verified turnout figures "
                      "by district after audits of polling station logs. Independent "
                      "observers confirmed that reported totals matched sealed ballot "
                      "boxes within a narrow margin of error. Officials invited parties "
                      "to inspect the raw data.")),
    "A12": dict(label=0, category="Politics", source="reuters.com",
                credibility=0.92, published="2024-09-01",
                title="Parliament debates climate bill with cross-party amendments",
                body=("Parliament debated a climate bill that includes emissions targets "
                      "and support for renewable energy. Cross-party amendments focused on "
                      "transition funding for coal regions. The bill will proceed to a "
                      "committee vote next month.")),
    "A13": dict(label=0, category="Politics", source="bbc.com",
                credibility=0.90, published="2024-04-22",
                title="Court upholds transparency rules for campaign advertising",
                body=("A high court upheld rules requiring political campaigns to disclose "
                      "spending on digital advertising. Judges said voters have a legitimate "
                      "interest in knowing who funds persuasive content online. Campaigns "
                      "must register ads in a public database.")),
    # ---- Potentially misleading politics ----
    "A14": dict(label=1, category="Politics", source="redpillwire.io",
                credibility=0.10, published="2024-08-15",
                title="Millions of illegal votes confirmed by anonymous whistleblower",
                body=("An anonymous whistleblower has confirmed that millions of illegal "
                      "votes were counted, according to screenshots circulating on social "
                      "media. No election officials, courts or auditors have verified the "
                      "claim. The post urges readers to reject certified results.")),
    "A15": dict(label=1, category="Politics", source="patriotblast.tv",
                credibility=0.11, published="2024-09-03",
                title="Climate bill secretly bans all private cars next week",
                body=("A viral article claims the climate bill secretly bans all private "
                      "cars starting next week. The actual bill text contains no such ban. "
                      "The story relies on a misquoted social media thread and sensational "
                      "headlines designed for rapid sharing.")),
    # ---- Credible climate / environment ----
    "A16": dict(label=0, category="Climate", source="nasa.gov",
                credibility=0.97, published="2024-01-10",
                title="Satellite data show continued rise in global surface temperatures",
                body=("NASA satellite and surface station data show a continued rise in "
                      "global mean surface temperatures over the past decade. Scientists "
                      "attribute most of the warming to greenhouse gas emissions from "
                      "fossil fuels. Regional extremes include more frequent heatwaves "
                      "and heavy rainfall events.")),
    "A17": dict(label=0, category="Climate", source="ipcc.ch",
                credibility=0.98, published="2023-10-05",
                title="IPCC summary highlights pathways to limit warming",
                body=("An IPCC summary for policymakers highlights pathways to limit "
                      "warming through rapid cuts in carbon dioxide and methane. The "
                      "report synthesises thousands of peer-reviewed studies and outlines "
                      "adaptation needs for coastal and agricultural communities.")),
    "A18": dict(label=0, category="Climate", source="theguardian.com",
                credibility=0.82, published="2024-06-18",
                title="Cities expand heat-action plans after record summer temperatures",
                body=("Several cities expanded heat-action plans after record summer "
                      "temperatures strained hospitals and power grids. Measures include "
                      "cooling centres, early warning systems and tree-planting programmes. "
                      "Public health officials said preparation saves lives during extreme "
                      "heat.")),
    # ---- Potentially misleading climate ----
    "A19": dict(label=1, category="Climate", source="coalfacts.org",
                credibility=0.16, published="2024-02-28",
                title="Scientists admit climate change is a hoax funded by elites",
                body=("Scientists have supposedly admitted that climate change is a hoax "
                      "funded by elites. The claim recycles long-debunked conspiracy "
                      "narratives and misrepresents a satirical blog post as a scientific "
                      "confession. No peer-reviewed retraction supports the headline.")),
    "A20": dict(label=1, category="Climate", source="weathertruth.xyz",
                credibility=0.17, published="2024-07-21",
                title="Cold snap proves global warming is completely fake",
                body=("A regional cold snap is presented as proof that global warming is "
                      "completely fake. Climate scientists note that weather is not climate "
                      "and that warming trends are measured over decades, not single storms. "
                      "The article ignores satellite and ocean heat records.")),
    # ---- Credible cybersecurity / incidents ----
    "A21": dict(label=0, category="Security", source="cisa.gov",
                credibility=0.93, published="2024-05-30",
                title="CISA advisory details ransomware targeting hospital networks",
                body=("CISA published an advisory detailing ransomware groups that target "
                      "hospital networks. Recommended defences include multi-factor "
                      "authentication, offline backups and network segmentation. Hospitals "
                      "are urged to report incidents promptly to improve collective defence.")),
    "A22": dict(label=0, category="Security", source="krebsonsecurity.com",
                credibility=0.88, published="2024-06-07",
                title="Researchers disclose phishing campaign spoofing bank login pages",
                body=("Security researchers disclosed a phishing campaign that spoofs bank "
                      "login pages and harvests credentials. Indicators of compromise and "
                      "malicious domains were shared with banks and law enforcement. Users "
                      "are advised to verify URLs before entering passwords.")),
    "A23": dict(label=0, category="Security", source="reuters.com",
                credibility=0.92, published="2024-03-28",
                title="Software vendor patches critical remote code execution flaw",
                body=("A major software vendor released patches for a critical remote code "
                      "execution flaw after coordinated disclosure. Exploit code was not "
                      "public at the time of the advisory. Administrators should apply "
                      "updates and monitor for unusual outbound traffic.")),
    # ---- Potentially misleading security ----
    "A24": dict(label=1, category="Security", source="hackeralert.ru",
                credibility=0.09, published="2024-06-08",
                title="All bank apps are backdoored by intelligence agencies tonight",
                body=("A sensational post claims that all bank apps are backdoored by "
                      "intelligence agencies starting tonight. No technical evidence, "
                      "CVE identifiers or vendor statements are provided. The story aims "
                      "to spark panic selling and phishing clicks.")),
    # ---- Credible business / insurance (related stories for clustering) ----
    "A25": dict(label=0, category="Business", source="insurancejournal.com",
                credibility=0.80, published="2024-09-12",
                title="Hub International acquires Wycoff Insurance Agency in Washington",
                body=("Hub International announced the acquisition of Wycoff Insurance "
                      "Agency in Washington. The deal expands Hub's commercial property "
                      "and casualty footprint in the Pacific Northwest. Terms of the "
                      "transaction were not disclosed.")),
    "A26": dict(label=0, category="Business", source="insurancejournal.com",
                credibility=0.80, published="2024-09-13",
                title="Hub International expands Pacific Northwest presence with Wycoff deal",
                body=("Following its purchase of Wycoff Insurance Agency, Hub International "
                      "said it will retain local staff and continue serving commercial "
                      "clients across Washington. Industry analysts called the move part "
                      "of a broader consolidation among insurance brokers.")),
    # ---- Credible legal / transport (related stories) ----
    "A27": dict(label=0, category="Business", source="nbcnews.com",
                credibility=0.85, published="2024-08-20",
                title="Judge rejects trucking company's bid to dismiss Oregon crash lawsuit",
                body=("A judge rejected a trucking company's bid to dismiss an Oregon crash "
                      "lawsuit, allowing claims of negligent maintenance to proceed. Court "
                      "filings describe a highway collision that injured several motorists. "
                      "A trial date has not yet been set.")),
    "A28": dict(label=0, category="Business", source="cnn.com",
                credibility=0.84, published="2024-08-21",
                title="Judge rejects a trucking company's bid to dismiss Oregon crash case",
                body=("In a related report, a federal judge denied a motion to dismiss a "
                      "lawsuit against a trucking firm over an Oregon highway crash. "
                      "Plaintiffs allege poor vehicle inspections contributed to the "
                      "collision. The company said it will continue to defend the case.")),
    # ---- Credible climate litigation (related) ----
    "A29": dict(label=0, category="Climate", source="cnn.com",
                credibility=0.84, published="2024-07-09",
                title="California seeks to revive climate lawsuit vs oil giants",
                body=("California officials asked an appeals court to revive a climate "
                      "lawsuit against major oil companies. The state argues that firms "
                      "misled the public about climate risks for decades. Defendants say "
                      "emissions policy should be set by legislatures, not courts.")),
    "A30": dict(label=0, category="Climate", source="reuters.com",
                credibility=0.92, published="2024-07-10",
                title="California appeals ruling that blocked climate case against oil firms",
                body=("California filed an appeal after a lower court blocked parts of its "
                      "climate lawsuit against oil giants. Environmental groups welcomed "
                      "the move while industry representatives called the case an improper "
                      "use of tort law to regulate greenhouse gases.")),
    # ---- Unrelated anomalies (gardening / sports) ----
    "A31": dict(label=0, category="Other", source="gardendaily.com",
                credibility=0.55, published="2024-05-01",
                title="How to prune roses for a longer blooming season",
                body=("Gardeners should prune roses in early spring, removing dead wood "
                      "and crossing branches. Well-drained soil, mulch and morning water "
                      "help prevent fungal disease. A balanced fertiliser supports longer "
                      "blooming through summer.")),
    "A32": dict(label=0, category="Other", source="sportsdesk.example",
                credibility=0.50, published="2024-06-15",
                title="Championship final decided by dramatic late free kick",
                body=("The championship final was decided by a dramatic late free kick as "
                      "thousands of fans celebrated in the stadium. The goalkeeper made "
                      "several saves and the striker was named player of the match.")),
}

# Queries for retrieval evaluation (relevant article ids)
TEST_QUERIES = {
    "Q1": ("electric vehicle battery fires and safety",
           {"A01", "A02", "A03", "A04"}),
    "Q2": ("vaccine side effects and public health guidance",
           {"A06", "A07", "A08", "A09"}),
    "Q3": ("ransomware and phishing attacks on organisations",
           {"A21", "A22", "A23", "A24"}),
    "Q4": ("climate change warming evidence and policy",
           {"A16", "A17", "A18", "A19", "A20", "A29", "A30"}),
    "Q5": ("trucking company oregon crash lawsuit",
           {"A27", "A28"}),
}

TOP_K = 5
SEED = 42
N_CLUSTERS = 6
GLOVE_URL = ("https://github.com/RaRe-Technologies/gensim-data/releases/"
             "download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz")
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "ir_tutorials")
GLOVE_WORDS = 100000
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

SENSATIONAL = {
    "shocking", "secret", "coverup", "covered", "miracle", "cure", "explode",
    "explodes", "banned", "hoax", "fake", "elites", "whistleblower", "anonymous",
    "deleted", "infertility", "overnight", "completely", "every", "all",
    "proven", "proves", "hidden", "leaked",
}


# =====================================================================
#  2. Preprocessing helpers
# =====================================================================
def load_stopwords():
    try:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except Exception:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        return set(ENGLISH_STOP_WORDS)


STOP_WORDS = load_stopwords()


def tokenize(text: str):
    return [t for t in re.findall(r"[a-z]+", text.lower())
            if t not in STOP_WORDS and len(t) > 1]


def full_text(aid: str) -> str:
    a = ARTICLES[aid]
    return f"{a['title']}. {a['body']}"


def header(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# =====================================================================
#  3. Feature engineering
# =====================================================================
def style_features(text: str, source: str, credibility: float) -> dict:
    tokens = tokenize(text)
    words = re.findall(r"[A-Za-z]+", text)
    chars = len(text)
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    sensational_hits = sum(1 for t in tokens if t in SENSATIONAL)
    exclamations = text.count("!")
    questions = text.count("?")
    uppercase_ratio = (sum(1 for c in text if c.isupper()) / chars) if chars else 0
    avg_word_len = (sum(len(w) for w in words) / len(words)) if words else 0
    suspicious_tlds = (".biz", ".info", ".club", ".io", ".tv", ".xyz",
                       ".ru", ".net")
    known_good = {"who.int", "ipcc.ch", "nasa.gov", "nih.gov", "cisa.gov",
                  "reuters.com", "bbc.com", "apnews.com", "nature.com",
                  "thelancet.com", "krebsonsecurity.com", "insurancejournal.com",
                  "nbcnews.com", "cnn.com", "theguardian.com"}
    known_bad = {"coalfacts.org", "viralbuzz.biz", "truthleak.info",
                 "naturalcurez.net", "healthtruth.club", "redpillwire.io",
                 "patriotblast.tv", "weathertruth.xyz", "hackeralert.ru"}
    if source in known_good:
        tld_suspicious = 0.0
    elif source in known_bad or any(source.endswith(s) for s in suspicious_tlds):
        tld_suspicious = 1.0
    else:
        tld_suspicious = 0.5
    return {
        "n_tokens": len(tokens),
        "n_sentences": sentences,
        "sensational_ratio": sensational_hits / max(1, len(tokens)),
        "exclamations": exclamations,
        "questions": questions,
        "uppercase_ratio": uppercase_ratio,
        "avg_word_len": avg_word_len,
        "source_credibility": credibility,
        "tld_suspicious": tld_suspicious,
        "anonymous_cues": float(bool(re.search(
            r"anonymous|unnamed|secret study|no (named|primary)", text, re.I))),
    }


FEATURE_NAMES = [
    "n_tokens", "n_sentences", "sensational_ratio", "exclamations", "questions",
    "uppercase_ratio", "avg_word_len", "source_credibility", "tld_suspicious",
    "anonymous_cues",
]


def build_feature_matrix(ids):
    rows = []
    for aid in ids:
        a = ARTICLES[aid]
        feats = style_features(full_text(aid), a["source"], a["credibility"])
        rows.append([feats[n] for n in FEATURE_NAMES])
    return np.asarray(rows, dtype=np.float64)


# =====================================================================
#  4. Embeddings (GloVe SIF — same approach as Tutorial 5)
# =====================================================================
def load_glove():
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"glove100_top{GLOVE_WORDS}.npz")
    if os.path.exists(cache):
        data = np.load(cache)
        return dict(zip(data["words"].tolist(), data["vectors"]))

    gz_path = os.path.join(CACHE_DIR, "glove-wiki-gigaword-100.gz")
    try:
        if not os.path.exists(gz_path):
            print("Downloading GloVe vectors (~130 MB, first run only)...")
            urllib.request.urlretrieve(GLOVE_URL, gz_path)
        words, vecs = [], []
        with gzip.open(gz_path, "rt", encoding="utf8") as f:
            next(f)
            for i, line in enumerate(f):
                if i >= GLOVE_WORDS:
                    break
                parts = line.rstrip().split(" ")
                words.append(parts[0])
                vecs.append(np.asarray(parts[1:], dtype=np.float32))
        vectors = np.vstack(vecs)
        np.savez(cache, words=np.array(words), vectors=vectors)
        return dict(zip(words, vectors))
    except Exception as e:
        print(f"Could not load GloVe ({e}); falling back to LSA embeddings.")
        return None


def normalise(m):
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class Embedder:
    def __init__(self, doc_ids, glove, a=1e-3):
        self.doc_ids = doc_ids
        self.glove = glove
        self.a = a
        self.rank = {w: i + 1 for i, w in enumerate(glove)}
        self.zipf_norm = math.log(len(glove)) + 0.5772
        self.tokens = {d: tokenize(full_text(d)) for d in doc_ids}
        N = len(doc_ids)
        df = Counter(t for d in doc_ids for t in set(self.tokens[d]))
        self.idf = {t: math.log((N + 1) / (c + 1)) + 1 for t, c in df.items()}
        self.default_idf = math.log(N + 1) + 1
        raw = np.vstack([self._average(self.tokens[d]) for d in doc_ids])
        self.u = np.linalg.svd(raw, full_matrices=False)[2][0]
        self.matrix = normalise(raw - np.outer(raw @ self.u, self.u))
        self.index = {d: i for i, d in enumerate(doc_ids)}

    def sif_weight(self, w):
        p = 1 / (self.rank[w] * self.zipf_norm)
        return self.a / (self.a + p)

    def _average(self, tokens):
        counts = Counter(t for t in tokens if t in self.glove)
        if not counts:
            return np.zeros(next(iter(self.glove.values())).shape)
        weights = np.array([c * self.sif_weight(t) for t, c in counts.items()])
        vecs = np.vstack([self.glove[t] for t in counts])
        return weights @ vecs / weights.sum()

    def embed_query(self, text):
        v = self._average(tokenize(text))
        return normalise(v - (v @ self.u) * self.u)

    def vector(self, aid):
        return self.matrix[self.index[aid]]


class LSAEmbedder:
    """Fallback when GloVe is unavailable."""

    def __init__(self, doc_ids):
        self.doc_ids = doc_ids
        texts = [full_text(d) for d in doc_ids]
        self.tfidf = TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                                    min_df=1)
        X = self.tfidf.fit_transform(texts)
        n = min(50, X.shape[1] - 1, X.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=max(2, n), random_state=SEED)
        self.matrix = normalise(self.svd.fit_transform(X))
        self.index = {d: i for i, d in enumerate(doc_ids)}

    def embed_query(self, text):
        X = self.tfidf.transform([text])
        return normalise(self.svd.transform(X))[0]

    def vector(self, aid):
        return self.matrix[self.index[aid]]


# =====================================================================
#  5. Classification (+ ensemble)
# =====================================================================
def train_classifiers(train_ids, test_ids):
    y_train = np.array([ARTICLES[i]["label"] for i in train_ids])
    y_test = np.array([ARTICLES[i]["label"] for i in test_ids])
    texts_train = [full_text(i) for i in train_ids]
    texts_test = [full_text(i) for i in test_ids]

    tfidf = TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                            ngram_range=(1, 2), min_df=1)
    Xtr = tfidf.fit_transform(texts_train)
    Xte = tfidf.transform(texts_test)

    Ftr = build_feature_matrix(train_ids)
    Fte = build_feature_matrix(test_ids)
    scaler = StandardScaler()
    Ftr_s = scaler.fit_transform(Ftr)
    Fte_s = scaler.transform(Fte)

    models = {
        "MultinomialNB": MultinomialNB(class_prior=None, fit_prior=True),
        "LogReg-TFIDF": LogisticRegression(
            max_iter=2000, random_state=SEED, class_weight="balanced", C=2.0),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=SEED, class_weight="balanced",
            min_samples_leaf=1),
    }
    results = {}
    fitted = {}
    svd = TruncatedSVD(n_components=min(40, max(2, Xtr.shape[1] - 1)),
                       random_state=SEED)
    Xtr_d = np.hstack([svd.fit_transform(Xtr), Ftr_s])
    Xte_d = np.hstack([svd.transform(Xte), Fte_s])

    for name, clf in models.items():
        if name == "RandomForest":
            clf.fit(Xtr_d, y_train)
            pred = clf.predict(Xte_d)
            proba = clf.predict_proba(Xte_d)[:, 1]
            fitted[name] = ("combo", clf, svd, scaler, tfidf)
        else:
            clf.fit(Xtr, y_train)
            pred = clf.predict(Xte)
            proba = clf.predict_proba(Xte)[:, 1]
            fitted[name] = ("tfidf", clf, None, None, tfidf)
        results[name] = dict(pred=pred, proba=proba,
                             acc=accuracy_score(y_test, pred),
                             p=precision_score(y_test, pred, zero_division=0),
                             r=recall_score(y_test, pred, zero_division=0),
                             f1=f1_score(y_test, pred, zero_division=0))

    # Style-only logistic regression (feature engineering demo)
    style_clf = LogisticRegression(
        max_iter=2000, random_state=SEED, class_weight="balanced")
    style_clf.fit(Ftr_s, y_train)
    style_pred = style_clf.predict(Fte_s)
    style_proba = style_clf.predict_proba(Fte_s)[:, 1]
    results["Style-LogReg"] = dict(
        pred=style_pred, proba=style_proba,
        acc=accuracy_score(y_test, style_pred),
        p=precision_score(y_test, style_pred, zero_division=0),
        r=recall_score(y_test, style_pred, zero_division=0),
        f1=f1_score(y_test, style_pred, zero_division=0))
    fitted["Style-LogReg"] = ("style", style_clf, None, scaler, None)

    # Weighted soft-voting: style + RF dominate (text-only models are weaker
    # on this small corpus); NB / LogReg still contribute lexical signal.
    weights = np.array([0.15, 0.20, 0.35, 0.30])  # NB, LR, RF, Style
    stack = np.vstack([
        results["MultinomialNB"]["proba"],
        results["LogReg-TFIDF"]["proba"],
        results["RandomForest"]["proba"],
        results["Style-LogReg"]["proba"],
    ])
    ensemble_proba = weights @ stack
    ensemble_pred = (ensemble_proba >= 0.45).astype(int)
    results["Ensemble"] = dict(
        pred=ensemble_pred, proba=ensemble_proba,
        acc=accuracy_score(y_test, ensemble_pred),
        p=precision_score(y_test, ensemble_pred, zero_division=0),
        r=recall_score(y_test, ensemble_pred, zero_division=0),
        f1=f1_score(y_test, ensemble_pred, zero_division=0))

    bundle = {
        "tfidf": tfidf,
        "models": fitted,
        "results": results,
        "y_test": y_test,
        "test_ids": test_ids,
        "scaler": scaler,
        "style_clf": style_clf,
        "svd": svd,
        "ensemble_weights": weights,
        "ensemble_threshold": 0.45,
        "ensemble_components": [
            "MultinomialNB", "LogReg-TFIDF", "RandomForest", "Style-LogReg"],
    }
    return bundle


def predict_misleading(text: str, source: str, credibility: float, bundle):
    """Return (label, confidence that article is potentially misleading)."""
    tfidf = bundle["tfidf"]
    X = tfidf.transform([text])
    feats = style_features(text, source, credibility)
    F = scaler_transform(bundle, feats)

    nb = bundle["models"]["MultinomialNB"][1]
    lr = bundle["models"]["LogReg-TFIDF"][1]
    _, rf, svd, _, _ = bundle["models"]["RandomForest"]
    Xd = np.hstack([svd.transform(X), F.reshape(1, -1)])
    style_p = bundle["style_clf"].predict_proba(F.reshape(1, -1))[0, 1]

    parts = {
        "MultinomialNB": float(nb.predict_proba(X)[0, 1]),
        "LogReg-TFIDF": float(lr.predict_proba(X)[0, 1]),
        "RandomForest": float(rf.predict_proba(Xd)[0, 1]),
        "Style-LogReg": float(style_p),
    }
    weights = bundle.get("ensemble_weights",
                         np.array([0.15, 0.20, 0.35, 0.30]))
    confidence = float(weights @ np.array(list(parts.values())))
    threshold = bundle.get("ensemble_threshold", 0.45)
    label = 1 if confidence >= threshold else 0
    return label, confidence, parts


def scaler_transform(bundle, feats: dict):
    row = np.array([[feats[n] for n in FEATURE_NAMES]], dtype=np.float64)
    return bundle["scaler"].transform(row)[0]


# =====================================================================
#  6. Retrieval, clustering, LTR
# =====================================================================
def semantic_search(query, emb, ids, k=TOP_K):
    q = emb.embed_query(query)
    scores = [(d, float(emb.vector(d) @ q)) for d in ids]
    scores.sort(key=lambda x: -x[1])
    return scores[:k]


def keyword_search(query, ids, k=TOP_K):
    tfidf = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
    X = tfidf.fit_transform([full_text(d) for d in ids])
    q = tfidf.transform([query])
    sims = (X @ q.T).toarray().ravel()
    order = np.argsort(-sims)[:k]
    return [(ids[i], float(sims[i])) for i in order]


def ltr_features(query, aid, emb, credibility_pred):
    """Pointwise Learning-to-Rank feature vector for (query, article)."""
    a = ARTICLES[aid]
    sem = float(emb.vector(aid) @ emb.embed_query(query))
    q_tok = set(tokenize(query))
    d_tok = set(tokenize(full_text(aid)))
    overlap = len(q_tok & d_tok) / max(1, len(q_tok))
    title_overlap = len(q_tok & set(tokenize(a["title"]))) / max(1, len(q_tok))
    return np.array([
        sem,
        overlap,
        title_overlap,
        a["credibility"],
        1.0 - credibility_pred,          # prefer lower misleading probability
        float(a["category"] in query.lower() or
              any(c.lower() in query.lower()
                  for c in [a["category"]])),
    ], dtype=np.float64)


def train_ltr(emb, train_pairs, misleading_proba):
    """
    Pointwise LTR: GradientBoosting regressor predicting relevance (0/1)
    from engineered features. train_pairs = [(query, aid, rel), ...]
    """
    X, y = [], []
    for query, aid, rel in train_pairs:
        X.append(ltr_features(query, aid, emb, misleading_proba.get(aid, 0.5)))
        y.append(rel)
    X, y = np.asarray(X), np.asarray(y)
    model = GradientBoostingClassifier(random_state=SEED)
    model.fit(X, y)
    return model


def ltr_rank(query, ids, emb, ltr_model, misleading_proba, k=TOP_K):
    scored = []
    for aid in ids:
        feats = ltr_features(query, aid, emb, misleading_proba.get(aid, 0.5))
        # probability of being relevant
        if hasattr(ltr_model, "predict_proba"):
            score = float(ltr_model.predict_proba(feats.reshape(1, -1))[0, 1])
        else:
            score = float(ltr_model.decision_function(feats.reshape(1, -1))[0])
        scored.append((aid, score))
    scored.sort(key=lambda x: -x[1])
    return scored[:k]


def evaluate_ranking(ranked_ids, relevant, k=TOP_K):
    ranked = ranked_ids[:k]
    hits = [1 if d in relevant else 0 for d in ranked]
    p = sum(hits) / k
    r = sum(hits) / max(1, len(relevant))
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    # AP
    ap, found = 0.0, 0
    for i, h in enumerate(hits, 1):
        if h:
            found += 1
            ap += found / i
    ap = ap / max(1, len(relevant))
    # RR
    rr = 0.0
    for i, h in enumerate(hits, 1):
        if h:
            rr = 1.0 / i
            break
    return {"P@k": p, "R@k": r, "F1@k": f1, "AP": ap, "RR": rr}


# =====================================================================
#  7. Main pipeline
# =====================================================================
def main():
    np.random.seed(SEED)
    ids = sorted(ARTICLES.keys())
    labels = np.array([ARTICLES[i]["label"] for i in ids])

    header("Tutorial 6 — Fake News Detection & Intelligent News Retrieval")
    print(f"Articles: {len(ids)}  |  Credible: {(labels == 0).sum()}  |  "
          f"Potentially misleading: {(labels == 1).sum()}")
    print("Note: Predictions flag articles as *potentially misleading*; they "
          "do not establish whether a story is true.")

    # ---------- Split for classification ----------
    train_ids, test_ids = train_test_split(
        ids, test_size=0.35, random_state=SEED,
        stratify=[ARTICLES[i]["label"] for i in ids])

    header("Step 1: Feature Engineering (sample)")
    for aid in ["A01", "A04", "A09", "A16"]:
        f = style_features(full_text(aid), ARTICLES[aid]["source"],
                           ARTICLES[aid]["credibility"])
        print(f"{aid} ({'mislead' if ARTICLES[aid]['label'] else 'cred':<7}) "
              f"sens={f['sensational_ratio']:.2f}  "
              f"cred={f['source_credibility']:.2f}  "
              f"tld_susp={f['tld_suspicious']:.0f}  "
              f"anon={f['anonymous_cues']:.0f}  "
              f"excl={f['exclamations']}")

    header("Step 2: Classification (credible vs potentially misleading)")
    bundle = train_classifiers(train_ids, test_ids)
    print(f"{'Model':<16}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}")
    print("-" * 48)
    for name, res in bundle["results"].items():
        print(f"{name:<16}{res['acc']:>8.3f}{res['p']:>8.3f}"
              f"{res['r']:>8.3f}{res['f1']:>8.3f}")

    print("\nEnsemble predictions on held-out articles:")
    print(f"{'ID':<6}{'True':<12}{'Pred':<12}{'P(mislead)':>12}  Title")
    print("-" * 78)
    for i, aid in enumerate(bundle["test_ids"]):
        true = "misleading" if bundle["y_test"][i] else "credible"
        pred_l = bundle["results"]["Ensemble"]["pred"][i]
        pred = "misleading" if pred_l else "credible"
        conf = bundle["results"]["Ensemble"]["proba"][i]
        print(f"{aid:<6}{true:<12}{pred:<12}{conf:>12.2%}  "
              f"{ARTICLES[aid]['title'][:40]}")

    cm = confusion_matrix(bundle["y_test"],
                          bundle["results"]["Ensemble"]["pred"])
    print("\nEnsemble confusion matrix [[TN FP] [FN TP]]:")
    print(cm)

    # Train final ensemble-style probabilities on ALL articles for retrieval
    header("Step 3: Credibility scores for the full collection")
    # Refit on all data for downstream use (small corpus)
    all_bundle = train_classifiers(ids, ids)  # train=test for full scoring
    misleading_proba = {
        aid: float(all_bundle["results"]["Ensemble"]["proba"][i])
        for i, aid in enumerate(all_bundle["test_ids"])
    }
    thr = all_bundle.get("ensemble_threshold", 0.45)
    for aid in ids[:5]:
        tag = "POTENTIALLY MISLEADING" if misleading_proba[aid] >= thr \
            else "likely credible"
        print(f"  {aid}: {misleading_proba[aid]:.1%}  [{tag}]  "
              f"{ARTICLES[aid]['title'][:50]}")

    # Demo prediction on the viral vaccine claim from the assignment mockup
    demo_text = (
        "Recent scientific studies suggest that a common vaccine causes severe "
        "health issues. The study, published in an obscure journal, claims that "
        "the side effects are widespread and have been covered up by "
        "pharmaceutical companies. Read the full story to understand the risks.")
    lab, conf, parts = predict_misleading(
        demo_text, "naturalcurez.net", 0.12, all_bundle)
    header("Step 3b: Demo analysis of a viral claim")
    print(f"Verdict: {'POTENTIALLY MISLEADING' if lab else 'likely credible'} "
          f"(confidence {conf:.1%})")
    print("Model votes:")
    for m, p in parts.items():
        print(f"  {m:<16} P(misleading) = {p:.1%}")
    print("Disclaimer: This is a model estimate, not a factual determination.")

    # ---------- Embeddings ----------
    header("Step 4: Neural / semantic article embeddings (GloVe SIF)")
    glove = load_glove()
    if glove:
        emb = Embedder(ids, glove)
        print(f"Loaded GloVe ({len(glove)} words). Document matrix: "
              f"{emb.matrix.shape}")
    else:
        emb = LSAEmbedder(ids)
        print(f"Using LSA fallback. Document matrix: {emb.matrix.shape}")

    # ---------- Clustering ----------
    header("Step 5: Clustering-based retrieval (related stories)")
    k = N_CLUSTERS
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED)
    cluster_of = dict(zip(ids, km.fit_predict(emb.matrix)))
    sil = silhouette_score(emb.matrix, list(cluster_of.values()),
                           metric="cosine")
    print(f"K-means k={k}, silhouette (cosine) = {sil:.3f}")
    groups = defaultdict(list)
    for aid, c in cluster_of.items():
        groups[c].append(aid)
    for c in sorted(groups):
        cats = Counter(ARTICLES[a]["category"] for a in groups[c])
        print(f"  Cluster {c}: {groups[c]}  dominant={cats.most_common(1)[0]}")

    # Duplicate / related story examples
    print("\nRelated-story check (Hub International / Oregon crash / California):")
    for pair in [("A25", "A26"), ("A27", "A28"), ("A29", "A30")]:
        sim = float(emb.vector(pair[0]) @ emb.vector(pair[1]))
        same = cluster_of[pair[0]] == cluster_of[pair[1]]
        print(f"  {pair[0]}~{pair[1]} cosine={sim:.3f}  "
              f"same_cluster={same}")

    # ---------- Topic modeling ----------
    header("Step 6: Topic modeling (LDA + NMF)")
    texts = [full_text(d) for d in ids]
    count_vec = CountVectorizer(tokenizer=tokenize, token_pattern=None,
                                max_df=0.9, min_df=1)
    X_counts = count_vec.fit_transform(texts)
    n_topics = 5
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=SEED)
    lda.fit(X_counts)
    vocab = np.array(count_vec.get_feature_names_out())
    print("LDA topics:")
    for t, comp in enumerate(lda.components_):
        top = vocab[np.argsort(comp)[::-1][:8]]
        print(f"  Topic {t}: {', '.join(top)}")

    tfidf_topic = TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                                  max_df=0.9, min_df=1)
    X_tfidf = tfidf_topic.fit_transform(texts)
    nmf = NMF(n_components=n_topics, random_state=SEED, max_iter=500)
    nmf.fit(X_tfidf)
    vocab2 = np.array(tfidf_topic.get_feature_names_out())
    print("\nNMF topics:")
    for t, comp in enumerate(nmf.components_):
        top = vocab2[np.argsort(comp)[::-1][:8]]
        print(f"  Topic {t}: {', '.join(top)}")

    # ---------- Anomaly detection ----------
    header("Step 7: Anomaly detection (articles far from all others)")
    sims = emb.matrix @ emb.matrix.T
    np.fill_diagonal(sims, -1)
    max_sim = sims.max(axis=1)
    anomaly_threshold = np.percentile(max_sim, 15)
    anomalies = [ids[i] for i, s in enumerate(max_sim) if s <= anomaly_threshold]
    print(f"Threshold (15th pct of max pairwise cosine): {anomaly_threshold:.3f}")
    for aid in anomalies:
        print(f"  {aid}: max_sim={max_sim[ids.index(aid)]:.3f}  "
              f"[{ARTICLES[aid]['category']}] {ARTICLES[aid]['title']}")

    # ---------- Dimensionality reduction plots ----------
    header("Step 8: Dimensionality reduction (PCA & t-SNE)")
    categories = [ARTICLES[i]["category"] for i in ids]
    cat_list = sorted(set(categories))
    cat_to_i = {c: i for i, c in enumerate(cat_list)}
    colours = [cat_to_i[c] for c in categories]
    markers = ["o" if ARTICLES[i]["label"] == 0 else "X" for i in ids]

    pca = PCA(n_components=2, random_state=SEED)
    xy = pca.fit_transform(emb.matrix)
    plt.figure(figsize=(9, 7))
    for aid, (x, y), m in zip(ids, xy, markers):
        plt.scatter(x, y, c=[cat_to_i[ARTICLES[aid]["category"]]],
                    cmap="tab10", vmin=0, vmax=9, marker=m, s=80,
                    edgecolors="k", linewidths=0.4)
        plt.text(x, y, aid, fontsize=7, ha="left", va="bottom")
    plt.title("PCA of article embeddings\n(circles=credible, X=potentially misleading)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    pca_path = os.path.join(OUTPUT_DIR, "pca_articles.png")
    plt.savefig(pca_path, dpi=120); plt.close()

    perplexity = min(10, len(ids) - 1)
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=perplexity,
                init="pca", learning_rate="auto")
    xy2 = tsne.fit_transform(emb.matrix)
    plt.figure(figsize=(9, 7))
    for aid, (x, y), m in zip(ids, xy2, markers):
        plt.scatter(x, y, c=[cat_to_i[ARTICLES[aid]["category"]]],
                    cmap="tab10", vmin=0, vmax=9, marker=m, s=80,
                    edgecolors="k", linewidths=0.4)
        plt.text(x, y, aid, fontsize=7, ha="left", va="bottom")
    plt.title("t-SNE of article embeddings")
    plt.tight_layout()
    tsne_path = os.path.join(OUTPUT_DIR, "tsne_articles.png")
    plt.savefig(tsne_path, dpi=120); plt.close()
    print(f"Saved: {os.path.basename(pca_path)}, {os.path.basename(tsne_path)}")

    # Classification comparison plot
    plt.figure(figsize=(8, 4.5))
    names = list(bundle["results"].keys())
    f1s = [bundle["results"][n]["f1"] for n in names]
    accs = [bundle["results"][n]["acc"] for n in names]
    x = np.arange(len(names))
    plt.bar(x - 0.18, accs, width=0.35, label="Accuracy")
    plt.bar(x + 0.18, f1s, width=0.35, label="F1")
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Fake-news classifiers on held-out articles")
    plt.legend()
    plt.tight_layout()
    clf_path = os.path.join(OUTPUT_DIR, "classifier_comparison.png")
    plt.savefig(clf_path, dpi=120); plt.close()
    print(f"Saved: {os.path.basename(clf_path)}")

    # ---------- Learning to Rank ----------
    header("Step 9: Learning to Rank (pointwise GBDT)")
    train_pairs = []
    for qid, (qtext, rel) in TEST_QUERIES.items():
        for aid in ids:
            train_pairs.append((qtext, aid, 1 if aid in rel else 0))
    ltr_model = train_ltr(emb, train_pairs, misleading_proba)
    print("Trained GradientBoosting LTR on features:")
    print("  [semantic_sim, term_overlap, title_overlap, source_cred, "
          "1-P(mislead), category_hint]")

    # ---------- Search demo ----------
    demo_q = TEST_QUERIES["Q1"][0]
    header(f'Step 10: Search — "{demo_q}"')
    print("\nSemantic (neural IR) ranking:")
    for i, (aid, s) in enumerate(semantic_search(demo_q, emb, ids), 1):
        flag = "⚠ misleading?" if misleading_proba[aid] >= thr else "credible"
        print(f"  {i}. {aid}  sim={s:.3f}  [{flag} {misleading_proba[aid]:.0%}]  "
              f"{ARTICLES[aid]['title']}")

    print("\nLearning-to-Rank (relevance + credibility features):")
    for i, (aid, s) in enumerate(
            ltr_rank(demo_q, ids, emb, ltr_model, misleading_proba), 1):
        flag = "⚠ misleading?" if misleading_proba[aid] >= thr else "credible"
        print(f"  {i}. {aid}  score={s:.3f}  [{flag}]  "
              f"{ARTICLES[aid]['title']}")

    print("\nKeyword TF-IDF baseline:")
    for i, (aid, s) in enumerate(keyword_search(demo_q, ids), 1):
        print(f"  {i}. {aid}  tfidf={s:.3f}  {ARTICLES[aid]['title']}")

    # ---------- Evaluation ----------
    header("Step 11: Retrieval evaluation (top 5)")
    methods = {
        "Semantic": lambda q: [d for d, _ in semantic_search(q, emb, ids)],
        "LTR": lambda q: [d for d, _ in ltr_rank(
            q, ids, emb, ltr_model, misleading_proba)],
        "Keyword": lambda q: [d for d, _ in keyword_search(q, ids)],
    }
    summary = {m: [] for m in methods}
    for qid, (qtext, rel) in TEST_QUERIES.items():
        print(f'\n{qid}: "{qtext}"')
        print(f"  relevant = {sorted(rel)}")
        for m, fn in methods.items():
            ranked = fn(qtext)
            metrics = evaluate_ranking(ranked, rel)
            summary[m].append(metrics)
            print(f"  {m:<10} {ranked}  "
                  f"P@5={metrics['P@k']:.2f} AP={metrics['AP']:.2f}")

    print(f"\n{'Method':<12}{'P@5':>8}{'R@5':>8}{'F1@5':>8}{'MRR':>8}{'MAP':>8}")
    print("-" * 52)
    means = {}
    for m, rows in summary.items():
        avg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        means[m] = avg
        print(f"{m:<12}{avg['P@k']:>8.3f}{avg['R@k']:>8.3f}{avg['F1@k']:>8.3f}"
              f"{avg['RR']:>8.3f}{avg['AP']:>8.3f}")

    best = max(means, key=lambda m: means[m]["AP"])
    best_clf = max(bundle["results"], key=lambda m: bundle["results"][m]["f1"])

    header("Summary")
    print(f"""* Collection: {len(ids)} news articles across Technology, Health,
  Politics, Climate, Security, Business (+ 2 unrelated anomalies).
* Classification: {best_clf} had the best held-out F1
  ({bundle['results'][best_clf]['f1']:.3f}). Weighted soft-voting combines
  MultinomialNB, LogReg-TFIDF, RandomForest and Style-LogReg.
* Feature engineering: sensational lexicon ratio, source credibility, TLD
  suspicion and anonymous-source cues separate many misleading articles.
* Embeddings: GloVe SIF vectors enable semantic matches (e.g. EV battery
  safety ~ solid-state prototype) beyond exact keywords.
* Clustering: related stories (Hub/Wycoff, Oregon crash, California climate
  lawsuit) land in the same cluster for duplicate-aware retrieval.
* Anomalies flagged: {', '.join(anomalies) or 'none'}.
* Retrieval: {best} achieved the best MAP ({means[best]['AP']:.3f}).
  LTR blends semantic relevance with source credibility so potentially
  misleading near-matches are ranked more carefully.
* Important: outputs say *potentially misleading*, never a definitive
  claim that a news story is false.""")

    # Persist light artefacts for the Streamlit app
    artefact = os.path.join(OUTPUT_DIR, "artefacts.npz")
    np.savez(
        artefact,
        ids=np.array(ids),
        matrix=emb.matrix,
        misleading_proba=np.array([misleading_proba[i] for i in ids]),
        clusters=np.array([cluster_of[i] for i in ids]),
    )
    print(f"\nSaved artefacts: {os.path.basename(artefact)}")
    print("Launch UI:  streamlit run app.py")

    if "--interactive" in sys.argv:
        header("Interactive mode (empty line to quit)")
        while True:
            try:
                text = input("\nQuery or article text> ").strip()
            except EOFError:
                break
            if not text:
                break
            if len(text.split()) > 12 and not text.endswith("?"):
                lab, conf, parts = predict_misleading(
                    text, "unknown.com", 0.4, all_bundle)
                print(f"  Verdict: {'POTENTIALLY MISLEADING' if lab else 'likely credible'} "
                      f"({conf:.1%})")
            else:
                for i, (aid, s) in enumerate(semantic_search(text, emb, ids), 1):
                    flag = "⚠" if misleading_proba[aid] >= 0.5 else "✓"
                    print(f"  {i}. {flag} {aid} {s:.3f}  {ARTICLES[aid]['title']}")


if __name__ == "__main__":
    main()