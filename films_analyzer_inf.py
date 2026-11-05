#!/usr/bin/env python3
"""
Film Director Analyser - Claude Code (cloud session) Edition
============================================================
SCRIPT_VERSION 2026-11-05.1-film-claude-code   (port of the film HuggingFace edition v7.0-hf)

  8 query modes  x  15 strategies (S1-S9, S11-S15) + all  x  7 RAG tiers

Question under test (per target film):
  Which directors have EVER won the Academy Award for Best Director OR the Palme d'Or, AND directed at least one film
  released in the same calendar year as <target film>? For each: birthplace and the population of that birthplace.

SCOPE
  Claude models ONLY (8 aliases). Every call is a streamed headless Claude Code invocation (`claude -p`) made from
  inside a Claude Code cloud session. No HuggingFace, Bedrock, API key or GPU. Prompts, modes, strategies, RAG tiers,
  evidence stores and the film-domain year quirk are those of the HF script: the source file has NO year column, so the
  release year every RAG lookup keys on is resolved from GROUND_TRUTH_FILE by S.No (also for tiers 1-3).

CALL CONFIGURATION
  claude -p --model <id> --output-format stream-json --verbose --include-partial-messages --tools "" --max-turns 1
         [--system-prompt <neutral text>]        (--system_mode neutral, default)
  prompt via stdin; cwd = empty temp dir; output cap via CLAUDE_CODE_MAX_OUTPUT_TOKENS; optional MAX_THINKING_TOKENS.
  temperature / top_p are NOT controllable. --system_mode default: no flag, Claude Code's own system prompt applies.

EVIDENCE ON DISK (nothing is discarded)
  <cell dir>/raw_calls/<label>_try<n>_prompt.txt | _events.jsonl | _text.txt | _stderr.txt
  <output root>/run.log
  A call that hits the output-token cap is NOT retried: it is stored as a truncated record with its partial text.
  Generated_Response is the model text exactly as returned; Self-Ask trimming only affects parsing. --reparse is offline.

Output roots:  ./claude_code_film_strategies (non-RAG)   ./rag_claude_code_film_strategies (RAG)
"""
import os
import re
import sys
import json
import math
import time
import atexit
import shutil
import logging
import tempfile
import argparse
import threading
import traceback
import subprocess
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

import numpy as np
import pandas as pd

SCRIPT_VERSION = "2026-11-05.1-film-claude-code"

# ---------------------------------------------------------------------------
# CONSTANTS - Claude models only
# ---------------------------------------------------------------------------
FAMILY = "claude"
MODEL_REGISTRY: Dict[str, str] = {
    "claude-opus-4-8":   "claude-opus-4-8",
    "claude-opus-4-7":   "claude-opus-4-7",
    "claude-opus-4-6":   "claude-opus-4-6",
    "claude-opus-4-5":   "claude-opus-4-5-20251101",
    "claude-opus-4-1":   "claude-opus-4-1-20250805",
    "claude-haiku-4-5":  "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
}
MODEL_SIZE_LABEL: Dict[str, str] = {
    "claude-opus-4-8": "opus-4-8", "claude-opus-4-7": "opus-4-7", "claude-opus-4-6": "opus-4-6",
    "claude-opus-4-5": "opus-4-5", "claude-opus-4-1": "opus-4-1", "claude-haiku-4-5": "haiku-4-5",
    "claude-sonnet-4-6": "sonnet-4-6", "claude-sonnet-4-5": "sonnet-4-5",
}
DEFAULT_MODEL_ALIAS = "claude-sonnet-4-6"

SOURCE_FILE       = "source_books_films.csv"        # query list; NO release-year column
REFERENCE_FILE    = "directors_films_by_year.csv"   # RAG tiers 1-3
GROUND_TRUTH_FILE = "df_gt_films.csv"               # RAG tiers 4-6 + year resolution

OUTPUT_ROOT     = Path("./claude_code_film_strategies")
RAG_OUTPUT_ROOT = Path("./rag_claude_code_film_strategies")
ANALYSIS_DIR_NAME = "analysis"

GENERATION_MAX_NEW_TOKENS = 3000      # output-token cap per call (raise with --max_new_tokens)
MAX_RETRIES    = 3
RETRY_BASE_SEC = 5.0
REQUEST_TIMEOUT = 1800
MAX_CONSECUTIVE_BAD = 3               # a cell stops after this many consecutive errors / token-cap hits
PROGRESS_LOG_EVERY_SEC = 30
NEUTRAL_SYSTEM = "You are a helpful assistant. Answer the user's message directly in plain text."

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Eight query modes
# ---------------------------------------------------------------------------
SUPPORTED_MODES = ["nlq_general", "nlq_wikipedia", "nlq_cot_general", "nlq_cot_wikipedia",
                   "sparql_dbpedia", "sparql_cot_dbpedia", "sparql_wikidata", "sparql_cot_wikidata"]
MODE_SOURCE_COLUMN = {
    "nlq_general": "Film Title", "nlq_wikipedia": "Film Title",
    "nlq_cot_general": "Film Title", "nlq_cot_wikipedia": "Film Title",
    "sparql_dbpedia": "SPARQL URI Identifier", "sparql_cot_dbpedia": "SPARQL URI Identifier",
    "sparql_wikidata": "Wikidata QID", "sparql_cot_wikidata": "Wikidata QID",
}
MODE_SUFFIX = {
    "nlq_general": "NLQ_GENERAL", "nlq_wikipedia": "NLQ_WIKIPEDIA",
    "nlq_cot_general": "NLQ_COT_GENERAL", "nlq_cot_wikipedia": "NLQ_COT_WIKIPEDIA",
    "sparql_dbpedia": "SPARQL_DBPEDIA", "sparql_cot_dbpedia": "SPARQL_COT_DBPEDIA",
    "sparql_wikidata": "SPARQL_WIKIDATA", "sparql_cot_wikidata": "SPARQL_COT_WIKIDATA",
}
WIKIDATA_MODES  = {"sparql_wikidata", "sparql_cot_wikidata"}
DBPEDIA_MODES   = {"sparql_dbpedia", "sparql_cot_dbpedia"}
SPARQL_MODES    = WIKIDATA_MODES | DBPEDIA_MODES
COT_MODES       = {"nlq_cot_general", "nlq_cot_wikipedia", "sparql_cot_dbpedia", "sparql_cot_wikidata"}
WIKIPEDIA_MODES = {"nlq_wikipedia", "nlq_cot_wikipedia"}

# ---------------------------------------------------------------------------
# Fifteen strategies
# ---------------------------------------------------------------------------
STRATEGY_FLAT_CONJUNCTIVE = "flat_conjunctive"
STRATEGY_SEQUENTIAL_CHAIN = "sequential_chain"
STRATEGY_VARIABLE_BINDING = "variable_binding"
STRATEGY_ENTITY_FIRST     = "entity_first"
STRATEGY_CONSTRAINT_FIRST = "constraint_first"
STRATEGY_PAIRWISE_EDGES   = "pairwise_edges"
STRATEGY_CONTRASTIVE      = "contrastive_constraint"
STRATEGY_SELF_CHECK       = "self_check"
STRATEGY_MATERIALIZATION  = "materialization_checkpoint"
STRATEGY_TEMPORAL_SCOPE   = "temporal_scope_disambiguation"
STRATEGY_SELF_ASK         = "self_ask"
STRATEGY_FACTORIZATION    = "factorization"
STRATEGY_PROB_CHAINING    = "probabilistic_chaining"
STRATEGY_BACKWARD_GOAL    = "backward_goal_decomposition"
STRATEGY_DUAL_PATH        = "dual_path_consistency"
STRATEGY_ALL              = "all"

SUPPORTED_STRATEGIES: List[str] = [
    STRATEGY_FLAT_CONJUNCTIVE, STRATEGY_SEQUENTIAL_CHAIN, STRATEGY_VARIABLE_BINDING,
    STRATEGY_ENTITY_FIRST, STRATEGY_CONSTRAINT_FIRST, STRATEGY_PAIRWISE_EDGES,
    STRATEGY_CONTRASTIVE, STRATEGY_SELF_CHECK, STRATEGY_MATERIALIZATION,
    STRATEGY_TEMPORAL_SCOPE, STRATEGY_SELF_ASK, STRATEGY_FACTORIZATION,
    STRATEGY_PROB_CHAINING, STRATEGY_BACKWARD_GOAL, STRATEGY_DUAL_PATH, STRATEGY_ALL,
]
INDIVIDUAL_STRATEGIES: List[str] = [s for s in SUPPORTED_STRATEGIES if s != STRATEGY_ALL]
STRATEGY_SUFFIX: Dict[str, str] = {
    STRATEGY_FLAT_CONJUNCTIVE: "S1_FLAT", STRATEGY_SEQUENTIAL_CHAIN: "S2_SEQ", STRATEGY_VARIABLE_BINDING: "S3_VAR",
    STRATEGY_ENTITY_FIRST: "S4A_ENT", STRATEGY_CONSTRAINT_FIRST: "S4B_CON", STRATEGY_PAIRWISE_EDGES: "S5_PAIR",
    STRATEGY_CONTRASTIVE: "S6_CNEG", STRATEGY_SELF_CHECK: "S7_CHK", STRATEGY_MATERIALIZATION: "S8_CKPT",
    STRATEGY_TEMPORAL_SCOPE: "S9_TSCOPE", STRATEGY_SELF_ASK: "S11_SASK", STRATEGY_FACTORIZATION: "S12_FACT",
    STRATEGY_PROB_CHAINING: "S13_PCHAIN", STRATEGY_BACKWARD_GOAL: "S14_BGOAL", STRATEGY_DUAL_PATH: "S15_DPATH",
}
STRATEGY_CODE: Dict[str, str] = {s: STRATEGY_SUFFIX[s].split("_")[0] for s in INDIVIDUAL_STRATEGIES}

STRATEGY_HINT: Dict[str, str] = {
    STRATEGY_FLAT_CONJUNCTIVE: "Apply all conditions simultaneously in a single pass.",
    STRATEGY_SEQUENTIAL_CHAIN: "Work through release year → co-year films → directors → award filter → birthplace → population in order.",
    STRATEGY_VARIABLE_BINDING: "Bind: year=release_year(film), co_directors=directors(films(year)), award_set={d in co_directors | won Academy Award for Best Director or Palme d'Or}, then report birthplace+population.",
    STRATEGY_ENTITY_FIRST:     "Start from the film, expand outward: film→year→co-year films→directors→award filter→birthplace→population.",
    STRATEGY_CONSTRAINT_FIRST: "Start from Academy Award for Best Director / Palme d'Or winners, filter to those with a co-year film, then report birthplace and population.",
    STRATEGY_PAIRWISE_EDGES:   "Establish each relation independently (director-film, film-year, director-award, person-birthplace, place-population) then compose.",
    STRATEGY_CONTRASTIVE:      "Include directors satisfying BOTH award AND co-year conditions; exclude any missing either.",
    STRATEGY_SELF_CHECK:       "After each reasoning step, verify the result before proceeding. Write 'unknown' for any unverified fact.",
    STRATEGY_MATERIALIZATION:  "Materialize intermediate results (especially release year) as checkpoints; use each checkpoint to anchor downstream reasoning.",
    STRATEGY_TEMPORAL_SCOPE:   "Separate two temporal scopes: (A) year-specific (films released in target year) vs. (B) career-wide (award in any year). Keep both in parallel.",
    STRATEGY_SELF_ASK:         "Use Self-Ask decomposition: explicitly ask and answer follow-up sub-questions before finalizing, then converge to a single self-contained answer.",
    STRATEGY_FACTORIZATION:    "Factorize the question into independent atomic factors (year, co-year films, award winners, birthplace, population); solve each factor independently, then combine via explicit logical AND.",
    STRATEGY_PROB_CHAINING:    "Decompose into ordered units; assign each unit a confidence score in [0,1] and propagate a joint confidence via the probability chain rule instead of a hard AND filter; keep only directors above a 0.5 joint-confidence threshold.",
    STRATEGY_BACKWARD_GOAL:    "Work backward from the required output line to its sub-goals until reaching a base case with no further dependencies (release year), then resolve forward from that base case.",
    STRATEGY_DUAL_PATH:        "Solve via two independent decomposition paths (year-first and award-first) and cross-check that both yield the same director set before finalizing the answer.",
}

# ---------------------------------------------------------------------------
# RAG evidence tiers
# ---------------------------------------------------------------------------
RAG_TIER_NO_EVIDENCE            = "no_evidence"
RAG_TIER_YEAR_LIST_ONLY         = "year_list_only"
RAG_TIER_TITLES_AUTHORS         = "titles_authors"
RAG_TIER_TITLES_AUTHORS_PRIZE   = "titles_authors_prize"
RAG_TIER_FULL_WITH_BIRTHPLACE   = "full_with_birthplace"
RAG_TIER_FULL_WITH_POPULATION   = "full_with_population"
RAG_TIER_FULL_WITH_DISTRACTORS  = "full_with_distractors"
RAG_TIERS: List[str] = [
    RAG_TIER_NO_EVIDENCE, RAG_TIER_YEAR_LIST_ONLY, RAG_TIER_TITLES_AUTHORS, RAG_TIER_TITLES_AUTHORS_PRIZE,
    RAG_TIER_FULL_WITH_BIRTHPLACE, RAG_TIER_FULL_WITH_POPULATION, RAG_TIER_FULL_WITH_DISTRACTORS,
]
REF_TIERS = (RAG_TIER_YEAR_LIST_ONLY, RAG_TIER_TITLES_AUTHORS, RAG_TIER_TITLES_AUTHORS_PRIZE)
GT_TIERS = (RAG_TIER_FULL_WITH_BIRTHPLACE, RAG_TIER_FULL_WITH_POPULATION, RAG_TIER_FULL_WITH_DISTRACTORS)
RAG_TIER_SUFFIX: Dict[str, str] = {
    RAG_TIER_NO_EVIDENCE: "RAG0_NONE", RAG_TIER_YEAR_LIST_ONLY: "RAG1_TITLES",
    RAG_TIER_TITLES_AUTHORS: "RAG2_DIRECTORS", RAG_TIER_TITLES_AUTHORS_PRIZE: "RAG3_AWARD",
    RAG_TIER_FULL_WITH_BIRTHPLACE: "RAG4_BIRTHPLACE", RAG_TIER_FULL_WITH_POPULATION: "RAG5_POPULATION",
    RAG_TIER_FULL_WITH_DISTRACTORS: "RAG6_DISTRACTORS",
}
RAG_TIER_DESCRIPTION: Dict[str, str] = {
    RAG_TIER_NO_EVIDENCE:           "No evidence — knowledge retrieval / reasoning from parametric memory.",
    RAG_TIER_YEAR_LIST_ONLY:        "Co-year title list only (REFERENCE_FILE) — candidate identification + knowledge retrieval.",
    RAG_TIER_TITLES_AUTHORS:        "Titles + directors (REFERENCE_FILE) — entity linking + further knowledge retrieval.",
    RAG_TIER_TITLES_AUTHORS_PRIZE:  "Titles + directors + award (REFERENCE_FILE) — filtering + further retrieval.",
    RAG_TIER_FULL_WITH_BIRTHPLACE:  "Candidate directors + birthplace, no population (GROUND_TRUTH_FILE) — multi-hop comprehension + population retrieval.",
    RAG_TIER_FULL_WITH_POPULATION:  "Complete evidence including population (GROUND_TRUTH_FILE) — reading comprehension + multi-hop synthesis.",
    RAG_TIER_FULL_WITH_DISTRACTORS: "Complete evidence + distractors (GROUND_TRUTH_FILE) — comprehension + relevance filtering + synthesis.",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def model_name_to_slug(model_name: str) -> str:
    return model_name.replace(":", "_").replace(".", "_")


def _resolve_path(filename: str) -> Path:
    for c in [Path(filename), Path(__file__).resolve().parent / filename, Path.cwd() / filename]:
        if c.exists():
            return c
    raise FileNotFoundError(f"'{filename}' not found in cwd or script directory.")


def _read_tabular(path: Path) -> pd.DataFrame:
    if str(path).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(str(path))
    return pd.read_csv(str(path))


def clean_year(raw: Any) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "unknown", ""):
        return ""
    m = re.match(r"^(\d{3,4})(?:\.0+)?$", s)
    if m:
        return m.group(1).zfill(4)
    try:
        f = float(s)
        if math.isnan(f):
            return ""
        return f"{int(f):04d}"
    except (ValueError, TypeError):
        return ""


_SENTINELS = {"no result", "unknown", "na", "none", "error", "no qualifying director"}


def is_sentinel_director(s: str) -> bool:
    n = " ".join(re.sub(r"[^a-z0-9]+", " ", str(s).lower()).split())
    return n in _SENTINELS or n.endswith(" error")


def is_error_row(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]+_ERROR", str(s).strip()))


def _ascii_fold(s: Any) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


def name_key(s: Any) -> str:
    t = re.sub(r"\([^)]*\)", " ", str(s))
    t = re.sub(r"[^a-z0-9 ]+", " ", _ascii_fold(t).lower())
    return " ".join(w for w in t.split() if w not in {"jr", "sr", "ii", "iii"})


def names_match(a: Any, b: Any) -> bool:
    ka, kb = name_key(a), name_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    ta, tb = set(ka.split()), set(kb.split())
    small, large = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return small <= large and (len(large) - len(small)) <= 2


def place_key(s: Any) -> str:
    first = str(s).split(",")[0]
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", _ascii_fold(first).lower()).split())


def places_match(a: Any, b: Any) -> bool:
    ka, kb = place_key(a), place_key(b)
    return bool(ka and kb and (ka == kb or ka in kb or kb in ka))


# ---- population parsing (float-first for machine numbers; unit-aware for model text) ----
_UNIT = {"billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "millions": 1e6, "mn": 1e6, "mm": 1e6, "m": 1e6,
         "thousand": 1e3, "k": 1e3, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "crore": 1e7, "cr": 1e7, "hundred": 100.0}
_UNIT_RE = r"billion|bn|millions?|mn|mm|thousand|lakhs?|lac|crore|cr|hundred|[bmk]"
_NUM_RE = re.compile(
    r"(?<![\w.])("
    r"\d{1,3}(?:[,\u00a0\u202f ]\d{3})+(?:\.\d+)?"
    r"|\d{1,3}(?:\.\d{3})+(?!\d)"
    r"|\d+(?:[.,]\d+)?"
    r")\s*(" + _UNIT_RE + r")?(?![a-z])", re.I)
_RANGE_RE = re.compile(r"(\d[\d.,]*)\s*(?:-|\u2013|\u2014|to)\s*(\d[\d.,]*)\s*(" + _UNIT_RE + r")(?![a-z])", re.I)
_SCI_RE = re.compile(r"[+-]?\d+(?:\.\d+)?[eE][+-]?\d+")
_WORD_NUM_RE = re.compile(r"\b(?:one|a)\s+(billion|million|thousand)\b", re.I)


def _to_float(tok: str, has_unit: bool) -> float:
    t = tok.replace("\u00a0", " ").replace("\u202f", " ").strip()
    if re.fullmatch(r"\d{1,3}(?:[, ]\d{3})+(?:\.\d+)?", t):
        return float(re.sub(r"[, ]", "", t))
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", t):
        return float(t) if (has_unit and t.count(".") == 1) else float(t.replace(".", ""))
    if re.fullmatch(r"\d+,\d{1,2}", t):
        return float(t.replace(",", "."))
    return float(t.replace(",", ""))


def parse_population_value(text: Any) -> Optional[float]:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    if _SCI_RE.fullmatch(s):
        return float(s)
    if not re.search(r"\d", s):
        m = _WORD_NUM_RE.search(s)
        return _UNIT[m.group(1).lower()] if m else None
    rng = _RANGE_RE.search(s)
    if rng:
        try:
            a, b = _to_float(rng.group(1), True), _to_float(rng.group(2), True)
            return (a + b) / 2.0 * _UNIT[rng.group(3).lower()]
        except ValueError:
            pass
    for m in _NUM_RE.finditer(s):
        tok, unit = m.group(1), (m.group(2) or "").lower()
        if not unit and re.fullmatch(r"(?:1[5-9]|20)\d{2}", tok) and \
                re.search(r"(?:as of|in|census|est\.?|\()\s*$", s[:m.start()].lower()):
            continue
        try:
            return _to_float(tok, bool(unit)) * _UNIT.get(unit, 1.0)
        except ValueError:
            continue
    return None


def pop_to_str(raw: Any) -> str:
    v = parse_population_value(raw)
    return str(int(round(v))) if v is not None else ""


def gt_pop_value(raw: Any) -> Optional[int]:
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    try:
        f = float(s.replace(",", ""))
        if math.isfinite(f):
            return int(round(f))
    except ValueError:
        pass
    v = parse_population_value(s)
    return int(round(v)) if v is not None else None


def gt_pop_to_str(raw: Any) -> str:
    v = gt_pop_value(raw)
    return str(v) if v is not None else ""


ST_OK, ST_TRUNC_LENGTH = "ok", "truncated_length"
ST_TRUNC_REASONING, ST_EMPTY = "truncated_in_reasoning", "empty"


# ---------------------------------------------------------------------------
# RAG evidence stores
# ---------------------------------------------------------------------------
@dataclass
class ReferenceRecord:
    year: str
    title: str
    director: str
    prize: str
    title_uri: str = ""
    director_uri: str = ""


@dataclass
class GroundTruthRecord:
    sno: str
    film_title: str
    uri: str
    award_type: str
    year: str
    director: str
    birthplace: str
    popcount: str
    award_category: str = ""


class ReferenceStore:
    """REFERENCE_FILE (Year,Prize,Title,Director,TitleURI,DirectorURI). Backs RAG tiers 1-3."""
    REQUIRED_COLUMNS = {"Year", "Prize", "Title", "Director"}

    def __init__(self, path: str = REFERENCE_FILE):
        self.path = path
        self.by_year: Dict[str, List[ReferenceRecord]] = {}
        self._load()

    def _load(self):
        candidate = _resolve_path(self.path)
        df = _read_tabular(candidate)
        df.columns = [c.strip() for c in df.columns]
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"'{candidate}' missing columns {sorted(missing)}. Found: {list(df.columns)}")
        df = df.fillna("")
        count = 0
        for _, row in df.iterrows():
            year = clean_year(row.get("Year", ""))
            if not year:
                continue
            self.by_year.setdefault(year, []).append(ReferenceRecord(
                year=year, title=str(row.get("Title", "")).strip(), director=str(row.get("Director", "")).strip(),
                prize=str(row.get("Prize", "")).strip(), title_uri=str(row.get("TitleURI", "")).strip(),
                director_uri=str(row.get("DirectorURI", "")).strip()))
            count += 1
        logger.info(f"ReferenceStore loaded {count} rows from '{candidate}' spanning {len(self.by_year)} years.")

    def co_year_records(self, year: str, exclude_title: str = "") -> List[ReferenceRecord]:
        recs = self.by_year.get(year, [])
        if exclude_title:
            recs = [r for r in recs if r.title.lower() != exclude_title.strip().lower()]
        return recs


class GroundTruthStore:
    """GROUND_TRUTH_FILE (S.No,Film Title,AwardType,URI,ReleaseYear,Director,Birthplace,PopCount,AwardCategory).
    Backs RAG tiers 4-6 AND is the sole source of release-year resolution for this domain."""
    REQUIRED_COLUMNS = {"S.No", "Film Title", "AwardType", "URI", "ReleaseYear", "Director", "Birthplace", "PopCount"}

    def __init__(self, path: str = GROUND_TRUTH_FILE):
        self.path = path
        self.by_sno: Dict[str, List[GroundTruthRecord]] = {}
        self.all_records: List[GroundTruthRecord] = []
        self.year_by_sno: Dict[str, str] = {}
        self._load()

    def _load(self):
        candidate = _resolve_path(self.path)
        df = _read_tabular(candidate)
        df.columns = [c.strip() for c in df.columns]
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"'{candidate}' missing columns {sorted(missing)}. Found: {list(df.columns)}")
        df = df.fillna("")
        year_votes: Dict[str, Counter] = {}
        for _, row in df.iterrows():
            sno = str(row.get("S.No", "")).strip()
            if not sno:
                continue
            year = clean_year(row.get("ReleaseYear", ""))
            rec = GroundTruthRecord(
                sno=sno, film_title=str(row.get("Film Title", "")).strip(), uri=str(row.get("URI", "")).strip(),
                award_type=str(row.get("AwardType", "")).strip(), year=year,
                director=str(row.get("Director", "")).strip(), birthplace=str(row.get("Birthplace", "")).strip(),
                popcount=gt_pop_to_str(row.get("PopCount", "")), award_category=str(row.get("AwardCategory", "")).strip())
            self.by_sno.setdefault(sno, []).append(rec)
            self.all_records.append(rec)
            if year:
                year_votes.setdefault(sno, Counter())[year] += 1
        for sno, votes in year_votes.items():
            top_year, _ = votes.most_common(1)[0]
            if len(votes) > 1:
                logger.warning(f"GroundTruthStore: S.No={sno} has inconsistent ReleaseYear values {dict(votes)} — using majority year {top_year}.")
            self.year_by_sno[sno] = top_year
        logger.info(f"GroundTruthStore loaded {len(self.all_records)} rows from '{candidate}' spanning "
                    f"{len(self.by_sno)} S.No entries ({len(self.year_by_sno)} with a resolvable year).")

    def resolve_year(self, sno: Any) -> str:
        return self.year_by_sno.get(str(sno), "")

    def records_for(self, sno: Any, expected_year: str) -> List[GroundTruthRecord]:
        verified = []
        for r in self.by_sno.get(str(sno), []):
            if expected_year and r.year and r.year != expected_year:
                logger.warning(f"GroundTruthStore: S.No={sno} row year mismatch (row={r.year} vs resolved={expected_year}) — skipping row.")
                continue
            verified.append(r)
        return verified

    def sample_distractors(self, exclude_sno: Any, n: int = 5) -> List[GroundTruthRecord]:
        pool = [r for r in self.all_records if r.sno != str(exclude_sno)]
        if not pool:
            return []
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        return [pool[i] for i in idx]


def build_rag_evidence_block(tier: str, sno: Any, year: str, film_title: str,
                             ref_store: Optional[ReferenceStore],
                             gt_store: Optional[GroundTruthStore]) -> Tuple[Optional[str], int]:
    """Returns (block, rows_injected). Never fabricates birthplace/population; 'NO RESULT' rows are surfaced explicitly."""
    if tier == RAG_TIER_NO_EVIDENCE or not year:
        return None, 0
    lines = [f"RETRIEVED EVIDENCE (S.No={sno}, release year = {year})", "-" * 60]
    rows = 0
    if tier in REF_TIERS:
        if ref_store is None:
            return None, 0
        co_year = ref_store.co_year_records(year, exclude_title=film_title)
        if not co_year:
            return None, 0
        rows = len(co_year)
        if tier == RAG_TIER_YEAR_LIST_ONLY:
            lines.append("Films released in this year (source: reference index):")
            lines += [f"  - {r.title}" for r in co_year]
        elif tier == RAG_TIER_TITLES_AUTHORS:
            lines.append("Films and directors released in this year:")
            lines += [f"  - {r.title} | {r.director}" for r in co_year]
        else:
            lines.append("Films, directors, and award provenance for this year:")
            lines += [f"  - {r.title} | {r.director} | award_source={r.prize or 'unknown'}" for r in co_year]
    elif tier in GT_TIERS:
        if gt_store is None:
            return None, 0
        gt_recs = gt_store.records_for(sno, expected_year=year)
        if not gt_recs:
            return None, 0
        rows = len(gt_recs)
        if tier == RAG_TIER_FULL_WITH_BIRTHPLACE:
            lines.append("Candidate directors with award provenance and birthplace (population withheld):")
            for r in gt_recs:
                if r.director.upper() == "NO RESULT":
                    lines.append(f"  - [{r.award_type}] No qualifying director found in ground-truth evidence.")
                    continue
                lines.append(f"  - [{r.award_type}] {r.director} | birthplace={r.birthplace or 'unknown'}")
            lines.append("  NOTE: population is not provided at this tier; resolve it yourself if needed.")
        else:
            lines.append("Complete evidence including population:")
            for r in gt_recs:
                if r.director.upper() == "NO RESULT":
                    lines.append(f"  - [{r.award_type}] No qualifying director found in ground-truth evidence.")
                    continue
                lines.append(f"  - [{r.award_type}] {r.director} | birthplace={r.birthplace or 'unknown'} | "
                             f"population={r.popcount or 'unknown'}")
            if tier == RAG_TIER_FULL_WITH_DISTRACTORS:
                lines.append("")
                lines.append("Additional records from OTHER films/years (DISTRACTORS — belong to a DIFFERENT "
                             "S.No/film; do NOT use unless independently verified relevant):")
                for r in gt_store.sample_distractors(exclude_sno=sno, n=max(len(gt_recs), 5)):
                    lines.append(f"  - [S.No={r.sno}, year={r.year}, {r.award_type}] {r.film_title} | {r.director}")
    else:
        return None, 0
    lines.append("-" * 60)
    lines.append("Use the evidence above only where it is directly relevant to THIS film's S.No and year. Do not use "
                 "distractor or mismatched-year rows as qualifying. Where evidence is silent or marked unknown/NO RESULT, "
                 "rely on your own verified knowledge rather than inventing a fact.")
    return "\n".join(lines), rows


# ---------------------------------------------------------------------------
# Response-shape helpers (used for PARSING only; the stored response is never altered)
# ---------------------------------------------------------------------------
def _is_interactive_pattern(line: str) -> bool:
    markers = [
        r"^\s*\?", r"(?:would you|can you|please|could you)\s+(?:provide|tell|give|help)",
        r"(?:additional|more|further|next)\s+(?:information|details|question)",
        r"(?:clarif|confirm|verify)\s+(?:this|that|the)", r"(?:do you|did you|have you|are you)",
        r"(?:waiting|expecting|need|require[ds]?)\s+(?:for|your)", r"let me (?:ask|search|check|clarify|know)",
        r"^\s*>", r"(?:your\s+)?(?:response|input|answer|feedback|question)",
        r"(?:need|request|require)\s+(?:clarification|more info)",
    ]
    low = line.lower()
    return any(re.search(p, low) for p in markers)


def _strip_interactive_patterns(text: str) -> str:
    cutoffs = [r"\n\s*(?:What|Which|Who|How|When|Where|Why|Please|To|Do|Can|Would|Could)\s+",
               r"\n\s*>", r"\n\s*let me (?:ask|search|clarify|check)", r"\n\s*(?:do|can|would|could) you",
               r"\n\s*(?:request|asking|seeking)\s+(?:further|additional|more)"]
    for pattern in cutoffs:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            text = text[:m.start()]
    return text.rstrip()


def enforce_single_shot(text: str) -> str:
    am = re.search(r"ANSWER\s*:", text, re.IGNORECASE)
    if not am:
        return text
    section = text[am.end():]
    em = re.search(r"END\s+OF\s+RESPONSE", section, re.IGNORECASE)
    if em:
        section = section[:em.start()]
    lines = section.split("\n")
    last_valid = -1
    for idx, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "|" in line:
            last_valid = idx
        elif last_valid >= 0 and _is_interactive_pattern(line):
            break
    if last_valid >= 0:
        section = "\n".join(lines[:last_valid + 1])
    return text[:am.end()] + section


def detect_degenerate(text: str) -> Tuple[bool, str]:
    """FLAG ONLY (the HF script replaced the text with 'DEGENERATE_OUTPUT...'; the text is now always kept)."""
    if not text:
        return False, ""
    for pattern, label in [(r"(\?[\s\n]*){3,}", "Multiple question marks"), (r"(please[\s\n]*){3,}", "Multiple 'please'"),
                           (r"(what[\s\n]*){5,}", "Repeated 'what'")]:
        if re.search(pattern, text, re.IGNORECASE):
            return True, label
    lines = text.split("\n")
    if len(lines) > 10:
        cnt = Counter(l.strip() for l in lines if l.strip())
        if cnt:
            top, n = cnt.most_common(1)[0]
            if n > 10:
                return True, f"'{top[:60]}' repeated {n}x"
    for tok, grp in groupby(text.split()):
        if sum(1 for _ in grp) > 10:
            return True, f"'{tok[:40]}' repeated consecutively"
    return False, ""


def _evidence_section(evidence_block: Optional[str]) -> str:
    return f"\n{evidence_block}\n" if evidence_block else ""


def _entity_label(query_text: str, mode: str) -> str:
    if mode in WIKIDATA_MODES:
        return f"Wikidata entity {query_text}"
    if mode in DBPEDIA_MODES:
        return query_text.rstrip("/").split("/")[-1].replace("_", " ")
    return query_text


# ---------------------------------------------------------------------------
# StrategyBuilder (S1-S9, S12-S15) — every strategy accepts evidence_block
# ---------------------------------------------------------------------------
class StrategyBuilder:
    @classmethod
    def build(cls, film_title: str, strategy: str, evidence_block: Optional[str] = None) -> str:
        builders = {
            STRATEGY_FLAT_CONJUNCTIVE: cls._flat_conjunctive, STRATEGY_SEQUENTIAL_CHAIN: cls._sequential_chain,
            STRATEGY_VARIABLE_BINDING: cls._variable_binding, STRATEGY_ENTITY_FIRST: cls._entity_first,
            STRATEGY_CONSTRAINT_FIRST: cls._constraint_first, STRATEGY_PAIRWISE_EDGES: cls._pairwise_edges,
            STRATEGY_CONTRASTIVE: cls._contrastive, STRATEGY_SELF_CHECK: cls._self_check,
            STRATEGY_MATERIALIZATION: cls._materialization_checkpoint,
            STRATEGY_TEMPORAL_SCOPE: cls._temporal_scope_disambiguation, STRATEGY_FACTORIZATION: cls._factorization,
            STRATEGY_PROB_CHAINING: cls._probabilistic_chaining, STRATEGY_BACKWARD_GOAL: cls._backward_goal_decomposition,
            STRATEGY_DUAL_PATH: cls._dual_path_consistency,
        }
        if strategy not in builders:
            raise ValueError(f"Unknown strategy '{strategy}'.")
        return builders[strategy](film_title, evidence_block)

    @staticmethod
    def _flat_conjunctive(film_title: str, evidence_block: Optional[str] = None) -> str:
        ev = _evidence_section(evidence_block)
        return (
            f"Task (Strategy 1 — flat conjunctive)\n--------------------------------------\n"
            f"Target film: \"{film_title}\"\n{ev}\n"
            f"Find every director who (a) has EVER won the Academy Award for Best "
            f"Director OR the Palme d'Or, AND (b) directed at least one film released "
            f"in the same calendar year as \"{film_title}\".  For each matching director "
            f"state their birthplace and the population of that birthplace.\n\n"
            f"Note: The award-winning film and the co-year film are typically different "
            f"films.  Condition (b) does not require an award-winning film — any film "
            f"released by that director in that year qualifies.\n"
            f"Do not invent or estimate any fact.")

    @staticmethod
    def _sequential_chain(film_title: str, evidence_block: Optional[str] = None) -> str:
        ev = _evidence_section(evidence_block)
        return (
            f"Task (Strategy 2 — sequential chain decomposition)\n----------------------------------------------------\n"
            f"Target film: \"{film_title}\"\n{ev}\n"
            f"Work through the following steps in order:\n\n"
            f"  Step 1 — Release year\n    Identify the release year of \"{film_title}\".\n\n"
            f"  Step 2 — Co-year films\n    List films released in that same year (any genre, any director).\n\n"
            f"  Step 3 — Co-year directors\n    From Step 2, identify the directors of those films.\n\n"
            f"  Step 4 — Award filter\n    From Step 3, keep only directors who have EVER won the Academy Award "
            f"for Best Director OR the Palme d'Or.\n\n"
            f"  Step 5 — Birthplace\n    For each director retained in Step 4, identify their birthplace.\n\n"
            f"  Step 6 — Population\n    For each birthplace from Step 5, state the population of that place.\n\n"
            f"If evidence is provided above, prefer it over uncertain recollection, but do "
            f"not treat unrelated-year entries as qualifying.\n"
            f"Label each step.  Keep each step concise.  Do not invent or estimate facts.")

    @staticmethod
    def _variable_binding(film_title: str, evidence_block: Optional[str] = None) -> str:
        ev = _evidence_section(evidence_block)
        return (
            f"Task (Strategy 3 — variable-centred decomposition)\n----------------------------------------------------\n"
            f"Target film: \"{film_title}\"\n{ev}\n"
            f"Evaluate the following bindings in order:\n\n"
            f"  let year         = release_year( \"{film_title}\" )\n"
            f"  let co_films     = films_released_in( year )\n"
            f"  let co_directors = directors_of( co_films )\n"
            f"  let award_set    = {{ d in co_directors |\n"
            f"                        d has EVER won the Academy Award for Best Director\n"
            f"                        OR the Palme d'Or }}\n"
            f"  let birthplaces  = birthplace_of( award_set )\n"
            f"  let populations  = population_of( birthplaces )\n\n"
            f"If evidence is provided above, ground each binding in it where relevant.\n"
            f"Produce the answer from the final bindings.  Do not invent or estimate facts.")

    @staticmethod
    def _entity_first(film_title: str, evidence_block: Optional[str] = None) -> str:
        ev = _evidence_section(evidence_block)
        return (
            f"Task (Strategy 4a — entity-first / outward expansion)\n-------------------------------------------------------\n"
            f"{ev}\nStart from the film entity and expand outward:\n\n  \"{film_title}\"\n"
            f"     -> release year\n         -> other films released in that same year\n"
            f"             -> directors of those films\n"
            f"                 -> filter: keep only directors who EVER won\n"
            f"                           the Academy Award for Best Director\n"
            f"                           OR the Palme d'Or\n"
            f"                     -> birthplace of each retained director\n"
            f"                         -> population of each birthplace\n\n"
            f"Do not invent or estimate facts.")

    @staticmethod
    def _constraint_first(film_title: str, evidence_block: Optional[str] = None) -> str:
        ev = _evidence_section(evidence_block)
        return (
            f"Task (Strategy 4b — constraint-first / inward traversal)\n---------------------------------------------------------\n"
            f"Target film: \"{film_title}\"\n{ev}\n"
            f"Start from the award constraint and traverse inward:\n\n"
            f"  {{ all directors who EVER won the Academy Award for Best Director\n     OR the Palme d'Or }}\n"
            f"     -> for each such director, list films they directed\n"
            f"         -> filter: keep only films released in the same year\n"
            f"                   as \"{film_title}\"\n"
            f"             -> from the remaining directors\n"
            f"                 -> birthplace of each director\n"
            f"                     -> population of each birthplace\n\n"
            f"Do not invent or estimate facts.")

    @staticmethod
    def _pairwise_edges(film_title: str, evidence_block: Optional[str] = None) -> str:
        ev = _evidence_section(evidence_block)
        return (
            f"Task (Strategy 5 — pairwise edge factoring)\n---------------------------------------------\n"
            f"Target film: \"{film_title}\"\n{ev}\n"
            f"First establish each relational edge independently:\n\n"
            f"  Edge A  director-film:     which directors directed which films?\n"
            f"  Edge B  film-year:         what year was \"{film_title}\" released?\n"
            f"  Edge C  year-film:         which films were released in that same year?\n"
            f"  Edge D  director-award:    which directors have EVER won the Academy Award\n"
            f"                             for Best Director OR the Palme d'Or?\n"
            f"  Edge E  person-birthplace: where was each qualifying director born?\n"
            f"  Edge F  place-population:  what is the population of each birthplace?\n\n"
            f"Then compose the join:\n  (Edge D intersect Edge A applied to Edge C results)\n  -> apply Edge E -> apply Edge F\n\n"
            f"State each edge result before composing.  Do not invent or estimate facts.")

    @staticmethod
    def _contrastive(film_title: str, evidence_block: Optional[str] = None) -> str:
        ev = _evidence_section(evidence_block)
        return (
            f"Task (Strategy 6 — contrastive constraint prompting)\n------------------------------------------------------\n"
            f"Target film: \"{film_title}\"\n{ev}\n"
            f"INCLUSION criteria — a director must satisfy ALL of the following:\n"
            f"  [INCLUDE A]  Has EVER won the Academy Award for Best Director OR the Palme d'Or.\n"
            f"  [INCLUDE B]  Directed >=1 film released in the same year as \"{film_title}\".\n\n"
            f"EXCLUSION criteria — drop a director if ANY of the following:\n"
            f"  [EXCLUDE 1]  Has NOT won either award.\n"
            f"  [EXCLUDE 2]  Every film is in a different year from \"{film_title}\".\n"
            f"  [EXCLUDE 3]  Only co-year film IS \"{film_title}\" itself.\n\n"
            f"Report birthplace and population for each surviving director.\nDo not invent or estimate facts.")

    @staticmethod
    def _self_check(film_title: str, evidence_block: Optional[str] = None) -> str:
        ev = _evidence_section(evidence_block)
        return (
            f"Task (Strategy 7 — intermediate verification / self-check)\n-------------------------------------------------------------\n"
            f"Target film: \"{film_title}\"\n{ev}\n"
            f"  Step 1 — Release year.  State it. VERIFY before continuing.\n"
            f"  Step 2 — Academy Award for Best Director / Palme d'Or winners. List them. VERIFY each name.\n"
            f"  Step 3 — Co-year filter. Keep only those with >=1 film in Step 1's year. VERIFY each.\n"
            f"  Step 4 — Birthplace and population. VERIFY each fact before including.\n\n"
            f"Write 'unknown' for any unverified fact.  Do not invent or estimate facts.")

    @staticmethod
    def _materialization_checkpoint(film_title: str, evidence_block: Optional[str] = None) -> str:
        ev = _evidence_section(evidence_block)
        return f"""Task (Strategy 8 — Materialization as Checkpointing)
{"=" * 50}
Target film: "{film_title}"
{ev}
CHECKPOINT 1 (Material Y, verify before proceeding)
──────────────────────────────────────────────────
State the exact release year of "{film_title}" as a single four-digit number.
Write: "Release year: [YYYY]"
This year becomes your anchor for all downstream filters. Do not proceed
without explicitly stating this checkpoint.

CHECKPOINT 2 (Filter co-year films)
────────────────────────────────────
Using ONLY the year from CHECKPOINT 1, list all other films released in
that same year (any genre, any director). Name each film.

CHECKPOINT 3 (Extract directors)
──────────────────────────────────
From the films listed in CHECKPOINT 2, extract the full name of each director.
Keep only unique director names.

CHECKPOINT 4 (Apply award filter)
──────────────────────────────────
From the directors in CHECKPOINT 3, keep ONLY those who have EVER won the
Academy Award for Best Director OR the Palme d'Or.
Write each retained director's name. If none qualify, write "No matching directors."

CHECKPOINT 5 (Geography)
────────────────────────
For each director from CHECKPOINT 4, state:
  (a) Birthplace city
  (b) Current population of that city
Write one director per line in the format:
  [Director Name] | [Birthplace] | [Population]

─────────────────────────────────────────────────────────
RULE: Each checkpoint must complete before the next begins.
If any checkpoint produces "unknown" or "not found", state that explicitly.
Do not invent facts. Do not skip checkpoints. Do not proceed without
materializing each result.
"""

    @staticmethod
    def _temporal_scope_disambiguation(film_title: str, evidence_block: Optional[str] = None) -> str:
        ev = _evidence_section(evidence_block)
        return f"""Task (Strategy 9 — Temporal-Scope Disambiguation)
{"=" * 50}
Target film: "{film_title}"
{ev}
SCOPE DECLARATION (Do this BEFORE reasoning)
═════════════════════════════════════════════
These two conditions must be kept in SEPARATE temporal scopes throughout
your reasoning. Do not merge or conflate them.

  CONDITION A (Year-Specific Scope)
  ─────────────────────────────────
    "directed at least one film released in the same calendar year as
     {film_title}"
    → ONLY films released in THAT YEAR qualify
    → The director's birthplace, etc., can be from ANY time
    → The films in this scope do NOT have to be award-winning

  CONDITION B (Career-Wide Scope)
  ──────────────────────────────────
    "has EVER won the Academy Award for Best Director OR the Palme d'Or"
    → The award can be from ANY year in the director's career (past or future)
    → Award year ≠ {film_title} year — they may differ completely
    → Both historical and recent awards count

TEMPORAL BOUNDARY
─────────────────
These scopes DO NOT OVERLAP in their time reference. A director can win the
Academy Award for Best Director in 2015 but direct a co-year film in 1995
(or vice versa). Both conditions must be true, but they are evaluated
against different years.

───────────────────────────────────────────────────────────────────────────

Now reason through the problem:

STEP 1: State the release year of "{film_title}"
────────────────────────────────────────────────
Write: "Release year: [YYYY]"
This is your anchor for Condition A (year-specific scope).

STEP 2: Find all films released in the year from STEP 1
─────────────────────────────────────────────────────────
In Condition A scope (year-specific): List films released in that year.

STEP 3: Extract directors from STEP 2
──────────────────────────────────────
In Condition A scope: Identify directors of the films from STEP 2.

STEP 4: Apply Condition B filter (career-wide scope)
────────────────────────────────────────────────────
From the directors in STEP 3, keep ONLY those who have EVER won the
Academy Award for Best Director OR the Palme d'Or (in Condition B scope —
ANY year in their career)

STEP 5: Birthplace and population
──────────────────────────────────
For each director from STEP 4, state:
  (a) Birthplace city
  (b) Current population of that city
Format: [Director Name] | [Birthplace] | [Population]

─────────────────────────────────────────────────────────────────────────────
VERIFICATION: At the end, confirm you have kept Condition A and B separate
throughout. Do not write 'unknown' unless truly unverified.
"""

    @staticmethod
    def _factorization(film_title: str, evidence_block: Optional[str] = None) -> str:
        ev = _evidence_section(evidence_block)
        return f"""Task (Strategy 12 — Factorization into atomic units)
{"=" * 50}
Target film: "{film_title}"
{ev}
Factorize the compositional question into independent atomic factors. Each
factor below must be solvable ON ITS OWN, without needing the answer to any
other factor first (factors F1 and F3 do not depend on each other).

  FACTOR F1  (film -> year)
    What is the release year Y of "{film_title}"?

  FACTOR F2  (year -> co-year film set)
    Given Y, which films (any director) were released in year Y?
    [depends only on F1's value of Y, not on F3]

  FACTOR F3  (career-wide award membership)
    Which directors have EVER won the Academy Award for Best Director OR
    the Palme d'Or?
    [this factor is independent of F1 and F2 — it is a standalone,
    career-wide fact list]

  FACTOR F4  (director -> birthplace)
    For a given director D, what is D's birthplace?

  FACTOR F5  (place -> population)
    For a given place P, what is the population of P?

SOLVE EACH FACTOR INDEPENDENTLY FIRST
──────────────────────────────────────
Write out:
  F1 = [YYYY]
  F2 = [list of co-year films and their directors]
  F3 = [list of Academy-Award/Palme-d'Or-winning directors]

COMBINATION STEP (explicit logical conjunction)
──────────────────────────────────────────────
Final director set = directors(F2) ∩ F3

For each director in the final set (and ONLY for those directors), now
resolve F4 then F5 and report:
  [Director Name] | [Birthplace] | [Population]

If the final set is empty, write: No matching directors found.
Do not invent or estimate any fact.
"""

    @staticmethod
    def _probabilistic_chaining(film_title: str, evidence_block: Optional[str] = None) -> str:
        ev = _evidence_section(evidence_block)
        return f"""Task (Strategy 13 — Probability-weighted chain-rule composition)
{"=" * 50}
Target film: "{film_title}"
{ev}
Decompose the question into the same ordered units as a sequential chain,
but instead of a hard boolean AND filter, propagate a CONFIDENCE score for
each unit and combine them using the probability chain rule.

UNITS
─────
  U1: release_year("{film_title}") = Y
  U2: films_released_in(Y) -> co-year film set
  U3: director_of(film) for each co-year film -> candidate director set
  U4: has_ever_won_award(director) for each candidate (Academy Award for
      Best Director OR Palme d'Or)
  U5: birthplace(director) for each director surviving U4
  U6: population(birthplace) for each place from U5

FOR EACH UNIT, REPORT
──────────────────────
  value_i : your best-estimate answer for that unit
  conf_i  : your genuine confidence in value_i, a number in [0,1]
            (1.0 = certain and verifiable, using evidence above if present;
            lower = less certain)

CHAIN-RULE COMPOSITION
───────────────────────
For each candidate director, compute the running joint confidence along the
full dependency chain (U1 -> U2 -> U3 -> U4 -> U5 -> U6) using:

  conf(U1..Uk) = conf_1 × conf(U2|U1) × conf(U3|U1,U2) × ... × conf(Uk|U1..Uk-1)

INCLUSION RULE
──────────────
Only retain a director in the final answer if their full-chain joint
confidence (through U6) is >= 0.5. Do not inflate confidence to force
inclusion.

OUTPUT
──────
One line per retained director:
  [Director Name] | [Birthplace] | [Population] | [Joint Confidence]

If no director reaches the threshold, write: No matching directors found.
"""

    @staticmethod
    def _backward_goal_decomposition(film_title: str, evidence_block: Optional[str] = None) -> str:
        ev = _evidence_section(evidence_block)
        return f"""Task (Strategy 14 — Backward goal decomposition)
{"=" * 50}
Target film: "{film_title}"
{ev}
Do NOT start from the film. Start from the required OUTPUT and work
BACKWARD, asking at each step "what must I already know to produce this?"
until you reach a base case that has no further dependencies.

  GOAL
    Produce lines of the form [Director Name] | [Birthplace] | [Population]
    for every director satisfying both conditions below.

  GOAL requires -> Population(Birthplace(Director))
    Sub-goal 1: I need Birthplace(Director) before I can look up Population.

  Sub-goal 1 requires -> Birthplace(Director)
    Sub-goal 2: I need the identity of every qualifying Director before I can
    state a birthplace for any of them.

  Sub-goal 2 requires -> the qualifying Director set
    Sub-goal 3: A director qualifies iff (a) they EVER won the Academy
    Award for Best Director OR the Palme d'Or, AND (b) they directed >=1
    film in the same calendar year as "{film_title}". I need BOTH the
    award-winner list (a) AND the target film's year plus its co-year
    films (b) before I can intersect them.

  Sub-goal 3(b) requires -> the target film's release year
    Sub-goal 4 (BASE CASE): the release year of "{film_title}" does not
    depend on any other unresolved sub-goal. Resolve it first, even though
    it was the LAST sub-goal derived in this backward pass.

EXECUTE FORWARD
────────────────
Now resolve the plan in the REVERSE of the order you derived it above
(i.e., base case first: release year -> co-year films -> award-winner
list -> intersected director set -> birthplace -> population), and write
the final answer:
  [Director Name] | [Birthplace] | [Population]

If no director qualifies, write: No matching directors found.
Do not invent or estimate any fact.
"""

    @staticmethod
    def _dual_path_consistency(film_title: str, evidence_block: Optional[str] = None) -> str:
        ev = _evidence_section(evidence_block)
        return f"""Task (Strategy 15 — Dual-path consistency check)
{"=" * 50}
Target film: "{film_title}"
{ev}
Solve the question via TWO independent decomposition paths, then cross-check
that they agree before finalizing your answer.

PATH A — year-first
────────────────────
  A1. Release year of "{film_title}".
  A2. Films released in that year, and their directors.
  A3. Filter A2's directors to those who EVER won the Academy Award for
      Best Director OR the Palme d'Or.
  A4. Birthplace + population for each director surviving A3.

PATH B — award-first
──────────────────────
  B1. All directors who EVER won the Academy Award for Best Director OR
      the Palme d'Or.
  B2. For each director in B1, their full filmography.
  B3. Filter B1 to directors with >=1 film in the same calendar year as
      "{film_title}" (reuse the year from A1 once computed, or recompute
      it independently as a check).
  B4. Birthplace + population for each director surviving B3.

CONSISTENCY CHECK
───────────────────
Compare the director set produced by A4 against the director set produced
by B4. They must be IDENTICAL.
  - If they match: report the common set as your final answer.
  - If they differ: state the discrepancy explicitly, re-examine both
    paths, and resolve to the set you can best verify. Do NOT silently
    discard the mismatch or pick a path without justification.

OUTPUT
──────
[Director Name] | [Birthplace] | [Population]
If no director qualifies under both paths, write: No matching directors found.
Do not invent or estimate any fact.
"""


# ---------------------------------------------------------------------------
# SelfAskPromptBuilder (S11)
# ---------------------------------------------------------------------------
class SelfAskPromptBuilder:
    _EXEMPLAR_1 = """\
Question: Who lived longer, Ingmar Bergman or Federico Fellini?
Are follow up questions needed here: Yes.
Follow up: When did Ingmar Bergman die and how old was he?
Intermediate answer: Ingmar Bergman died in 2007 at age 89.
Follow up: When did Federico Fellini die and how old was he?
Intermediate answer: Federico Fellini died in 1993 at age 73.
So the final answer is: Ingmar Bergman lived longer.
END OF RESPONSE"""

    _EXEMPLAR_2 = """\
Question: What is the population of the birthplace of the director who won the Academy Award for Best Director or the Palme d'Or for "Parasite"?
Are follow up questions needed here: Yes.
Follow up: Who directed "Parasite"?
Intermediate answer: Bong Joon-ho directed "Parasite".
Follow up: Did Bong Joon-ho win the Academy Award for Best Director or the Palme d'Or for "Parasite"?
Intermediate answer: Yes, Bong Joon-ho won both the Palme d'Or in 2019 and the Academy Award for Best Director in 2020 for "Parasite".
Follow up: Where was Bong Joon-ho born?
Intermediate answer: Bong Joon-ho was born in Daegu, South Korea.
Follow up: What is the population of Daegu?
Intermediate answer: Daegu has a population of approximately 2,400,000.
So the final answer is: Bong Joon-ho | Daegu, South Korea | 2400000
END OF RESPONSE"""

    _OUTPUT_FORMAT = """\
Output format
-------------
After "So the final answer is:" write:
  ANSWER:
  Release year: [YYYY]
  [Director Name] | [Birthplace] | [Population]
One line per matching director. If none match write: No matching directors found.
Write "unknown" for facts you cannot verify. Do not invent or estimate.
Then write: END OF RESPONSE
Do NOT ask follow-up questions. Do NOT wait for user input. Complete immediately."""

    _CRITICAL_INSTRUCTION = """\
CRITICAL SINGLE-SHOT INSTRUCTIONS
----------------------------------
1. You MUST complete your ENTIRE reasoning in ONE response.
2. Do NOT ask the user any questions or requests.
3. Do NOT request clarification, additional input, or feedback.
4. Do NOT suggest that more information is needed.
5. Do NOT generate text that expects a user reply.
6. Do NOT say "Let me search for" or "Can you provide".
7. After writing "ANSWER:" with results, immediately write "END OF RESPONSE".
8. Your response must be SELF-CONTAINED and CONCLUSIVE.
9. Generate follow-ups only INTERNALLY, then consolidate into final answer."""

    _ROLE_GENERAL = (
        "You are a film history research assistant with comprehensive knowledge of "
        "Academy Award and Palme d'Or records, director filmographies, and biographical "
        "reference data. You reason carefully and only state facts you can verify.")
    _ROLE_WIKIPEDIA = (
        "You are an encyclopaedic film research assistant. You reason exclusively "
        "from information consistent with Wikipedia articles, infoboxes, and "
        "filmography tables. You do not include any fact not clearly supported by "
        "encyclopaedic sources.")
    _NLQ_QUESTION = (
        'Which directors have EVER won the Academy Award for Best Director or the '
        'Palme d\'Or, AND directed at least one film released in the same calendar '
        'year as "{film_title}"? For each such director state their birthplace and '
        'the population of that birthplace.')
    _KG_QUESTION = (
        'Using knowledge-graph data for {entity}, which directors have EVER won the '
        'Academy Award for Best Director or the Palme d\'Or, AND directed at least '
        'one film released in the same calendar year as this entity? For each such '
        'director state their birthplace and the population of that birthplace.')
    _QUERY_TEMPLATE = "Question: {question}\nAre follow up questions needed here:"

    @classmethod
    def build(cls, query_text: str, mode: str, query_id: Any, evidence_block: Optional[str] = None) -> str:
        role = cls._ROLE_WIKIPEDIA if mode in WIKIPEDIA_MODES else cls._ROLE_GENERAL
        question = cls._build_question(query_text, mode, query_id)
        parts = [role, "", cls._CRITICAL_INSTRUCTION, ""]
        if evidence_block:
            parts += [evidence_block, ""]
        parts += ["# Few-shot examples of Self-Ask reasoning", "", cls._EXEMPLAR_1, "", cls._EXEMPLAR_2, "",
                  cls._OUTPUT_FORMAT, "", "# Your task", "", cls._QUERY_TEMPLATE.format(question=question)]
        return "\n".join(parts)

    @classmethod
    def _build_question(cls, query_text: str, mode: str, query_id: Any) -> str:
        if mode in SPARQL_MODES:
            body = cls._KG_QUESTION.format(entity=_entity_label(query_text, mode))
        else:
            body = cls._NLQ_QUESTION.format(film_title=query_text)
        return f"[Query ID: {query_id}]  " + body


# ---------------------------------------------------------------------------
# PromptBuilder (full builder — all 8 modes, 14 strategy bodies)
# ---------------------------------------------------------------------------
class PromptBuilder:
    _OUTPUT_FORMAT = """Output format
-------------
Start your answer with the line:
  ANSWER:
On the very next line write:
  Release year: [YYYY]
Then list one line per matching director in this exact format:
  [Director Name] | [Birthplace] | [Population]
Where:
  Director Name  = the director's full name in English
  Birthplace     = the city or place of birth in English (write "unknown" if not found)
  Population     = the population of that place as a number (write "unknown" if not found)
If no directors match the criteria, write:
  No matching directors found.
Do not include any other text after the ANSWER: section.
Do not invent or estimate facts. Omit any director you cannot verify."""

    _ROLE_GENERAL = (
        "You are a film history research assistant with comprehensive knowledge of "
        "Academy Award and Palme d'Or records, director filmographies, and international "
        "biographical reference data.  You reason carefully and only state facts you can "
        "verify.")
    _ROLE_WIKIPEDIA = SelfAskPromptBuilder._ROLE_WIKIPEDIA
    _COT_SCAFFOLD = (
        "Reasoning steps (work through before writing ANSWER:)\n"
        "STEP 1: State the release year of the target film.\n"
        "STEP 2: List directors who have EVER won Academy Award for Best Director "
        "or Palme d'Or.\n"
        "STEP 3: Keep only those who directed >=1 film in that same year (not "
        "necessarily an award-winning film).\n"
        "STEP 4: For each retained director, state birthplace and population.\n"
        "STEP 5: Write ANSWER: using the output format above.")

    @classmethod
    def build(cls, query_text: str, mode: str, query_id: Any, strategy: str = STRATEGY_FLAT_CONJUNCTIVE,
              evidence_block: Optional[str] = None) -> str:
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unknown mode '{mode}'")
        role = cls._ROLE_WIKIPEDIA if mode in WIKIPEDIA_MODES else cls._ROLE_GENERAL
        task = StrategyBuilder.build(_entity_label(query_text, mode), strategy, evidence_block)
        parts = [role, f"[Query ID: {query_id}]  [Mode: {mode}]  [Strategy: {strategy}]", task]
        if mode in COT_MODES:
            parts.append(cls._COT_SCAFFOLD)
        parts += [cls._OUTPUT_FORMAT, "STEP 1:" if mode in COT_MODES else "ANSWER:"]
        return "\n\n".join(parts)


def build_prompt(query_text: str, mode: str, query_id: Any, strategy: str,
                 evidence_block: Optional[str] = None) -> Tuple[str, str]:
    if strategy == STRATEGY_SELF_ASK:
        return SelfAskPromptBuilder.build(query_text, mode, query_id, evidence_block), "self_ask"
    return PromptBuilder.build(query_text, mode, query_id, strategy, evidence_block), "v5_strategy_preamble"


# ---------------------------------------------------------------------------
# Response parser (with JointConfidence for S13)
# ---------------------------------------------------------------------------
def parse_response(text: str) -> List[Dict[str, str]]:
    if not text or text.startswith("ERROR") or text.startswith("DEGENERATE_OUTPUT"):
        return [{"ReleaseYear": "", "Director": "ERROR", "Birthplace": "", "PopCount": "", "JointConfidence": ""}]
    am = re.search(r"ANSWER\s*:", text, re.IGNORECASE)
    body = text[am.end():] if am else text.strip()
    em = re.search(r"END\s+OF\s+RESPONSE", body, re.IGNORECASE)
    if em:
        body = body[:em.start()]
    body = _strip_interactive_patterns(body).strip() if am else body
    year = ""
    ym = re.search(r"(?:release\s+year|year)\s*[:\-]?\s*((?:19|20)\d{2})", body, re.IGNORECASE)
    if ym:
        year = ym.group(1)
    rows = []
    for line in body.splitlines():
        line = re.sub(r"\*+", "", line).strip()
        if not line:
            continue
        if _is_interactive_pattern(line):
            break
        if re.match(r"^(release\s+year|year|no matching|answer|note|step|follow up|intermediate answer|so the final|end of response)",
                    line, re.IGNORECASE):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2 and parts[0]:
            pop_raw = parts[2] if len(parts) > 2 else ""
            pop = "unknown" if pop_raw.lower() == "unknown" else pop_to_str(pop_raw)
            rows.append({"ReleaseYear": year, "Director": parts[0], "Birthplace": parts[1] if len(parts) > 1 else "",
                         "PopCount": pop, "JointConfidence": parts[3] if len(parts) > 3 else ""})
    return rows or [{"ReleaseYear": year, "Director": "NO RESULT", "Birthplace": "", "PopCount": "", "JointConfidence": ""}]


def parse_for_strategy(raw: str, strategy: str) -> Tuple[List[Dict[str, str]], bool]:
    """Parse the UNALTERED response; Self-Ask trimming is applied to the parser's input only."""
    parse_text = enforce_single_shot(raw) if strategy == STRATEGY_SELF_ASK else raw
    return parse_response(parse_text), parse_text != raw


class LatentPathExtractor:
    """Output-schema compatibility only: Claude Code returns no per-token data, so every field is 0."""
    _ZERO = {"attention_mean": 0.0, "attention_min": 0.0, "attention_max": 0.0, "attention_std": 0.0,
             "attention_entropy": 0.0, "hidden_state_norm_mean": 0.0, "hidden_state_norm_min": 0.0,
             "hidden_state_norm_max": 0.0, "hidden_state_norm_std": 0.0, "token_prob_mean": 0.0,
             "token_prob_min": 0.0, "token_prob_max": 0.0, "token_prob_p10": 0.0, "token_prob_p90": 0.0,
             "confidence_score": 0.0, "perplexity_estimate": 0.0}

    def extract(self, outputs: Any = None) -> Dict[str, float]:
        return dict(self._ZERO)


# ---------------------------------------------------------------------------
# CLAUDE CODE CLIENT - streamed headless `claude -p`; every call leaves raw evidence on disk
# ---------------------------------------------------------------------------
_SYSTEM_FLAG_COMPLAINT_RE = re.compile(r"system[- ]prompt|unknown (?:option|argument)|unrecognized (?:option|argument)", re.I)
_BILLING_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX")


def _completion_tokens(usage: Any) -> float:
    if not isinstance(usage, dict):
        return float("nan")
    for k in ("output_tokens", "completion_tokens", "outputTokens", "generation_token_count"):
        if usage.get(k) is not None:
            try:
                return float(usage[k])
            except (TypeError, ValueError):
                pass
    return float("nan")


class ClaudeCodeClient:
    def __init__(self, model_alias: str = DEFAULT_MODEL_ALIAS, max_new_tokens: int = GENERATION_MAX_NEW_TOKENS,
                 system_mode: str = "neutral", claude_bin: str = "claude", timeout: int = REQUEST_TIMEOUT,
                 max_thinking_tokens: Optional[int] = None):
        if model_alias not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model alias '{model_alias}'")
        found = shutil.which(claude_bin)
        if not found:
            raise RuntimeError(f"'{claude_bin}' (Claude Code CLI) not found on PATH")
        self.bin, self.timeout = found, timeout
        self.model_alias, self.model_id, self.family = model_alias, MODEL_REGISTRY[model_alias], FAMILY
        self.max_tokens = int(max_new_tokens)
        self.max_thinking_tokens = max_thinking_tokens
        self.system_mode_requested = system_mode
        self.system_text: Optional[str] = NEUTRAL_SYSTEM if system_mode == "neutral" else None
        self.system_native = True
        self.dropped: List[str] = []
        self.raw_dir: Optional[Path] = None
        self.call_label = "call"
        self.workdir = tempfile.mkdtemp(prefix="cc_film_")      # empty cwd: no CLAUDE.md / repo context is loaded
        atexit.register(shutil.rmtree, self.workdir, True)
        logger.info(f"ClaudeCodeClient ready  alias={model_alias}  model={self.model_id}  bin={self.bin}  "
                    f"max_output_tokens={self.max_tokens}  max_thinking_tokens={max_thinking_tokens}  system_mode={system_mode}")

    @property
    def system_mode(self) -> str:
        return "default" if not self.system_text else ("native" if self.system_native else "prepended")

    def _stem(self, attempt: int) -> Path:
        d = self.raw_dir or Path(self.workdir)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.call_label}_try{attempt}"

    def _run_streaming(self, cmd: List[str], user_text: str, system_text: Optional[str],
                       env: Dict[str, str], stem: Path) -> Dict[str, Any]:
        """Run one call, writing everything to disk AS IT ARRIVES (a failed / killed call still leaves evidence)."""
        p_prompt, p_in = Path(f"{stem}_prompt.txt"), Path(f"{stem}_stdin.tmp")
        p_jsonl, p_text, p_err = Path(f"{stem}_events.jsonl"), Path(f"{stem}_text.txt"), Path(f"{stem}_stderr.txt")
        p_prompt.write_text((f"[SYSTEM]\n{system_text}\n\n" if system_text else "") + "[USER]\n" + user_text, encoding="utf-8")
        p_in.write_text(user_text, encoding="utf-8")
        text_parts: List[str] = []
        assistant_text, think_chars = "", 0
        result_evt: Optional[Dict[str, Any]] = None
        timed_out = {"v": False}
        t0 = time.perf_counter()
        last_log = t0
        with open(p_in, "rb") as fin, open(p_err, "wb") as ferr, \
                open(p_jsonl, "w", encoding="utf-8") as fj, open(p_text, "w", encoding="utf-8") as ft:
            proc = subprocess.Popen(cmd, stdin=fin, stdout=subprocess.PIPE, stderr=ferr, cwd=self.workdir, env=env,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1)

            def _kill():
                timed_out["v"] = True
                proc.kill()

            watchdog = threading.Timer(self.timeout, _kill)
            watchdog.start()
            try:
                for line in proc.stdout:
                    fj.write(line); fj.flush()
                    try:
                        evt = json.loads(line)
                    except ValueError:
                        continue
                    et = evt.get("type")
                    if et == "stream_event":
                        ev = evt.get("event") or {}
                        if ev.get("type") == "content_block_delta":
                            d = ev.get("delta") or {}
                            if d.get("type") == "text_delta":
                                s = d.get("text", "")
                                text_parts.append(s); ft.write(s); ft.flush()
                            elif d.get("type") == "thinking_delta":
                                think_chars += len(d.get("thinking", ""))
                    elif et == "assistant":
                        for b in (evt.get("message") or {}).get("content") or []:
                            if isinstance(b, dict) and b.get("type") == "text":
                                assistant_text += b.get("text", "")
                    elif et == "result":
                        result_evt = evt
                    now = time.perf_counter()
                    if now - last_log >= PROGRESS_LOG_EVERY_SEC:
                        last_log = now
                        logger.info(f"  [{self.call_label}] streaming {now - t0:.0f}s: "
                                    f"text={sum(len(x) for x in text_parts)} chars  thinking={think_chars} chars")
                proc.wait()
            finally:
                watchdog.cancel()
        text = "".join(text_parts) or assistant_text
        if not text and result_evt and not result_evt.get("is_error"):
            text = str(result_evt.get("result") or "")
        if text and not text_parts:
            p_text.write_text(text, encoding="utf-8")
        stderr_text = p_err.read_text(encoding="utf-8", errors="replace")
        blob = stderr_text + " " + (str(result_evt.get("result", "")) if result_evt else "")
        try:
            p_in.unlink()
        except OSError:
            pass
        return {"text": text, "thinking_chars": think_chars, "result_evt": result_evt, "returncode": proc.returncode,
                "stderr": stderr_text, "timed_out": timed_out["v"], "latency": time.perf_counter() - t0,
                "hit_cap": bool(re.search(r"output token maximum", blob, re.I)),
                "files": [str(x) for x in (p_prompt, p_jsonl, p_text, p_err)]}

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        mt = int(max_tokens or self.max_tokens)
        sysmsg = self.system_text
        wait, last_exc = RETRY_BASE_SEC, None
        for attempt in range(1, MAX_RETRIES + 1):
            native = bool(sysmsg and self.system_native)
            user_text = (sysmsg + "\n\n" + prompt) if (sysmsg and not self.system_native) else prompt
            cmd = [self.bin, "-p", "--model", self.model_id, "--output-format", "stream-json", "--verbose",
                   "--include-partial-messages", "--tools", "", "--max-turns", "1"]
            if native:
                cmd += ["--system-prompt", sysmsg]
            env = dict(os.environ, CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(mt))
            if self.max_thinking_tokens is not None:
                env["MAX_THINKING_TOKENS"] = str(self.max_thinking_tokens)
            r = self._run_streaming(cmd, user_text, sysmsg if native else None, env, self._stem(attempt))
            text = r["text"].strip()
            evt = r["result_evt"] or {}
            usage = evt.get("usage") if isinstance(evt.get("usage"), dict) else None
            base = {"raw_response": text, "response": text, "latency_sec": round(r["latency"], 3), "usage": usage,
                    "cost_usd": evt.get("total_cost_usd"), "max_tokens_sent": mt, "system_mode": self.system_mode,
                    "thinking_chars": r["thinking_chars"], "text_chars": len(text), "raw_files": r["files"]}
            if r["hit_cap"]:       # deterministic: do NOT retry; keep the partial text and flag it
                logger.warning(f"  [{self.call_label}] hit the {mt}-token output cap after {r['latency']:.0f}s "
                               f"(text={len(text)} chars, thinking={r['thinking_chars']} chars). Not retried. Raw: {r['files'][1]}")
                return dict(base, status=ST_TRUNC_LENGTH, finish_reason="output_token_cap")
            ok = r["returncode"] == 0 and bool(evt) and not evt.get("is_error") and not r["timed_out"]
            if ok:
                finish = evt.get("stop_reason") or evt.get("subtype")
                out_tok = _completion_tokens(usage)
                status = ST_EMPTY if not text else ST_OK
                if text and ((out_tok == out_tok and out_tok >= 0.98 * mt) or str(finish or "").lower() in ("max_tokens", "length")):
                    status = ST_TRUNC_LENGTH
                return dict(base, status=status, finish_reason=None if finish is None else str(finish))
            msg = ("TIMEOUT " if r["timed_out"] else "") + (str(evt.get("result", "")) if evt else "") + " " + r["stderr"][:400]
            if native and _SYSTEM_FLAG_COMPLAINT_RE.search(msg):
                self.system_native = False
                self.dropped.append("system_prompt_flag->prepended_to_user_turn")
                logger.warning(f"--system-prompt rejected ({msg[:120]!r}) - prepending the system text to the user turn")
                continue
            logger.warning(f"claude exit={r['returncode']} attempt {attempt}/{MAX_RETRIES}: {msg[:200]!r} "
                           f"(raw: {r['files'][1]}) - retry in {wait:.0f}s")
            time.sleep(wait); wait *= 2.0; last_exc = RuntimeError(msg[:300])
        raise RuntimeError(f"claude -p failed after {MAX_RETRIES} attempts. Last error: {last_exc}. "
                           f"Raw logs: {self.raw_dir or self.workdir}") from last_exc

    def validate_access(self) -> bool:
        logger.info(f"Validating Claude Code access  alias={self.model_alias}  model={self.model_id} ...")
        self.call_label = "validate"
        res = self.generate("Respond with only the word: OK", 256)
        logger.info(f"Validation OK - response: '{res['response'][:60]}'")
        return True


# ---------------------------------------------------------------------------
# RestartManager (a truncated record is re-run when the current output cap is larger than the one it used)
# ---------------------------------------------------------------------------
class RestartManager:
    def __init__(self, output_file: Path, max_tokens: int = 0):
        self.output_file = output_file
        self.max_tokens = max_tokens
        self.processed_ids: Set[str] = set()
        self.existing_results: List[Dict[str, Any]] = []

    def load_existing_results(self) -> bool:
        if not self.output_file.exists():
            logger.info("No existing results file found. Starting fresh.")
            return False
        try:
            data = json.loads(self.output_file.read_text(encoding="utf-8"))
            self.existing_results = data.get("results", [])
            for r in self.existing_results:
                qid, answer = r.get("Query_ID"), str(r.get("Generated_Response", "")).strip()
                if qid is None or not answer or answer.startswith("ERROR") or answer.startswith("DEGENERATE_OUTPUT"):
                    continue
                if (r.get("generation_status") in (ST_TRUNC_LENGTH, ST_TRUNC_REASONING)
                        and int(r.get("max_tokens_sent") or 0) < self.max_tokens):
                    logger.info(f"Query_ID={qid}: truncated under a smaller output cap - it will be re-run.")
                    continue
                self.processed_ids.add(str(qid))
            logger.info(f"Loaded existing results: {len(self.existing_results)} entries, {len(self.processed_ids)} reusable.")
            return len(self.processed_ids) > 0
        except Exception as e:
            logger.warning(f"Could not load existing results ({e}). Starting fresh.")
            try:
                shutil.copy2(self.output_file, self.output_file.with_suffix(".json.bak"))
            except Exception:
                pass
            self.existing_results, self.processed_ids = [], set()
            return False


# ---------------------------------------------------------------------------
# FilmDirectorAnalyser
# ---------------------------------------------------------------------------
class FilmDirectorAnalyser:
    def __init__(self, strategy: str, mode: str, client: ClaudeCodeClient, no_resume: bool = False,
                 rag_tier: str = RAG_TIER_NO_EVIDENCE, ref_store: Optional[ReferenceStore] = None,
                 gt_store: Optional[GroundTruthStore] = None, output_root: Path = OUTPUT_ROOT):
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode '{mode}'")
        if strategy not in INDIVIDUAL_STRATEGIES:
            raise ValueError(f"Unsupported strategy '{strategy}'. Use one of: {INDIVIDUAL_STRATEGIES}")
        if rag_tier not in RAG_TIERS:
            raise ValueError(f"Unsupported rag_tier '{rag_tier}'")
        self.strategy, self.mode, self.no_resume, self.rag_tier = strategy, mode, no_resume, rag_tier
        self.client, self.ref_store, self.gt_store, self.output_root = client, ref_store, gt_store, output_root
        self.model_alias, self.model_id, self.family = client.model_alias, client.model_id, client.family
        self.model_slug = model_name_to_slug(self.model_id)
        extra = f"_{RAG_TIER_SUFFIX[rag_tier]}" if rag_tier != RAG_TIER_NO_EVIDENCE else ""
        sysdef = "_SYSDEF" if client.system_mode_requested == "default" else ""
        self.suffix = f"{MODE_SUFFIX[mode]}_{STRATEGY_SUFFIX[strategy]}{extra}{sysdef}"
        self.output_dir = output_root / f"{self.model_slug}_{self.suffix}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / f"{self.model_slug}_{self.suffix}.json"
        self.response_txt = self.output_dir / f"{self.model_slug}_{self.suffix}_responses.txt"
        self.stopped_early = False
        if no_resume and self.output_file.exists():
            logger.info("--no-resume: deleting existing output and reprocessing from scratch.")
            self.output_file.unlink()
        self.restart_manager = RestartManager(self.output_file, max_tokens=client.max_tokens)
        if self.restart_manager.load_existing_results() and not no_resume:
            logger.info(f"RESUME MODE — {len(self.restart_manager.processed_ids)} records already done.")
        self._latent = LatentPathExtractor()

    def load_input_data(self) -> pd.DataFrame:
        path = _resolve_path(SOURCE_FILE)
        logger.info(f"Loading input data from: {path}")
        df = _read_tabular(path)
        df.columns = [c.strip() for c in df.columns]
        if "Film Title" not in df.columns:
            raise ValueError(f"Input file must have 'Film Title' column. Found: {list(df.columns)}")
        for col in ("SPARQL URI Identifier", "Wikidata QID"):
            if col not in df.columns:
                df[col] = ""
        df = df.fillna("")
        if "S.No" in df.columns:
            df["Query_ID"] = df["S.No"]
        elif "Query_ID" not in df.columns:
            df.insert(0, "Query_ID", range(1, len(df) + 1))
        df["Query_ID_STR"] = df["Query_ID"].astype(str).str.strip()
        logger.info(f"Loaded {len(df)} rows.")
        return df

    def _resolve_release_year(self, query_id: Any) -> str:
        return self.gt_store.resolve_year(query_id) if self.gt_store is not None else ""

    def _build_evidence_block(self, query_id: Any, film_title: str, release_year: str) -> Tuple[Optional[str], int]:
        if self.rag_tier == RAG_TIER_NO_EVIDENCE:
            return None, 0
        if not release_year:
            logger.warning(f"Q_ID={query_id}: no resolvable release year — skipping RAG evidence for this row.")
            return None, 0
        return build_rag_evidence_block(self.rag_tier, sno=query_id, year=release_year, film_title=film_title,
                                        ref_store=self.ref_store, gt_store=self.gt_store)

    @staticmethod
    def _result_sort_key(rec: Dict[str, Any]):
        s = str(rec.get("Query_ID", "")).strip()
        return (0, int(s), "") if s.lstrip("-").isdigit() else (1, 0, s)

    def _save(self, results: List[Dict[str, Any]]) -> None:
        errors = sum(1 for r in results if str(r.get("Generated_Response", "")).startswith("ERROR"))
        c = self.client
        output = {
            "metadata": {
                "script_version": SCRIPT_VERSION, "domain": "film", "model_alias": self.model_alias,
                "model_name": self.model_id, "model_family": self.family,
                "model_size": MODEL_SIZE_LABEL.get(self.model_alias, "?"),
                "inference_backend": "claude_code_cli_cloud_session",
                "query_mode": self.mode, "strategy": self.strategy, "strategy_suffix": STRATEGY_SUFFIX[self.strategy],
                "rag_tier": self.rag_tier, "rag_tier_description": RAG_TIER_DESCRIPTION.get(self.rag_tier, ""),
                "rag_enabled": self.rag_tier != RAG_TIER_NO_EVIDENCE, "output_root_used": str(self.output_root),
                "output_suffix": self.suffix, "source_column": MODE_SOURCE_COLUMN[self.mode],
                "source_file": SOURCE_FILE, "reference_file": REFERENCE_FILE, "ground_truth_file": GROUND_TRUTH_FILE,
                "timestamp": datetime.now().isoformat(), "total_queries": len(results),
                "successful_records": len(results) - errors, "error_records": errors,
                "max_output_tokens_requested": c.max_tokens, "max_thinking_tokens_env": c.max_thinking_tokens,
                "params_adapted": c.dropped, "request_timeout_s": c.timeout,
                "system_mode_requested": c.system_mode_requested, "system_mode": c.system_mode, "system_text": c.system_text,
                "sampling_note": "temperature / top_p / top_k are not controllable through `claude -p`; CLI defaults apply",
                "claude_code_call": "claude -p --model <id> --output-format stream-json --verbose --include-partial-messages "
                                    "--tools '' --max-turns 1 [--system-prompt <neutral text>]; cwd = empty temp dir; "
                                    "output cap via CLAUDE_CODE_MAX_OUTPUT_TOKENS; raw events in <cell dir>/raw_calls",
                "prompt_builder": "self_ask" if self.strategy == STRATEGY_SELF_ASK else "v5_strategy_preamble",
                "response_text_policy": "Generated_Response is the model text as returned; never altered",
                "award_criteria": ["Academy Award for Best Director", "Palme d'Or"],
                "year_resolution": "release year resolved from GROUND_TRUTH_FILE by S.No (source file has no year column)",
                "population_handling": "GT values parsed with float() first; model answers with unit-aware parser",
                "latent_note": "Claude Code returns no per-token data; latent fields are zeros",
                "modes_available": SUPPORTED_MODES, "rag_tiers_available": RAG_TIER_DESCRIPTION,
            },
            "results": sorted(results, key=self._result_sort_key),
        }
        self.output_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        parts = [f"===== Query_ID {r.get('Query_ID')} | {r.get('Film_Title', '')} =====\n{r.get('Generated_Response', '')}"
                 for r in sorted(results, key=self._result_sort_key)]
        try:
            self.response_txt.write_text("\n\n".join(parts), encoding="utf-8")
        except OSError as e:
            logger.warning(f"Could not write {self.response_txt}: {e}")
        logger.info(f"Saved {len(results)} results → {self.output_file}")

    def _base_fields(self, q_id, film_title, source_uri, sparql_uri, wikidata_qid, release_year, evidence_injected) -> Dict[str, Any]:
        return {"Query_ID": q_id, "Film_Title": film_title, "Source_URI": source_uri, "SPARQL_URI": sparql_uri,
                "Wikidata_QID": wikidata_qid, "Source_ReleaseYear": release_year, "mode": self.mode,
                "strategy": self.strategy, "rag_tier": self.rag_tier, "rag_evidence_injected": evidence_injected,
                "system_mode_requested": self.client.system_mode_requested, "model_alias": self.model_alias,
                "model_id": self.model_id, "model_family": self.family}

    def process(self, max_queries: Optional[int] = None) -> List[Dict[str, Any]]:
        df, src_col, total_in = self.load_input_data(), MODE_SOURCE_COLUMN[self.mode], 0
        total_in = len(df)
        pending = df[~df["Query_ID_STR"].isin(self.restart_manager.processed_ids)]
        if max_queries:
            pending = pending.head(max_queries)
        if pending.empty:
            logger.info("All records already processed. Nothing to do.")
            return self.restart_manager.existing_results
        results = [r for r in self.restart_manager.existing_results
                   if not str(r.get("Generated_Response", "")).startswith("ERROR")
                   and str(r.get("Query_ID")) in self.restart_manager.processed_ids]
        skipped_count, new_count, bad_streak = len(self.restart_manager.processed_ids), 0, 0
        self.client.raw_dir = self.output_dir / "raw_calls"
        logger.info(f"Processing {len(pending)}/{total_in} queries ({skipped_count} already done)  mode={self.mode}  "
                    f"strategy={self.strategy}  rag_tier={self.rag_tier}  model={self.model_id}  backend=claude_code")
        for _, row in pending.iterrows():
            q_id = row["Query_ID"]
            film_title = str(row.get("Film Title", "")).strip()
            sparql_uri = str(row.get("SPARQL URI Identifier", "")).strip()
            wikidata_qid = str(row.get("Wikidata QID", "")).strip()
            query_text = {"Film Title": film_title, "SPARQL URI Identifier": sparql_uri,
                          "Wikidata QID": wikidata_qid}.get(src_col, film_title)
            source_uri = wikidata_qid if self.mode in WIKIDATA_MODES else sparql_uri
            if not str(query_text).strip():
                logger.warning(f"Skipping Query_ID={q_id}: empty source column '{src_col}'.")
                continue
            logger.info(f"[{new_count + skipped_count + 1}/{total_in}] Query_ID={q_id}  '{film_title}'  "
                        f"source='{str(query_text)[:60]}'  rag_tier={self.rag_tier}")
            self.client.call_label = f"{self.model_slug}_{self.suffix}_q{q_id}"
            release_year = self._resolve_release_year(q_id)
            bad = False
            try:
                evidence_block, rag_rows = self._build_evidence_block(q_id, film_title, release_year)
                prompt, builder_label = build_prompt(str(query_text).strip(), self.mode, q_id, self.strategy, evidence_block)
                res = self.client.generate(prompt)
                raw = res["response"]                                  # the model's text, NEVER altered
                parsed, altered = parse_for_strategy(raw, self.strategy)
                deg, deg_why = detect_degenerate(raw)
                year = parsed[0].get("ReleaseYear", "") if parsed else ""
                record = self._base_fields(q_id, film_title, source_uri, sparql_uri, wikidata_qid, release_year, evidence_block is not None)
                record.update({
                    "system_mode": res["system_mode"], "prompt_sent": prompt, "rag_rows_injected": rag_rows,
                    "est_prompt_tokens": len(prompt) // 4, "Generated_Response": raw, "Parse_Text_Altered": altered,
                    "Degenerate_Flag": deg, "Degenerate_Reason": deg_why, "Parsed_Rows": parsed,
                    "ReleaseYear_Parsed": year,
                    "Director_Parsed": "; ".join(r["Director"] for r in parsed),
                    "Birthplace_Parsed": "; ".join(r["Birthplace"] for r in parsed),
                    "PopCount_Parsed": "; ".join(r["PopCount"] for r in parsed),
                    "JointConfidence_Parsed": "; ".join(r.get("JointConfidence", "") for r in parsed),
                    "generation_status": res["status"], "finish_reason": res["finish_reason"],
                    "max_tokens_sent": res["max_tokens_sent"], "usage": res["usage"],
                    "cost_usd_reported": res.get("cost_usd"), "thinking_chars": res.get("thinking_chars"),
                    "text_chars": res.get("text_chars"), "raw_files": res.get("raw_files"),
                    "generation_time_seconds": res["latency_sec"], "processing_timestamp": datetime.now().isoformat(),
                    "prompt_builder": builder_label})
                record.update(self._latent.extract())
                results.append(record)
                new_count += 1
                bad = res["status"] in (ST_TRUNC_LENGTH, ST_TRUNC_REASONING)
                real = [r for r in parsed if not is_sentinel_director(r["Director"])]
                logger.info(f"  OK {res['latency_sec']:.2f}s  status={res['status']}  year={year}  directors={len(real)}  "
                            f"rag_rows={rag_rows}  text_chars={res.get('text_chars')}  thinking_chars={res.get('thinking_chars')}  "
                            f"usage={res['usage']}  cost_usd={res.get('cost_usd')}")
                if deg:
                    logger.warning(f"  Possible degenerate output ({deg_why}) - flagged only, text kept")
                if bad:
                    logger.warning(f"  TRUNCATED at the output cap (raw files: {res.get('raw_files')})")
            except Exception as e:
                logger.error(f"Query_ID={q_id}: {e}")
                bad = True
                err = self._base_fields(q_id, film_title, source_uri, sparql_uri, wikidata_qid, release_year, False)
                err.update({"system_mode": "n/a", "Generated_Response": f"ERROR: {e}", "Parsed_Rows": [],
                            "ReleaseYear_Parsed": "", "Director_Parsed": "ERROR", "Birthplace_Parsed": "",
                            "PopCount_Parsed": "", "JointConfidence_Parsed": "", "Degenerate_Flag": False,
                            "Degenerate_Reason": "", "Parse_Text_Altered": False, "generation_status": "error",
                            "generation_time_seconds": 0.0, "processing_timestamp": datetime.now().isoformat()})
                err.update(self._latent.extract())
                results.append(err)
            self._save(results)                                          # calls are slow: checkpoint after every query
            bad_streak = bad_streak + 1 if bad else 0
            if bad_streak >= MAX_CONSECUTIVE_BAD:
                self.stopped_early = True
                logger.error(f"{bad_streak} consecutive errors / output-cap hits - stopping this cell to protect the credit. "
                             f"Finished queries are saved; inspect raw_calls/, then rerun (resume skips the finished ones; "
                             f"truncated ones are re-run only with a larger --max_new_tokens).")
                break
        self._save(results)
        errors = sum(1 for r in results if str(r.get("Generated_Response", "")).startswith("ERROR"))
        logger.info("=" * 80)
        logger.info(f"SUMMARY  model={self.model_alias}  mode={self.mode}  strategy={self.strategy}  rag={self.rag_tier}  "
                    f"new={new_count}  previously_done={skipped_count}  total={len(results)}  errors={errors}")
        logger.info(f"  output -> {self.output_file}")
        logger.info("=" * 80)
        return results


# ---------------------------------------------------------------------------
# REPARSE - re-run the parser over STORED response text (no Claude calls)
# ---------------------------------------------------------------------------
def reparse_outputs(root: Path) -> int:
    n = 0
    for jf in sorted(root.glob("*/*.json")):
        if jf.parent.name.startswith(ANALYSIS_DIR_NAME):
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Reparse: cannot read {jf} ({e})")
            continue
        changed = False
        for r in data.get("results", []):
            txt = str(r.get("Generated_Response", ""))
            if not txt or txt.startswith("ERROR") or txt.startswith("DEGENERATE_OUTPUT"):
                continue
            parsed, altered = parse_for_strategy(txt, r.get("strategy", ""))
            deg, why = detect_degenerate(txt)
            r.update({"Parsed_Rows": parsed, "Parse_Text_Altered": altered, "Degenerate_Flag": deg, "Degenerate_Reason": why,
                      "ReleaseYear_Parsed": parsed[0].get("ReleaseYear", "") if parsed else "",
                      "Director_Parsed": "; ".join(x["Director"] for x in parsed),
                      "Birthplace_Parsed": "; ".join(x["Birthplace"] for x in parsed),
                      "PopCount_Parsed": "; ".join(x["PopCount"] for x in parsed),
                      "JointConfidence_Parsed": "; ".join(x.get("JointConfidence", "") for x in parsed)})
            n += 1
            changed = True
        if changed:
            jf.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Reparse: {n} stored response(s) re-parsed under {root} (no model calls).")
    return n


# ---------------------------------------------------------------------------
# ANALYSIS & REPORTING - computed only from the saved outputs (ground truth optional)
# ---------------------------------------------------------------------------
def collect_records(root: Path) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for jf in sorted(root.glob("*/*.json")):
        if jf.parent.name.startswith(ANALYSIS_DIR_NAME):
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Analysis: cannot read {jf} ({e})")
            continue
        md = data.get("metadata", {}) or {}
        for r in data.get("results", []):
            r = dict(r)
            r.setdefault("model_alias", md.get("model_alias"))
            r.setdefault("strategy", md.get("strategy"))
            r.setdefault("mode", md.get("query_mode"))
            r.setdefault("rag_tier", md.get("rag_tier"))
            r.setdefault("system_mode_requested", md.get("system_mode_requested", "neutral"))
            recs.append(r)
    return recs


def _gt_directors(gt_store: Optional[GroundTruthStore], sno: str) -> Optional[List[Dict[str, Any]]]:
    if gt_store is None:
        return None
    recs = gt_store.by_sno.get(str(sno))
    if not recs or any(is_error_row(r.director) for r in recs):
        return None
    merged: Dict[str, Dict[str, Any]] = {}
    for r in recs:
        if is_sentinel_director(r.director):
            continue
        d = merged.setdefault(name_key(r.director), {"director": r.director, "birthplace": r.birthplace,
                                                      "pop": gt_pop_value(r.popcount)})
        if not d["birthplace"]:
            d["birthplace"] = r.birthplace
        if d["pop"] is None:
            d["pop"] = gt_pop_value(r.popcount)
    return list(merged.values())


def _match_pairs(g_list: List[Dict[str, Any]], p_list: List[Dict[str, Any]]):
    rem_gt, unm, pairs = list(range(len(g_list))), list(range(len(p_list))), []
    for exact in (True, False):
        for gi in list(rem_gt):
            for pi in list(unm):
                a, b = g_list[gi]["director"], p_list[pi]["Director"]
                if (name_key(a) == name_key(b)) if exact else names_match(a, b):
                    pairs.append((gi, pi)); rem_gt.remove(gi); unm.remove(pi)
                    break
    return pairs, rem_gt, unm


def evaluate_record(rec: Dict[str, Any], gt_store: Optional[GroundTruthStore]) -> Dict[str, Any]:
    resp = str(rec.get("Generated_Response", ""))
    is_err = resp.startswith("ERROR")
    rows = [] if is_err else (rec.get("Parsed_Rows") or [])
    pred = [r for r in rows if not is_sentinel_director(r.get("Director", ""))]
    sid = str(rec.get("Query_ID", ""))
    gt_year = gt_store.resolve_year(sid) if gt_store is not None else ""
    yp = str(rec.get("ReleaseYear_Parsed", ""))
    out: Dict[str, Any] = {
        "model_alias": rec.get("model_alias"), "mode": rec.get("mode"), "strategy": rec.get("strategy"),
        "rag_tier": rec.get("rag_tier"), "system_mode": rec.get("system_mode_requested", "neutral"), "Query_ID": sid,
        "status": rec.get("generation_status", ""), "is_error": is_err,
        "truncated": rec.get("generation_status") in (ST_TRUNC_LENGTH, ST_TRUNC_REASONING),
        "empty": rec.get("generation_status") == ST_EMPTY, "degenerate": bool(rec.get("Degenerate_Flag")),
        "latency": rec.get("generation_time_seconds"), "completion_tokens": _completion_tokens(rec.get("usage")),
        "text_chars": rec.get("text_chars"), "thinking_chars": rec.get("thinking_chars"),
        "cost_usd": rec.get("cost_usd_reported"), "year_parsed": yp, "gt_year": gt_year,
        "year_ok": (yp == gt_year) if (yp and gt_year) else np.nan,
        "n_pred": len(pred), "no_result": bool(rows) and not pred,
        "pop_numeric_n": sum(str(r.get("PopCount", "")).isdigit() for r in pred), "pop_total_n": len(pred),
        "tp": 0, "n_gt": 0, "scored": False, "birth_ok": 0, "birth_n": 0, "pop_5_n": 0, "pop_cmp_n": 0,
        "fn_names": [], "fp_names": [],
    }
    g_list = None if is_err else _gt_directors(gt_store, sid)
    if g_list is None:
        return out
    pairs, rem_gt, unm = _match_pairs(g_list, pred)
    out.update({"tp": len(pairs), "n_gt": len(g_list), "scored": True,
                "fn_names": [g_list[i]["director"] for i in rem_gt], "fp_names": [pred[i]["Director"] for i in unm]})
    for gi, pi in pairs:
        g, p = g_list[gi], pred[pi]
        out["birth_n"] += 1
        out["birth_ok"] += int(places_match(p.get("Birthplace", ""), g["birthplace"]))
        pp = str(p.get("PopCount", "")).strip()
        if pp.isdigit() and g["pop"]:
            out["pop_cmp_n"] += 1
            out["pop_5_n"] += int(abs(int(pp) - g["pop"]) / g["pop"] <= 0.05)
    return out


def _agg(g: pd.DataFrame) -> Dict[str, Any]:
    sc = g[g["scored"]]
    tp, npred, ngt = sc["tp"].sum(), sc["n_pred"].sum(), sc["n_gt"].sum()
    p = tp / npred if npred else np.nan
    r = tp / ngt if ngt else np.nan
    f = 2 * p * r / (p + r) if (p == p and r == r and (p + r) > 0) else np.nan

    def rate(a: str, b: str) -> float:
        d = g[b].sum()
        return g[a].sum() / d if d else np.nan

    return {"records": len(g), "error_rate": g["is_error"].mean(), "truncated_rate": g["truncated"].mean(),
            "empty_rate": g["empty"].mean(), "degenerate_rate": g["degenerate"].mean(),
            "no_result_rate": g["no_result"].mean(), "mean_directors": g["n_pred"].mean(),
            "year_accuracy": pd.to_numeric(g["year_ok"], errors="coerce").mean(),
            "pop_numeric_rate": rate("pop_numeric_n", "pop_total_n"),
            "gt_scored": int(g["scored"].sum()), "gt_precision": p, "gt_recall": r, "gt_f1": f,
            "gt_birthplace_acc": rate("birth_ok", "birth_n"), "gt_pop_within_5pct": rate("pop_5_n", "pop_cmp_n"),
            "mean_latency_s": pd.to_numeric(g["latency"], errors="coerce").mean(),
            "mean_completion_tokens": g["completion_tokens"].mean(),
            "mean_text_chars": pd.to_numeric(g["text_chars"], errors="coerce").mean(),
            "mean_thinking_chars": pd.to_numeric(g["thinking_chars"], errors="coerce").mean(),
            "total_cost_usd": pd.to_numeric(g["cost_usd"], errors="coerce").sum()}


def summarise(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        d = dict(zip(by, key))
        d.update(_agg(g))
        rows.append(d)
    return pd.DataFrame(rows)


def _tbl(d: pd.DataFrame) -> str:
    return "(no data)" if d is None or d.empty else d.to_string(index=False, float_format=lambda x: f"{x:.3f}")


def _make_plots(df: pd.DataFrame, outdir: Path, has_gt: bool) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"matplotlib unavailable ({e}) - plots skipped")
        return []
    saved: List[str] = []

    def save(fig, name: str) -> None:
        p = outdir / name
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    def code(s: str) -> str:
        return STRATEGY_SUFFIX.get(s, s).split("_")[0]

    def outcome_plot(by: str, labels, title: str, name: str) -> None:
        s = summarise(df, [by])
        if s.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 4.6))
        x = np.arange(len(s))
        ok = (1 - s["truncated_rate"] - s["empty_rate"] - s["error_rate"]).clip(lower=0)
        bottom = np.zeros(len(s))
        for vals, lab in ((ok, "ok"), (s["truncated_rate"], "truncated at cap"), (s["empty_rate"], "empty"), (s["error_rate"], "error")):
            ax.bar(x, vals.values, bottom=bottom, label=lab); bottom += vals.values
        ax.set_xticks(x); ax.set_xticklabels([labels(v) for v in s[by]], rotation=30, ha="right")
        ax.set_ylim(0, 1.02); ax.legend(fontsize=8); ax.set_title(title)
        save(fig, name)

    outcome_plot("strategy", code, "Call outcomes by strategy (share of records)", "outcomes_by_strategy.png")
    outcome_plot("mode", lambda v: str(v), "Call outcomes by mode (share of records)", "outcomes_by_mode.png")
    bs = summarise(df, ["strategy"]).assign(code=lambda d: d["strategy"].map(code))
    bm = summarise(df, ["model_alias"])
    if not bs.empty:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.bar(bs["code"], bs["mean_directors"], color="#2a6fbb")
        ax.set_ylabel("mean directors parsed per film"); ax.set_title("Answer size by strategy")
        save(fig, "directors_by_strategy.png")
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.bar(bs["code"], bs["mean_latency_s"], color="#c0392b")
        ax.set_ylabel("mean latency per call (s)"); ax.set_title("Latency by strategy")
        save(fig, "latency_by_strategy.png")
    if not bm.empty:
        fig, ax = plt.subplots(figsize=(9, max(3.2, 0.4 * len(bm) + 1.5)))
        y = np.arange(len(bm))
        ax.barh(y - 0.2, bm["mean_text_chars"].fillna(0), 0.4, label="visible text (chars)", color="#2a6fbb")
        ax.barh(y + 0.2, bm["mean_thinking_chars"].fillna(0), 0.4, label="thinking (chars)", color="#c0392b")
        ax.set_yticks(y); ax.set_yticklabels(bm["model_alias"]); ax.legend(fontsize=8)
        ax.set_title("Mean visible vs thinking length per call, by model")
        save(fig, "text_vs_thinking_by_model.png")
    if has_gt and df["scored"].any():
        sc = df[df["scored"]]
        for fac, lab_fn, fname in (("strategy", code, "gt_f1_by_strategy.png"), ("mode", str, "gt_f1_by_mode.png"),
                                   ("model_alias", str, "gt_f1_by_model.png")):
            s = summarise(sc, [fac]).sort_values("gt_f1")
            fig, ax = plt.subplots(figsize=(9, max(3.5, 0.36 * len(s) + 1.5)))
            y, hh = np.arange(len(s)), 0.27
            ax.barh(y - hh, s["gt_precision"].fillna(0), hh, label="precision")
            ax.barh(y, s["gt_recall"].fillna(0), hh, label="recall")
            ax.barh(y + hh, s["gt_f1"].fillna(0), hh, label="F1 (micro)")
            ax.set_yticks(y); ax.set_yticklabels([lab_fn(v) for v in s[fac]]); ax.set_xlim(0, 1); ax.legend(loc="lower right")
            ax.set_title(f"Ground-truth scoring by {fac}")
            save(fig, fname)
    return saved


def run_analysis(root: Path) -> None:
    recs = collect_records(root)
    if not recs:
        logger.warning(f"Analysis: no result files under {root.resolve()}.")
        return
    gt_store: Optional[GroundTruthStore] = None
    try:
        gt_store = GroundTruthStore(GROUND_TRUTH_FILE)
    except (FileNotFoundError, ValueError):
        logger.info("Analysis: no ground-truth file found - structural analysis only (no accuracy scoring).")
    df = pd.DataFrame([evaluate_record(r, gt_store) for r in recs])
    df = df.drop_duplicates(subset=["model_alias", "mode", "strategy", "rag_tier", "system_mode", "Query_ID"], keep="last").reset_index(drop=True)
    outdir = root / ANALYSIS_DIR_NAME
    outdir.mkdir(parents=True, exist_ok=True)
    has_gt = bool(df["scored"].any())
    df.drop(columns=["fn_names", "fp_names"]).to_csv(outdir / "eval_per_record.csv", index=False, encoding="utf-8-sig")
    by_model, by_mode = summarise(df, ["model_alias"]), summarise(df, ["mode"])
    by_strat, by_rag = summarise(df, ["strategy"]), summarise(df, ["rag_tier"])
    by_cfg = summarise(df, ["model_alias", "mode", "strategy", "rag_tier", "system_mode"])
    for name, t in (("summary_by_model", by_model), ("summary_by_mode", by_mode), ("summary_by_strategy", by_strat),
                    ("summary_by_rag_tier", by_rag), ("summary_by_config", by_cfg)):
        t.to_csv(outdir / f"{name}.csv", index=False, encoding="utf-8-sig")
    plots = _make_plots(df, outdir, has_gt)
    h = _agg(df)
    L: List[str] = ["=" * 100, "FILM DIRECTOR ANALYSER (CLAUDE CODE) - ANALYSIS REPORT",
                    f"generated {datetime.now().isoformat(timespec='seconds')}  (script version {SCRIPT_VERSION})", "=" * 100,
                    "\n1. SCOPE",
                    f"  records={len(df)}  models={df['model_alias'].nunique()}  modes={df['mode'].nunique()}  "
                    f"strategies={df['strategy'].nunique()}  rag_tiers={df['rag_tier'].nunique()}  system modes={sorted(df['system_mode'].unique())}",
                    "  Backend: `claude -p` in a Claude Code cloud session; temperature / top_p not controllable; tools disabled; "
                    "raw events per call in <cell dir>/raw_calls.",
                    f"  Ground-truth scoring: {'ON (' + str(int(df['scored'].sum())) + ' scored records; a film with no GT rows, or with a GT retrieval error, is not scored)' if has_gt else 'OFF'}",
                    "\n2. HEADLINE (all records pooled)"]
    for k, v in h.items():
        L.append(f"  {k:<26}{v:.4f}" if isinstance(v, (float, np.floating)) and v == v else f"  {k:<26}{v}")
    L += ["\n3. BY MODEL", _tbl(by_model), "\n4. BY MODE", _tbl(by_mode),
          "\n5. BY STRATEGY", _tbl(by_strat.assign(code=by_strat['strategy'].map(lambda s: STRATEGY_SUFFIX.get(s, s).split('_')[0]))),
          "\n6. BY RAG TIER", _tbl(by_rag), "\n7. GENERATION HEALTH",
          _tbl(df.groupby(["model_alias", "system_mode", "status"]).size().reset_index(name="count"))]
    if has_gt:
        sc = df[df["scored"]]
        miss = Counter(n for lst in sc["fn_names"] for n in lst).most_common(15)
        fps = Counter(n for lst in sc["fp_names"] for n in lst).most_common(15)
        L += ["\n8. GROUND-TRUTH DETAIL",
              "  Most missed directors: " + ("; ".join(f"{a}: {b}" for a, b in miss) or "none"),
              "  Most frequent false positives: " + ("; ".join(f"{a}: {b}" for a, b in fps) or "none")]
    ins: List[str] = []
    if h["truncated_rate"] == h["truncated_rate"] and h["truncated_rate"] > 0:
        tm = by_model[by_model["truncated_rate"] > 0].sort_values("truncated_rate", ascending=False)
        ins.append(f"{h['truncated_rate']:.1%} of calls hit the output-token cap (" +
                   ", ".join(f"{r['model_alias']} {r['truncated_rate']:.0%}" for _, r in tm.head(6).iterrows()) +
                   "). Compare text_chars with thinking_chars to see what used the budget; raise --max_new_tokens or set --max_thinking_tokens.")
    if has_gt:
        bf = by_strat.dropna(subset=["gt_f1"]).sort_values("gt_f1")
        if len(bf) > 1:
            ins.append(f"Strategy F1 spread: best {STRATEGY_CODE[bf.iloc[-1]['strategy']]} ({bf.iloc[-1]['gt_f1']:.3f}), "
                       f"worst {STRATEGY_CODE[bf.iloc[0]['strategy']]} ({bf.iloc[0]['gt_f1']:.3f}).")
        bmo = by_mode.dropna(subset=["gt_f1"]).sort_values("gt_f1")
        if len(bmo) > 1:
            ins.append(f"Mode F1 spread: best {bmo.iloc[-1]['mode']} ({bmo.iloc[-1]['gt_f1']:.3f}), worst {bmo.iloc[0]['mode']} ({bmo.iloc[0]['gt_f1']:.3f}).")
    if h["year_accuracy"] == h["year_accuracy"]:
        ins.append(f"Release-year accuracy against the GT year: {h['year_accuracy']:.1%} (over records where both years are known).")
    if h["total_cost_usd"] and h["total_cost_usd"] == h["total_cost_usd"]:
        ins.append(f"Reported cost summed over all records (Claude Code 'total_cost_usd'): {h['total_cost_usd']:.2f} USD.")
    L += ["\n9. KEY INSIGHTS"] + [f"  - {s}" for s in ins]
    L += ["\n10. LIMITATIONS",
          "  - One call per (model, mode, strategy, RAG tier, film): small differences may be sampling noise (no replicates).",
          "  - Director matching uses normalised names with token-subset tolerance; a shared surname can merge two directors.",
          "  - Sampling parameters cannot be set through `claude -p`; a call that hits the cap is stored truncated, not retried.",
          "  - SPARQL modes only name an entity (URI / QID); Claude Code has no query tool here (tools are disabled), so they test recall from the identifier.",
          "  - system_mode neutral vs default changes the system prompt; do not pool them when comparing models.",
          "\nFILES: " + (", ".join(Path(p).name for p in plots) if plots else "(no plots written)")]
    (outdir / "analysis_report.txt").write_text("\n".join(L), encoding="utf-8")
    logger.info(f"Analysis written to {outdir} ({len(plots)} plots, report + CSVs)")
    print("\n".join(L[:24]))
    print(f"\n  Full report: {outdir / 'analysis_report.txt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    global SOURCE_FILE, REFERENCE_FILE, GROUND_TRUTH_FILE
    p = argparse.ArgumentParser(
        description=(f"Film Director Analyser - Claude Code edition (script version {SCRIPT_VERSION})\n"
                     "Claude models only; every call is a streamed headless `claude -p` run inside a Claude Code cloud session.\n"
                     "8 modes x 15 strategies x 7 RAG tiers. --strategy / --rag_tier are independent; a RAG tier only controls\n"
                     "what evidence is injected and which output root is used. Raw events of every call are saved under\n"
                     "<cell dir>/raw_calls; the full log is <output root>/run.log.\n"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(["", "Models (--model; several aliases or 'all')", "-" * 42]
                         + [f"  {a:<20} -> {MODEL_REGISTRY[a]}{'  [DEFAULT]' if a == DEFAULT_MODEL_ALIAS else ''}" for a in sorted(MODEL_REGISTRY)]
                         + ["", "Modes: " + ", ".join(SUPPORTED_MODES), "", "Strategies (--strategy takes the LONG names):"]
                         + [f"  {STRATEGY_CODE[s]:<4} {s}" for s in INDIVIDUAL_STRATEGIES] + ["  all  runs all 15"]
                         + ["", "RAG tiers: " + ", ".join(RAG_TIERS),
                            f"Output roots: {OUTPUT_ROOT} (non-RAG) | {RAG_OUTPUT_ROOT} (RAG)"]))
    p.add_argument("--model", nargs="+", default=[DEFAULT_MODEL_ALIAS], choices=sorted(MODEL_REGISTRY) + ["all"])
    p.add_argument("--mode", default=None, choices=SUPPORTED_MODES)
    p.add_argument("--strategy", default=None, choices=SUPPORTED_STRATEGIES)
    p.add_argument("--rag_tier", default=RAG_TIER_NO_EVIDENCE, choices=RAG_TIERS)
    p.add_argument("--claude_bin", default="claude", help="Claude Code CLI executable (default: claude)")
    p.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT, help="seconds allowed per `claude -p` call")
    p.add_argument("--max_thinking_tokens", type=int, default=None,
                   help="sets MAX_THINKING_TOKENS for the call (0 = try to disable thinking); default: untouched")
    p.add_argument("--system_mode", choices=["neutral", "default"], default="neutral",
                   help="neutral: a short neutral --system-prompt (default); default: no flag, Claude Code's own system prompt")
    p.add_argument("--max_new_tokens", type=int, default=GENERATION_MAX_NEW_TOKENS, help="output-token cap per call")
    p.add_argument("--max_queries", type=int, default=None, help="process at most N pending queries per cell (pilot run)")
    p.add_argument("--no-resume", dest="no_resume", action="store_true", default=False)
    p.add_argument("--source_file", default=SOURCE_FILE)
    p.add_argument("--reference_file", default=REFERENCE_FILE)
    p.add_argument("--ground_truth_file", default=GROUND_TRUTH_FILE)
    p.add_argument("--reparse", action="store_true", default=False, help="re-parse STORED responses (no Claude calls), then analyse")
    p.add_argument("--analyze_only", action="store_true", default=False)
    p.add_argument("--no_analysis", action="store_true", default=False)
    args = p.parse_args()
    SOURCE_FILE, REFERENCE_FILE, GROUND_TRUTH_FILE = args.source_file, args.reference_file, args.ground_truth_file

    is_rag_run = args.rag_tier != RAG_TIER_NO_EVIDENCE
    root = RAG_OUTPUT_ROOT if is_rag_run else OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)                        # full log of every run, next to the results
    fh = logging.FileHandler(root / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info(f"=== run started: {' '.join(sys.argv)} (script version {SCRIPT_VERSION}) ===")

    if args.reparse:
        reparse_outputs(root)
        run_analysis(root)
        return 0
    if args.analyze_only:
        run_analysis(root)
        return 0
    if not args.mode or not args.strategy:
        p.error("--mode and --strategy are required unless --analyze_only or --reparse is used")
    if not shutil.which(args.claude_bin):
        print(f"ERROR: '{args.claude_bin}' (Claude Code CLI) not found on PATH. This script only runs through Claude Code.")
        return 1
    set_vars = [v for v in _BILLING_ENV_VARS if os.environ.get(v)]
    if set_vars:
        logger.warning(f"Environment variable(s) {set_vars} are set: the Claude Code CLI may bill through them instead of the cloud-session credit.")

    aliases = sorted(MODEL_REGISTRY) if "all" in args.model else list(dict.fromkeys(args.model))
    strategies_to_run = INDIVIDUAL_STRATEGIES if args.strategy == STRATEGY_ALL else [args.strategy]

    print("=" * 72)
    print(f"FILM DIRECTOR ANALYSER - CLAUDE CODE   (script version {SCRIPT_VERSION})")
    print("=" * 72)
    print(f"  Models ({len(aliases)}) : {', '.join(aliases)}")
    print(f"  Mode             : {args.mode}   (source column: {MODE_SOURCE_COLUMN[args.mode]})")
    print(f"  Strategy(ies)    : {args.strategy}" + (f" -> {len(strategies_to_run)} will run" if args.strategy == STRATEGY_ALL else ""))
    print(f"  RAG tier         : {args.rag_tier}" + ("" if is_rag_run else "  (no evidence injected)"))
    print(f"  Output root      : {root}" + ("   <-- RAG output (separate from non-RAG)" if is_rag_run else ""))
    print(f"  Source file      : {SOURCE_FILE}  (no year column — resolved from ground truth when RAG is active)")
    print(f"  Reference file   : {REFERENCE_FILE}")
    print(f"  Ground truth file: {GROUND_TRUTH_FILE}")
    print(f"  System mode      : {args.system_mode}")
    print(f"  Resume           : {'OFF (--no-resume)' if args.no_resume else 'ON (default)'}")
    print(f"  Max new tokens   : {args.max_new_tokens} per call; max thinking tokens: {args.max_thinking_tokens}; timeout {args.timeout}s")
    print(f"  Max queries      : {args.max_queries or 'all'}")
    print(f"  Award criterion  : Academy Award for Best Director OR the Palme d'Or")
    print("=" * 72)

    try:
        _resolve_path(SOURCE_FILE)
    except FileNotFoundError:
        print(f"ERROR: Input file '{SOURCE_FILE}' not found.")
        return 1

    ref_store: Optional[ReferenceStore] = None
    gt_store: Optional[GroundTruthStore] = None
    if is_rag_run:
        try:
            gt_store = GroundTruthStore(GROUND_TRUTH_FILE)            # needed by EVERY tier: year resolution
            if args.rag_tier in REF_TIERS:
                ref_store = ReferenceStore(REFERENCE_FILE)
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR loading RAG evidence stores: {e}")
            return 1

    overall_start = datetime.now()
    failed: List[str] = []
    for m_idx, alias in enumerate(aliases, 1):
        print(f"\n{'#' * 72}\n  MODEL [{m_idx}/{len(aliases)}] {alias}\n{'#' * 72}")
        try:
            client = ClaudeCodeClient(model_alias=alias, max_new_tokens=args.max_new_tokens, system_mode=args.system_mode,
                                      claude_bin=args.claude_bin, timeout=args.timeout,
                                      max_thinking_tokens=args.max_thinking_tokens)
            client.raw_dir = root / "raw_calls"
            client.validate_access()
        except Exception as e:
            logger.error(f"{alias}: startup check failed - skipping this model ({e})")
            failed.append(f"{alias}[startup]")
            continue
        for strategy in strategies_to_run:
            print(f"\n{'-' * 72}\n  {alias} | {args.mode} | {strategy} ({STRATEGY_SUFFIX[strategy]}) | RAG: {args.rag_tier}\n"
                  f"  Hint: {STRATEGY_HINT[strategy]}\n{'-' * 72}")
            try:
                analyser = FilmDirectorAnalyser(strategy=strategy, mode=args.mode, client=client, no_resume=args.no_resume,
                                                rag_tier=args.rag_tier, ref_store=ref_store, gt_store=gt_store, output_root=root)
                t0 = datetime.now()
                results = analyser.process(max_queries=args.max_queries)
                ok = sum(1 for r in results if not str(r.get("Generated_Response", "")).startswith("ERROR"))
                print(f"  done: total={len(results)} success={ok} errors={len(results) - ok} "
                      f"time={datetime.now() - t0}  -> {analyser.output_file}")
                if len(results) - ok or analyser.stopped_early:
                    failed.append(f"{alias}[{strategy}]")
            except KeyboardInterrupt:
                print("\n  Interrupted - progress saved.")
                return 130
            except Exception as e:
                print(f"  Fatal error in {alias}/{strategy}: {e}")
                traceback.print_exc()
                failed.append(f"{alias}[{strategy}]")

    print(f"\n{'=' * 72}\nALL RUNS COMPLETE\n  Total wall-clock time : {datetime.now() - overall_start}\n  Output root           : {root}")
    if failed:
        print(f"  Failed / incomplete: {failed}")
    print("=" * 72)
    if not args.no_analysis:
        try:
            run_analysis(root)
        except Exception as e:
            logger.error(f"Analysis stage failed: {e}")
            traceback.print_exc()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())