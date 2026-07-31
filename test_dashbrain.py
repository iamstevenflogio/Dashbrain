"""
Regression tests for DashBrain.

Two layers, on purpose:

1. Pure-logic unit tests (extract_ticket_id, normalize_text, singularize,
   keyword_match_count) - these import app.py's helper functions directly.
   They don't need the real embedding model, so a `sentence_transformers`
   stub is injected before import to keep this fast and offline-runnable.

2. Data integrity tests against the live issue_cards.json - these catch
   the exact classes of bugs already found once (typo'd keys, duplicate
   IDs, inconsistent module casing, missing concern_summary) so they
   don't silently come back after the next batch of tickets is added.

Run with: pytest test_dashbrain.py -v
(Place this file in the same folder as app.py and issue_cards.json.)
"""
import json
import os
import sys
import types
import re
import pytest

# --- Stub sentence_transformers so importing app.py doesn't try to
# download/load the real model just to test the pure logic functions. ---
if "sentence_transformers" not in sys.modules:
    stub = types.ModuleType("sentence_transformers")

    class _FakeModel:
        def __init__(self, *a, **kw):
            pass

        def encode(self, texts, convert_to_numpy=True):
            import numpy as np
            # deterministic fake vectors, just enough for shape/threshold tests
            return np.zeros((len(texts), 8))

    stub.SentenceTransformer = _FakeModel
    sys.modules["sentence_transformers"] = stub

import app  # noqa: E402


# ---------------------------------------------------------------------
# 1. Pure-logic unit tests
# ---------------------------------------------------------------------

class TestExtractTicketId:
    def test_finds_standard_id(self):
        assert app.extract_ticket_id("Concern: DASH-14290 login issue") == "DASH-14290"

    def test_case_insensitive(self):
        assert app.extract_ticket_id("re: dash-12194 ledger") == "DASH-12194"

    def test_no_id_returns_none(self):
        assert app.extract_ticket_id("client can't find the export button") is None

    def test_short_prefix_id(self):
        assert app.extract_ticket_id("Ticket CS-105 needs review") == "CS-105"


class TestNormalizeAndSingularize:
    def test_plural_signatories_matches_singular(self):
        # this is the exact bug that was fixed: "signatory" vs "signatories"
        assert app.singularize("signatories") == "signatory"

    def test_normalize_handles_case_and_plural(self):
        assert app.normalize_text("Signatories") == app.normalize_text("signatory")

    def test_normalize_preserves_multiword_phrase_order(self):
        # multi-word tags/modules must match as a phrase, not bag-of-words
        assert app.normalize_text("Documents and Signatories") == \
            app.normalize_text("document and signatory")

    def test_silent_e_plurals_still_match_their_singular(self):
        """
        KNOWN BUG (found while writing this suite, not yet fixed in app.py):
        singularize() blindly strips the last 2 chars off any word ending
        in "es", which is correct for box/boxes, index/indexes, etc, but
        wrong for words whose singular already ends in "e" - service,
        file, template, type, fee, queue, scope, package, place. Right
        now: normalize_text("services") -> "servic" while
        normalize_text("service") -> "service", so a user typing the
        singular form will NOT match a card tagged with the plural.
        This test documents the bug; it should start passing once
        singularize() is fixed (e.g. try stripping just "s" first and
        only fall back to the "es" rule for known consonant-cluster
        endings, or keep a small exceptions list).
        """
        for singular, plural in [
            ("service", "services"),
            ("file", "files"),
            ("template", "templates"),
            ("type", "types"),
            ("fee", "fees"),
            ("queue", "queues"),
            ("scope", "scopes"),
        ]:
            assert app.normalize_text(singular) == app.normalize_text(plural), (
                f"'{singular}' vs '{plural}' -> "
                f"{app.normalize_text(singular)!r} vs {app.normalize_text(plural)!r}"
            )


class TestKeywordMatchCount:
    def _card(self, module="", tags=None):
        return {"module": module, "tags": tags or []}

    def test_generic_shared_tag_scores_one(self):
        card = self._card(module="Signatories", tags=["signatories"])
        assert app.keyword_match_count("remove a signatory from the account", card) >= 1

    def test_specific_multi_tag_scores_higher_than_generic(self):
        generic = self._card(module="Signatories", tags=["signatories"])
        specific = self._card(module="Signatories", tags=["signatories", "remove"])
        query = "please remove this signatory"
        assert app.keyword_match_count(query, specific) > app.keyword_match_count(query, generic)

    def test_no_match_scores_zero(self):
        card = self._card(module="Ledgers", tags=["referral"])
        assert app.keyword_match_count("completely unrelated barcode issue", card) == 0


# ---------------------------------------------------------------------
# 2. Data integrity tests against issue_cards.json
# ---------------------------------------------------------------------

CARDS_PATH = os.path.join(os.path.dirname(__file__), "issue_cards.json")


@pytest.fixture(scope="module")
def cards():
    with open(CARDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_no_duplicate_ticket_ids(cards):
    ids = [c.get("ticket_id") for c in cards]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"Duplicate ticket_id(s) found: {dupes}"


def test_every_card_has_ticket_id(cards):
    missing = [i for i, c in enumerate(cards) if not c.get("ticket_id")]
    assert not missing, f"Cards missing ticket_id at indices: {missing}"


def test_every_card_has_concern_summary(cards):
    # catches the old typo'd "concern summary" (space) key bug
    missing = [c["ticket_id"] for c in cards if not c.get("concern_summary")]
    assert not missing, f"Cards missing concern_summary: {missing}"


def test_no_stray_typo_keys(cards):
    # any key that looks like a near-miss of a known field name
    known = {"ticket_id", "lab", "module", "status", "date_reported",
              "concern_summary", "root_cause", "actions", "solver", "tags"}
    seen = set()
    for c in cards:
        seen.update(c.keys())
    unexpected = seen - known
    assert not unexpected, f"Unexpected/typo'd keys found in issue_cards.json: {unexpected}"


def test_module_naming_is_consistent(cards):
    """
    Flags case/variant duplicates like 'documents' vs
    'Documents and document files', 'services' vs 'Services', etc.
    This won't break search (normalize_text lowercases before matching)
    but it's exactly the drift that keeps sneaking back in with new
    ticket batches - fail loudly so it gets cleaned up at the source.
    """
    modules = [c.get("module", "") for c in cards]
    normalized_groups = {}
    for m in modules:
        key = app.normalize_text(m)
        normalized_groups.setdefault(key, set()).add(m)

    inconsistent = {k: v for k, v in normalized_groups.items() if len(v) > 1}
    assert not inconsistent, (
        "Module name has multiple raw spellings for the same normalized "
        f"value - pick one canonical form for each: {inconsistent}"
    )


def test_no_single_module_dominates_the_dataset(cards):
    """
    Regression guard for the earlier bug where one module (documents)
    was ~40% of the dataset and caused unrelated concerns to falsely
    match an 'inactive document' card. Adjust the threshold if the
    dataset legitimately grows skewed and you've re-verified matching
    quality - don't just raise it to make this pass.
    """
    from collections import Counter
    counts = Counter(app.normalize_text(c.get("module", "")) for c in cards)
    total = len(cards)
    top_module, top_count = counts.most_common(1)[0]
    share = top_count / total
    assert share <= 0.35, (
        f"'{top_module}' is {share:.0%} of all cards ({top_count}/{total}) - "
        "re-verify that specific concerns still outrank generic document "
        "matches (see keyword_match_count boost) before adding more."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))