"""Persistent per-conversation memory store for the Tier-2 pilot (system side: never sees gold, injection metadata or QA references).

Design (deliberately minimal and non-destructive):
* The store keeps every written entry (one entry per user turn whose extraction succeeded; verbatim surface text for configs C and D, the extraction
  for A and B) for the whole conversation, across all six sessions.
* Candidate retrieval ranks ALL earlier raw entries by cosine similarity (a caller-supplied metric, bge-m3 in the pilot) between the new entry's surface
  text and each stored entry's surface text, descending; ties break by write order (earlier first). The top-k are the candidates that get compared.
  Because candidates depend only on the raw entries (never on earlier merge decisions), the set of compared pairs is identical for every
  configuration and a tau sweep of a score-only host (config D) stays exact offline.
* A merge decision joins the two entries (and their clusters) into one MEMORY RECORD (union-find). A record keeps every member's source turn id and
  surface text; nothing is deleted. The text a QA answerer sees for a record is its member surfaces in write order joined by " | ".
  (An overwrite/supersede store, where a merge discards the older text, is a different semantics and is NOT exercised by this pilot.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

RECORD_SEPARATOR = " | "


@dataclass
class StoredEntry:
    entry_id: str
    turn_id: str
    session_index: int
    surface: str
    language: str
    extraction: object            # harness extraction object (has .distinction, .surface, .pair_id)
    entry_dict: dict              # the MemoryEntry dict written into pairwise records
    order: int = 0
    text: str | None = None       # the STORED text when it is not the surface (configs A and B: English gloss / native fact); None = the surface


@dataclass
class PersistentStore:
    conversation_id: str
    entries: list[StoredEntry] = field(default_factory=list)
    _parent: dict[str, str] = field(default_factory=dict)

    # ---- raw entries
    def add(self, entry: StoredEntry) -> None:
        entry.order = len(self.entries)
        self.entries.append(entry)
        self._parent[entry.entry_id] = entry.entry_id

    def rank(self, text: str, metric) -> list[tuple[StoredEntry, float]]:
        """All stored entries, most similar first (ties: earlier entry first). Does not depend on any merge decision."""
        if not self.entries:
            return []
        metric.prepare([text] + [e.surface for e in self.entries])
        scored = [(e, metric(e.surface, text)) for e in self.entries]
        return sorted(scored, key=lambda p: (-p[1], p[0].order))

    # ---- merge links (union-find over entry ids)
    def _find(self, x: str) -> str:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def merge(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            first = min((ra, rb), key=lambda r: next(e.order for e in self.entries if e.entry_id == r))
            second = rb if first == ra else ra
            self._parent[second] = first

    def same_record(self, a: str, b: str) -> bool:
        return self._find(a) == self._find(b)

    # ---- memory records (what QA retrieval sees)
    def records(self) -> list[dict]:
        groups: dict[str, list[StoredEntry]] = {}
        for e in self.entries:
            groups.setdefault(self._find(e.entry_id), []).append(e)
        out = []
        for members in groups.values():
            members.sort(key=lambda e: e.order)
            out.append({"memory_id": f"{self.conversation_id}::mem::{members[0].entry_id}",
                        "member_entry_ids": [e.entry_id for e in members], "member_turn_ids": [e.turn_id for e in members],
                        "member_session_indices": [e.session_index for e in members],
                        "text": RECORD_SEPARATOR.join(e.surface if e.text is None else e.text for e in members)})
        return sorted(out, key=lambda r: r["member_entry_ids"][0])
