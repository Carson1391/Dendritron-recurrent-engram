"""Deterministic longest 3 -> 2 -> 1 Dendritron memory routing.

The index is tokenizer-agnostic. JTD supplies variable-length recipient-token
tuples for three-word and two-word Engrams. The dictionary compiler supplies
variable-length recipient-token tuples for one-word sense records.

At every recipient position, the router selects the longest *word-order* match:

    three-word donor Engram
    else two-word donor Engram
    else one-word dictionary sense candidates

Recipient-token length and word order remain separate. A three-word phrase can
occupy more recipient tokens than another three-word phrase. At the dictionary
fallback, every matching sense is retained for contextual sense selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable, Literal, Sequence


BoundaryMode = Literal["bos", "internal"]
MemoryKind = Literal["donor_engram", "dictionary_sense"]


@dataclass(frozen=True)
class MemoryCandidate:
    kind: MemoryKind
    word_order: int
    row_index: int
    surface_text: str
    recipient_ids: tuple[Hashable, ...]
    boundary_mode: BoundaryMode
    frequency: int = 0
    sense_id: str | None = None

    def __post_init__(self) -> None:
        if self.word_order not in {1, 2, 3}:
            raise ValueError("word_order must be 1, 2, or 3")
        if self.kind == "donor_engram" and self.word_order not in {2, 3}:
            raise ValueError("donor Engrams must have word_order 2 or 3")
        if self.kind == "dictionary_sense" and self.word_order != 1:
            raise ValueError("dictionary senses must have word_order 1")
        if not self.recipient_ids:
            raise ValueError("recipient_ids cannot be empty")
        if self.row_index < 0:
            raise ValueError("row_index must be nonnegative")


@dataclass(frozen=True)
class ResolvedMemory:
    end_position: int
    start_position: int
    word_order: int
    selected: tuple[MemoryCandidate, ...]
    decomposition_candidates: tuple[MemoryCandidate, ...] = ()

    @property
    def requires_sense_selection(self) -> bool:
        return self.word_order == 1 and len(self.selected) > 1


@dataclass
class _TrieNode:
    children: dict[Hashable, "_TrieNode"] = field(default_factory=dict)
    candidates: list[MemoryCandidate] = field(default_factory=list)


class LongestEngramRouter:
    """Reverse suffix trie implementing the locked 3 -> 2 -> 1 rule."""

    def __init__(self, candidates: Iterable[MemoryCandidate] = ()) -> None:
        self._root = _TrieNode()
        self._max_recipient_tokens = 0
        for candidate in candidates:
            self.add(candidate)

    def add(self, candidate: MemoryCandidate) -> None:
        node = self._root
        for token_id in reversed(candidate.recipient_ids):
            node = node.children.setdefault(token_id, _TrieNode())
        node.candidates.append(candidate)
        self._max_recipient_tokens = max(
            self._max_recipient_tokens,
            len(candidate.recipient_ids),
        )

    @staticmethod
    def _priority(candidate: MemoryCandidate) -> tuple[int, int, str]:
        return (-int(candidate.frequency), int(candidate.row_index), candidate.surface_text)

    def _matches_ending_at(
        self,
        recipient_ids: Sequence[Hashable],
        end_position: int,
    ) -> list[tuple[int, MemoryCandidate]]:
        if not 0 <= end_position < len(recipient_ids):
            raise IndexError("end_position falls outside recipient_ids")
        node = self._root
        matches: list[tuple[int, MemoryCandidate]] = []
        lower_bound = max(-1, end_position - self._max_recipient_tokens)
        for position in range(end_position, lower_bound, -1):
            node = node.children.get(recipient_ids[position])
            if node is None:
                break
            expected_boundary: BoundaryMode = "bos" if position == 0 else "internal"
            matches.extend(
                (position, candidate)
                for candidate in node.candidates
                if candidate.boundary_mode == expected_boundary
            )
        return matches

    def resolve(
        self,
        recipient_ids: Sequence[Hashable],
        end_position: int,
        *,
        include_decomposition: bool = False,
    ) -> ResolvedMemory | None:
        matches = self._matches_ending_at(recipient_ids, end_position)
        if not matches:
            return None

        for word_order in (3, 2, 1):
            current = [
                (start, candidate)
                for start, candidate in matches
                if candidate.word_order == word_order
            ]
            if not current:
                continue
            current.sort(key=lambda item: self._priority(item[1]))
            if word_order == 1:
                selected = tuple(candidate for _, candidate in current)
                start_position = min(start for start, _ in current)
            else:
                start_position, winner = current[0]
                selected = (winner,)

            decomposition: tuple[MemoryCandidate, ...] = ()
            if include_decomposition:
                decomposition = tuple(
                    candidate
                    for _, candidate in sorted(
                        (
                            item
                            for item in matches
                            if item[1].word_order < word_order
                        ),
                        key=lambda item: (
                            -item[1].word_order,
                            *self._priority(item[1]),
                        ),
                    )
                )
            return ResolvedMemory(
                end_position=end_position,
                start_position=start_position,
                word_order=word_order,
                selected=selected,
                decomposition_candidates=decomposition,
            )
        return None

    def resolve_sequence(
        self,
        recipient_ids: Sequence[Hashable],
        *,
        include_decomposition: bool = False,
    ) -> tuple[ResolvedMemory | None, ...]:
        return tuple(
            self.resolve(
                recipient_ids,
                end_position,
                include_decomposition=include_decomposition,
            )
            for end_position in range(len(recipient_ids))
        )
