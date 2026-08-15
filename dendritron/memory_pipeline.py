"""Unified surface-memory addressing while preserving three memory spaces.

The surface path runs before the first reasoning round:

1. Qwen token IDs query frozen donor Engrams and dictionary senses through JTD.
2. Exact donor misses also receive trainable Hash-Engram addresses.
3. LNGram later receives the live hidden states and creates latent addresses.

The third step belongs to :class:`dendritron.lngram.LNGramMemory`; keeping it
outside this CPU addressor prevents token hashes and latent hashes from being
treated as one table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .hash_engram import HashEngramAddresses, HashEngramAddressor
from .jtd import SurfaceMemoryIndex
from .retrieval import ResolvedMemory
from .tokenizer import align_qwen_input


@dataclass(frozen=True)
class SurfaceMemoryPlan:
    end_position: int
    exact_memory: ResolvedMemory | None
    hash_engram: HashEngramAddresses | None
    aligned_memories: tuple[ResolvedMemory, ...] = ()

    @property
    def exact_memories(self) -> tuple[ResolvedMemory, ...]:
        if self.aligned_memories:
            return self.aligned_memories
        return () if self.exact_memory is None else (self.exact_memory,)

    @property
    def has_frozen_donor_hit(self) -> bool:
        return bool(
            any(memory.word_order in {2, 3} for memory in self.exact_memories)
        )

    @property
    def has_dictionary_candidates(self) -> bool:
        return bool(
            any(memory.word_order == 1 for memory in self.exact_memories)
        )


class SurfaceMemoryAddressor:
    """Combine exact 3→2→1 routing with Hash-Engram miss coverage."""

    def __init__(
        self,
        exact_index: SurfaceMemoryIndex,
        *,
        hash_engram: HashEngramAddressor | None = None,
    ) -> None:
        self.exact_index = exact_index
        self.hash_engram = hash_engram or HashEngramAddressor()

    def resolve(
        self,
        token_ids: Sequence[int],
        end_position: int,
        *,
        include_decomposition: bool = False,
    ) -> SurfaceMemoryPlan:
        exact = self.exact_index.resolve(
            token_ids,
            end_position,
            include_decomposition=include_decomposition,
        )
        donor_hit = bool(exact is not None and exact.word_order in {2, 3})
        projected_ids = self.exact_index.project_token_ids(token_ids)
        fallback = (
            None
            if donor_hit
            else self.hash_engram.addresses_ending_at(projected_ids, end_position)
        )
        return SurfaceMemoryPlan(
            end_position=end_position,
            exact_memory=exact,
            hash_engram=fallback,
        )

    def resolve_sequence(
        self,
        token_ids: Sequence[int],
        *,
        include_decomposition: bool = False,
    ) -> tuple[SurfaceMemoryPlan, ...]:
        return tuple(
            self.resolve(
                token_ids,
                end_position,
                include_decomposition=include_decomposition,
            )
            for end_position in range(len(token_ids))
        )

    def resolve_text(
        self,
        tokenizer,
        text: str,
        *,
        include_decomposition: bool = False,
    ) -> tuple[SurfaceMemoryPlan, ...]:
        """Resolve exact word memory with Qwen offsets and punctuation barriers.

        Hash-Engram continues to see every canonicalized Qwen token, including
        punctuation. Frozen phrase and dictionary rows are attached to the
        token position containing the complete word's end. This remains valid
        when Qwen fuses trailing punctuation with that word.
        """

        aligned = align_qwen_input(tokenizer, text)
        exact_by_token: dict[int, list[ResolvedMemory]] = {}
        surfaces = [word.surface for word in aligned.words]
        boundaries = [word.boundary_before for word in aligned.words]
        for word_position, word in enumerate(aligned.words):
            exact = self.exact_index.resolve_words(
                surfaces,
                boundaries,
                word_position,
                include_decomposition=include_decomposition,
            )
            if exact is None:
                continue
            token_end = word.token_positions[-1]
            token_start = aligned.words[exact.start_position].token_positions[0]
            remapped = ResolvedMemory(
                end_position=token_end,
                start_position=token_start,
                word_order=exact.word_order,
                selected=exact.selected,
                decomposition_candidates=exact.decomposition_candidates,
            )
            exact_by_token.setdefault(token_end, []).append(remapped)

        projected = self.exact_index.project_token_ids(aligned.input_ids)
        plans: list[SurfaceMemoryPlan] = []
        for end_position in range(len(aligned.input_ids)):
            aligned_memories = tuple(exact_by_token.get(end_position, ()))
            exact = (
                max(
                    enumerate(aligned_memories),
                    key=lambda item: (item[1].word_order, item[0]),
                )[1]
                if aligned_memories
                else None
            )
            donor_hit = any(
                memory.word_order in {2, 3} for memory in aligned_memories
            )
            fallback = (
                None
                if donor_hit
                else self.hash_engram.addresses_ending_at(projected, end_position)
            )
            plans.append(
                SurfaceMemoryPlan(
                    end_position=end_position,
                    exact_memory=exact,
                    hash_engram=fallback,
                    aligned_memories=aligned_memories,
                )
            )
        return tuple(plans)
