from __future__ import annotations

import unittest

from dendritron.expert_graph import BranchSpec, ExpertGraph, ExpertRecord


class ExpertGraphTests(unittest.TestCase):
    def test_expert_is_many_to_many_junction(self):
        expert = ExpertRecord(
            expert_id="protective-barrier",
            knowledge_anchor=(0.2, -0.1, 0.4),
            task_relation="causal-protection",
            skill_ids=(2, 7, 11),
            concept_ids=("outer-covering", "environmental-threat"),
            branches=(
                BranchSpec(
                    branch_id="moisture",
                    operator="counterfactual",
                    relation="reduces-loss",
                    skill_ids=(2, 11),
                ),
            ),
            coefficient_prior_id="prior-0042",
            success_count=9,
        )
        graph = ExpertGraph()
        graph.add(expert)

        self.assertEqual(graph.adjacent_to_skill(2), (expert,))
        self.assertEqual(graph.adjacent_to_skill(7), (expert,))
        self.assertEqual(graph.adjacent_to_skill(99), ())
        routed = graph.route(
            active_skill_ids=(2, 11),
            active_concept_ids=("outer-covering",),
            task_relation="causal-protection",
        )
        self.assertEqual(routed[0].expert, expert)
        self.assertEqual(routed[0].matched_skill_ids, (2, 11))
        self.assertTrue(routed[0].task_relation_match)

    def test_expert_can_exist_before_a_coefficient_prior(self):
        expert = ExpertRecord(
            expert_id="new-junction",
            knowledge_anchor=(1.0,),
            task_relation="comparison",
            skill_ids=(),
        )
        self.assertIsNone(expert.coefficient_prior_id)


    def test_concept_only_expert_not_routed_without_skill_overlap(self):
        # D-083: concept_ids must not expand the candidate set.
        # An expert sharing a concept but no skills with the active set
        # must not appear in route() results.
        expert_a = ExpertRecord(
            expert_id="skill-expert",
            knowledge_anchor=(0.1, 0.2),
            task_relation="task-a",
            skill_ids=(3, 5),
            concept_ids=("shared-concept",),
        )
        expert_b = ExpertRecord(
            expert_id="concept-only-expert",
            knowledge_anchor=(0.3, 0.4),
            task_relation="task-b",
            skill_ids=(7, 9),
            concept_ids=("shared-concept",),
        )
        graph = ExpertGraph()
        graph.add(expert_a)
        graph.add(expert_b)

        routed = graph.route(
            active_skill_ids=(3,),
            active_concept_ids=("shared-concept",),
        )
        routed_ids = tuple(c.expert.expert_id for c in routed)
        self.assertIn("skill-expert", routed_ids)
        self.assertNotIn("concept-only-expert", routed_ids)

    def test_empty_skill_ids_returns_no_candidates(self):
        # D-083: with no active skills, route() returns nothing even if
        # concept_ids match.
        expert = ExpertRecord(
            expert_id="lonely-expert",
            knowledge_anchor=(0.5,),
            task_relation="solo",
            skill_ids=(1,),
            concept_ids=("some-concept",),
        )
        graph = ExpertGraph()
        graph.add(expert)

        routed = graph.route(
            active_skill_ids=(),
            active_concept_ids=("some-concept",),
        )
        self.assertEqual(routed, ())


if __name__ == "__main__":
    unittest.main()
