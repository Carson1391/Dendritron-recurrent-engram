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


if __name__ == "__main__":
    unittest.main()
