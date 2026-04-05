"""
Core environment implementing OpenEnv step()/reset()/state() API
for Clinical Trial AE Triage.
"""
import uuid
from typing import Dict, Optional, Tuple
from .models import (
    Action, Observation, State, AdverseEventReport, TaskID
)
from .tasks import CASE_BANK, TASKS, GRADERS


class AETriageEnvironment:
    """
    Clinical Trial Adverse Event Triage Environment.

    Simulates the real-world task of pharmacovigilance case intake
    and triage, where an agent must classify adverse event reports
    for seriousness, causality, expectedness (SUSAR detection),
    MedDRA coding, and regulatory routing.
    """

    def __init__(self):
        self._state: Optional[State] = None
        self._current_case: Optional[Dict] = None
        self._episode_scores: list = []
        self._last_reward: float = 0.0

    def reset(self, task_id: str = "task_seriousness",
              case_index: Optional[int] = None) -> Observation:
        """
        Start a new episode.

        Args:
            task_id: Which task to run (task_seriousness, task_susar,
                     task_full_triage)
            case_index: Specific case to load (None = sequential)

        Returns:
            Initial observation with the AE report.
        """
        tid = TaskID(task_id)
        task_def = TASKS[tid]

        # Select case
        if case_index is not None:
            idx = case_index % len(CASE_BANK)
        else:
            idx = 0

        case = CASE_BANK[idx]

        # Build state
        self._state = State(
            episode_id=str(uuid.uuid4())[:8],
            task_id=tid,
            step_count=0,
            case_index=idx,
            ground_truth=case["ground_truth"],
            cumulative_reward=0.0,
            is_done=False,
        )
        self._current_case = case
        self._episode_scores = []
        self._last_reward = 0.0

        # Build observation (agent sees the report but NOT ground truth)
        ae_report = AdverseEventReport(
            report_id=case["report_id"],
            narrative=case["narrative"],
            drug_name=case["drug_name"],
            known_side_effects=case["known_side_effects"],
            reporter_type=case["reporter_type"],
            report_source=case["report_source"],
        )

        return Observation(
            done=False,
            reward=None,
            task_id=tid,
            ae_report=ae_report,
            step_count=0,
            max_steps=task_def["max_steps"],
            feedback="New case loaded. Analyze the adverse event report and provide your triage assessment.",
            score=None,
        )

    def step(self, action: Action) -> Observation:
        """
        Process agent's action and return observation + reward.

        The reward function uses asymmetric safety-first shaping:
        - Correct SUSAR identification: high positive reward
        - Missed SUSAR (false negative): severe penalty
        - False alarm (false positive): mild penalty
        - Partial progress: incremental rewards

        Args:
            action: Agent's triage decision

        Returns:
            Observation with reward and feedback
        """
        if self._state is None or self._state.is_done:
            raise ValueError("Episode not active. Call reset() first.")

        self._state.step_count += 1
        truth = self._state.ground_truth
        tid = self._state.task_id
        task_def = TASKS[tid]

        # Grade the action using task-specific grader
        grader = GRADERS[tid]
        score = grader(action, truth)
        self._episode_scores.append(score)

        # Compute shaped reward (not just the score)
        reward = self._compute_reward(action, truth, score, tid)
        self._state.cumulative_reward += reward
        self._last_reward = reward

        # Check if done
        done = self._state.step_count >= task_def["max_steps"]
        self._state.is_done = done

        # Generate feedback for partial progress
        feedback = self._generate_feedback(action, truth, score, tid)

        # Build observation
        ae_report = AdverseEventReport(
            report_id=self._current_case["report_id"],
            narrative=self._current_case["narrative"],
            drug_name=self._current_case["drug_name"],
            known_side_effects=self._current_case["known_side_effects"],
            reporter_type=self._current_case["reporter_type"],
            report_source=self._current_case["report_source"],
        )

        return Observation(
            done=done,
            reward=reward,
            task_id=tid,
            ae_report=ae_report,
            step_count=self._state.step_count,
            max_steps=task_def["max_steps"],
            feedback=feedback,
            score=score,
        )

    def state(self) -> State:
        """Return current internal state."""
        if self._state is None:
            return State(
                episode_id="none",
                task_id=TaskID.SERIOUSNESS,
                is_done=True,
            )
        return self._state

    def _compute_reward(self, action: Action, truth: Dict,
                        score: float, tid: TaskID) -> float:
        """
        Asymmetric safety-first reward shaping.

        Novel design: Missing a real SUSAR is penalized far more
        heavily than a false alarm, encoding the pharmacovigilance
        principle that patient safety always comes first.
        """
        base_reward = score  # 0.0 to 1.0 from grader

        # Safety-first asymmetric bonus/penalty
        if tid in (TaskID.SUSAR_DETECTION, TaskID.FULL_TRIAGE):
            is_true_susar = truth.get("is_susar", False)

            if action.triage_decision is not None:
                pred_susar = action.triage_decision.value == "SUSAR"

                if pred_susar and is_true_susar:
                    # Correctly caught a SUSAR — big bonus
                    base_reward += 0.3
                elif not pred_susar and is_true_susar:
                    # MISSED A SUSAR — severe penalty
                    base_reward -= 0.5
                elif pred_susar and not is_true_susar:
                    # False alarm — small penalty (better safe than sorry)
                    base_reward -= 0.05
                elif not pred_susar and not is_true_susar:
                    # Correctly identified non-SUSAR
                    base_reward += 0.1

        # Step penalty to encourage efficiency
        base_reward -= 0.02 * self._state.step_count

        return max(-1.0, min(2.0, base_reward))

    def _generate_feedback(self, action: Action, truth: Dict,
                           score: float, tid: TaskID) -> str:
        """Generate partial progress feedback for the agent."""
        parts = []
        parts.append(f"Score: {score:.2f}/1.00")

        if tid == TaskID.SERIOUSNESS:
            if action.seriousness is not None:
                correct = action.seriousness.value == truth["seriousness"]
                parts.append(
                    f"Seriousness: {'CORRECT' if correct else 'INCORRECT'}"
                )
            else:
                parts.append("Seriousness: NOT PROVIDED — please classify.")

        elif tid == TaskID.SUSAR_DETECTION:
            if action.seriousness is not None:
                parts.append(
                    f"Seriousness: {'✓' if action.seriousness.value == truth['seriousness'] else '✗'}"
                )
            if action.causality is not None:
                parts.append(
                    f"Causality: {'✓' if action.causality.value == truth['causality'] else '~'}"
                )
            if action.expectedness is not None:
                parts.append(
                    f"Expectedness: {'✓' if action.expectedness.value == truth['expectedness'] else '✗'}"
                )
            if action.triage_decision is not None:
                pred = action.triage_decision.value == "SUSAR"
                parts.append(
                    f"SUSAR Decision: {'✓' if pred == truth['is_susar'] else '✗ CRITICAL'}"
                )

        elif tid == TaskID.FULL_TRIAGE:
            filled = sum([
                action.seriousness is not None,
                action.causality is not None,
                action.expectedness is not None,
                action.triage_decision is not None,
                action.meddra_codings is not None and len(action.meddra_codings) > 0,
                action.regulatory_route is not None,
                action.expedited_report is not None,
                action.narrative_summary is not None and len(action.narrative_summary) > 10,
            ])
            parts.append(f"Fields completed: {filled}/8")

        if self._state.is_done:
            parts.append("EPISODE COMPLETE.")

        return " | ".join(parts)

    def get_episode_summary(self) -> Dict:
        """Return summary for the completed episode."""
        return {
            "episode_id": self._state.episode_id if self._state else "none",
            "task_id": self._state.task_id.value if self._state else "none",
            "total_steps": self._state.step_count if self._state else 0,
            "cumulative_reward": self._state.cumulative_reward if self._state else 0,
            "scores": self._episode_scores,
            "final_score": self._episode_scores[-1] if self._episode_scores else 0.0,
        }
