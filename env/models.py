"""
Pydantic models for Clinical Trial AE Triage OpenEnv.
Defines Action, Observation, and State types per OpenEnv spec.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum


# ──────────────────────────────────────────────
# Enums for structured action/observation fields
# ──────────────────────────────────────────────

class SeriousnessLevel(str, Enum):
    SERIOUS = "serious"
    NON_SERIOUS = "non_serious"

class CausalityLevel(str, Enum):
    RELATED = "related"
    POSSIBLY_RELATED = "possibly_related"
    UNLIKELY = "unlikely"
    UNRELATED = "unrelated"

class ExpectednessLevel(str, Enum):
    EXPECTED = "expected"
    UNEXPECTED = "unexpected"

class TriageDecision(str, Enum):
    SUSAR = "SUSAR"
    NOT_SUSAR = "NOT_SUSAR"
    NEEDS_REVIEW = "NEEDS_REVIEW"

class RegulatoryRoute(str, Enum):
    FDA = "FDA"
    EMA = "EMA"
    PMDA = "PMDA"
    MHRA = "MHRA"
    NONE = "NONE"

class TaskID(str, Enum):
    SERIOUSNESS = "task_seriousness"       # Easy
    SUSAR_DETECTION = "task_susar"          # Medium
    FULL_TRIAGE = "task_full_triage"        # Hard


# ──────────────────────────────────────────────
# Observation: What the agent sees
# ──────────────────────────────────────────────

class AdverseEventReport(BaseModel):
    """A single AE report presented to the agent."""
    report_id: str = Field(description="Unique case identifier")
    narrative: str = Field(description="Free-text AE report narrative")
    drug_name: str = Field(description="Name of suspect drug")
    known_side_effects: List[str] = Field(
        description="Listed side effects in Reference Safety Information"
    )
    reporter_type: str = Field(description="HCP, Patient, or Consumer")
    report_source: str = Field(description="Source channel: email, call, EHR, etc.")

class Observation(BaseModel):
    """Returned by reset() and step()."""
    done: bool = Field(default=False, description="Whether episode is finished")
    reward: Optional[float] = Field(default=None, description="Reward from last action")
    task_id: TaskID = Field(description="Current task being evaluated")
    ae_report: AdverseEventReport = Field(description="The adverse event report to triage")
    step_count: int = Field(default=0, description="Current step in episode")
    max_steps: int = Field(default=5, description="Maximum steps allowed")
    feedback: Optional[str] = Field(
        default=None,
        description="Feedback from previous action (partial progress signal)"
    )
    score: Optional[float] = Field(
        default=None,
        description="Running score 0.0-1.0 for grader"
    )


# ──────────────────────────────────────────────
# Action: What the agent can do
# ──────────────────────────────────────────────

class MedDRACoding(BaseModel):
    """MedDRA coding for an adverse event term."""
    raw_term: str = Field(description="Original term from report")
    preferred_term: str = Field(description="MedDRA Preferred Term")
    soc: str = Field(default="", description="System Organ Class")

class Action(BaseModel):
    """Agent's triage action."""
    # Task 1: Seriousness only
    seriousness: Optional[SeriousnessLevel] = Field(
        default=None,
        description="Seriousness classification"
    )
    seriousness_reason: Optional[str] = Field(
        default=None,
        description="Why serious/non-serious (death, hospitalization, etc.)"
    )
    # Task 2: Full SUSAR criteria
    causality: Optional[CausalityLevel] = Field(
        default=None,
        description="Causality assessment"
    )
    expectedness: Optional[ExpectednessLevel] = Field(
        default=None,
        description="Expected or unexpected based on RSI"
    )
    triage_decision: Optional[TriageDecision] = Field(
        default=None,
        description="Final SUSAR triage decision"
    )
    # Task 3: Full triage extras
    meddra_codings: Optional[List[MedDRACoding]] = Field(
        default=None,
        description="MedDRA coded adverse events"
    )
    regulatory_route: Optional[RegulatoryRoute] = Field(
        default=None,
        description="Which regulatory authority to report to"
    )
    narrative_summary: Optional[str] = Field(
        default=None,
        description="Brief ICSR narrative summary"
    )
    expedited_report: Optional[bool] = Field(
        default=None,
        description="Whether expedited reporting is needed (7 or 15 day)"
    )


# ──────────────────────────────────────────────
# State: Internal environment state
# ──────────────────────────────────────────────

class State(BaseModel):
    """Internal state returned by state()."""
    episode_id: str = Field(default="none", description="Current episode identifier")
    task_id: TaskID = Field(default=TaskID.SERIOUSNESS, description="Active task")
    step_count: int = Field(default=0)
    case_index: int = Field(default=0, description="Which case in the bank")
    ground_truth: Optional[Dict] = Field(
        default=None,
        description="Ground truth labels (hidden from agent)"
    )
    cumulative_reward: float = Field(default=0.0)
    is_done: bool = Field(default=False)




















