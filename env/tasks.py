"""
Task definitions with case banks and graders for AE Triage OpenEnv.
3 tasks: Easy (Seriousness) → Medium (SUSAR) → Hard (Full Triage)
Each grader returns 0.0–1.0 with partial credit.
"""
from typing import Dict, List, Any
from .models import (
    Action, SeriousnessLevel, CausalityLevel,
    ExpectednessLevel, TriageDecision, RegulatoryRoute, TaskID
)


# ──────────────────────────────────────────────
# Case Bank: Realistic AE reports with ground truth
# ──────────────────────────────────────────────

CASE_BANK: List[Dict[str, Any]] = [
    {
        "report_id": "AE-2025-001",
        "narrative": (
            "Dr. Smith reports: A 65-year-old male patient developed acute "
            "liver failure and jaundice approximately 14 days after starting "
            "DrugX 200mg daily. The patient was hospitalized for 8 days. "
            "Liver function tests showed ALT > 10x ULN. Symptoms resolved "
            "upon discontinuation of DrugX. No prior liver disease. "
            "No concomitant hepatotoxic medications."
        ),
        "drug_name": "DrugX",
        "known_side_effects": [
            "Headache", "Nausea", "Dizziness", "Rash", "Fatigue"
        ],
        "reporter_type": "HCP",
        "report_source": "clinical_trial_site",
        "ground_truth": {
            "seriousness": "serious",
            "seriousness_reason": "hospitalization",
            "causality": "related",
            "expectedness": "unexpected",
            "is_susar": True,
            "meddra_terms": [
                {"raw": "liver failure", "pt": "Hepatic failure", "soc": "Hepatobiliary disorders"},
                {"raw": "jaundice", "pt": "Jaundice", "soc": "Hepatobiliary disorders"},
            ],
            "regulatory_route": "FDA",
            "expedited": True,
        },
    },
    {
        "report_id": "AE-2025-002",
        "narrative": (
            "Patient self-report: I took DrugX and had a mild headache "
            "for about 2 hours. It went away on its own. I didn't need "
            "to see a doctor or take any other medicine. I continued "
            "taking DrugX without any further issues."
        ),
        "drug_name": "DrugX",
        "known_side_effects": [
            "Headache", "Nausea", "Dizziness", "Rash", "Fatigue"
        ],
        "reporter_type": "Patient",
        "report_source": "patient_app",
        "ground_truth": {
            "seriousness": "non_serious",
            "seriousness_reason": "none",
            "causality": "possibly_related",
            "expectedness": "expected",
            "is_susar": False,
            "meddra_terms": [
                {"raw": "headache", "pt": "Headache", "soc": "Nervous system disorders"},
            ],
            "regulatory_route": "NONE",
            "expedited": False,
        },
    },
    {
        "report_id": "AE-2025-003",
        "narrative": (
            "Physician report: A 45-year-old female experienced cardiac "
            "arrest approximately 3 hours after receiving the first dose "
            "of Pembrolizumab. The patient was immediately resuscitated "
            "in the ICU. She had no prior cardiac history. The event is "
            "considered life-threatening. Pembrolizumab was permanently "
            "discontinued. The cardiac arrest is not listed in the current "
            "Investigator's Brochure for this indication."
        ),
        "drug_name": "Pembrolizumab",
        "known_side_effects": [
            "Fatigue", "Rash", "Diarrhea", "Nausea", "Pruritus"
        ],
        "reporter_type": "HCP",
        "report_source": "clinical_trial_site",
        "ground_truth": {
            "seriousness": "serious",
            "seriousness_reason": "life_threatening",
            "causality": "related",
            "expectedness": "unexpected",
            "is_susar": True,
            "meddra_terms": [
                {"raw": "cardiac arrest", "pt": "Cardiac arrest", "soc": "Cardiac disorders"},
            ],
            "regulatory_route": "FDA",
            "expedited": True,
        },
    },
    {
        "report_id": "AE-2025-004",
        "narrative": (
            "Nurse at trial site reports: A 58-year-old male patient "
            "experienced nausea and mild dizziness after taking Metformin "
            "500mg. Both symptoms are listed in the product label. Symptoms "
            "resolved within 4 hours without intervention. Patient continued "
            "treatment. No dose modification required."
        ),
        "drug_name": "Metformin",
        "known_side_effects": [
            "Nausea", "Vomiting", "Diarrhea", "Dizziness", "Headache"
        ],
        "reporter_type": "HCP",
        "report_source": "email",
        "ground_truth": {
            "seriousness": "non_serious",
            "seriousness_reason": "none",
            "causality": "related",
            "expectedness": "expected",
            "is_susar": False,
            "meddra_terms": [
                {"raw": "nausea", "pt": "Nausea", "soc": "Gastrointestinal disorders"},
                {"raw": "dizziness", "pt": "Dizziness", "soc": "Nervous system disorders"},
            ],
            "regulatory_route": "NONE",
            "expedited": False,
        },
    },
    {
        "report_id": "AE-2025-005",
        "narrative": (
            "Dr. Patel reports: A 72-year-old male on Remdesivir developed "
            "acute kidney failure and required emergency dialysis on day 5 "
            "of treatment. The patient was hospitalized. Renal failure is "
            "not a known adverse reaction for Remdesivir based on current "
            "reference safety information. Symptoms improved after drug "
            "discontinuation. This event is considered medically significant."
        ),
        "drug_name": "Remdesivir",
        "known_side_effects": [
            "Nausea", "Headache", "Rash", "Elevated transaminases"
        ],
        "reporter_type": "HCP",
        "report_source": "clinical_trial_site",
        "ground_truth": {
            "seriousness": "serious",
            "seriousness_reason": "hospitalization",
            "causality": "related",
            "expectedness": "unexpected",
            "is_susar": True,
            "meddra_terms": [
                {"raw": "kidney failure", "pt": "Renal failure", "soc": "Renal disorders"},
            ],
            "regulatory_route": "EMA",
            "expedited": True,
        },
    },
    {
        "report_id": "AE-2025-006",
        "narrative": (
            "Patient's wife called the hotline: My husband (age 60) has been "
            "on Ibuprofen 400mg for knee pain. He developed a skin rash on "
            "his arms yesterday. It's mild and not spreading. He hasn't "
            "stopped taking the medicine. No fever, no swelling, no difficulty "
            "breathing. Rash is listed as a known side effect."
        ),
        "drug_name": "Ibuprofen",
        "known_side_effects": [
            "Nausea", "Headache", "Rash", "Gastrointestinal bleeding",
            "Dizziness"
        ],
        "reporter_type": "Consumer",
        "report_source": "call_center",
        "ground_truth": {
            "seriousness": "non_serious",
            "seriousness_reason": "none",
            "causality": "possibly_related",
            "expectedness": "expected",
            "is_susar": False,
            "meddra_terms": [
                {"raw": "skin rash", "pt": "Rash", "soc": "Skin disorders"},
            ],
            "regulatory_route": "NONE",
            "expedited": False,
        },
    },
    {
        "report_id": "AE-2025-007",
        "narrative": (
            "Investigator report: A 30-year-old female developed "
            "anaphylactic shock within 15 minutes of IV infusion of "
            "DrugX during a Phase III trial. Life-threatening event "
            "requiring epinephrine and ICU admission. Anaphylaxis is "
            "not described in the Investigator's Brochure. Drug was "
            "permanently discontinued. Patient recovered after 3 days."
        ),
        "drug_name": "DrugX",
        "known_side_effects": [
            "Headache", "Nausea", "Dizziness", "Rash", "Fatigue"
        ],
        "reporter_type": "HCP",
        "report_source": "clinical_trial_site",
        "ground_truth": {
            "seriousness": "serious",
            "seriousness_reason": "life_threatening",
            "causality": "related",
            "expectedness": "unexpected",
            "is_susar": True,
            "meddra_terms": [
                {"raw": "anaphylactic shock", "pt": "Anaphylactic reaction",
                 "soc": "Immune system disorders"},
            ],
            "regulatory_route": "FDA",
            "expedited": True,
        },
    },
    {
        "report_id": "AE-2025-008",
        "narrative": (
            "Patient report via online form: I've been taking DrugX for "
            "3 months. Yesterday I felt a bit tired and had a low-grade "
            "fever (37.5°C) that lasted a few hours. I rested and felt "
            "fine the next morning. Fatigue is listed as a common side "
            "effect. I will continue my medication."
        ),
        "drug_name": "DrugX",
        "known_side_effects": [
            "Headache", "Nausea", "Dizziness", "Rash", "Fatigue"
        ],
        "reporter_type": "Patient",
        "report_source": "web_form",
        "ground_truth": {
            "seriousness": "non_serious",
            "seriousness_reason": "none",
            "causality": "possibly_related",
            "expectedness": "expected",
            "is_susar": False,
            "meddra_terms": [
                {"raw": "fatigue", "pt": "Fatigue", "soc": "General disorders"},
                {"raw": "fever", "pt": "Pyrexia", "soc": "General disorders"},
            ],
            "regulatory_route": "NONE",
            "expedited": False,
        },
    },
    {
        "report_id": "AE-2025-009",
        "narrative": (
            "Dr. Lee urgently reports: A 55-year-old male developed "
            "Stevens-Johnson Syndrome (SJS) on day 10 of Pembrolizumab "
            "therapy. Extensive skin blistering and mucosal involvement. "
            "Hospitalized in burn unit. SJS is not in the current RSI "
            "for Pembrolizumab. Drug was immediately stopped. Patient "
            "condition is critical. This is a life-threatening event."
        ),
        "drug_name": "Pembrolizumab",
        "known_side_effects": [
            "Fatigue", "Rash", "Diarrhea", "Nausea", "Pruritus"
        ],
        "reporter_type": "HCP",
        "report_source": "phone_urgent",
        "ground_truth": {
            "seriousness": "serious",
            "seriousness_reason": "life_threatening",
            "causality": "related",
            "expectedness": "unexpected",
            "is_susar": True,
            "meddra_terms": [
                {"raw": "Stevens-Johnson Syndrome", "pt": "Stevens-Johnson syndrome",
                 "soc": "Skin disorders"},
            ],
            "regulatory_route": "FDA",
            "expedited": True,
        },
    },
]


# ──────────────────────────────────────────────
# Task Definitions
# ──────────────────────────────────────────────

TASKS = {
    TaskID.SERIOUSNESS: {
        "name": "Seriousness Classification",
        "difficulty": "easy",
        "description": (
            "Classify each adverse event report as 'serious' or 'non_serious' "
            "based on ICH E2A criteria: death, life-threatening, hospitalization, "
            "disability, congenital anomaly, or medically significant."
        ),
        "max_steps": 2,
        "cases": list(range(len(CASE_BANK))),
    },
    TaskID.SUSAR_DETECTION: {
        "name": "SUSAR Detection",
        "difficulty": "medium",
        "description": (
            "For each report, assess three SUSAR criteria: (1) Seriousness, "
            "(2) Causality (is there a suspected causal relationship?), "
            "(3) Expectedness (is the reaction unexpected based on the drug's "
            "known side effects?). Determine if the case qualifies as a SUSAR."
        ),
        "max_steps": 3,
        "cases": list(range(len(CASE_BANK))),
    },
    TaskID.FULL_TRIAGE: {
        "name": "Complete AE Triage & Routing",
        "difficulty": "hard",
        "description": (
            "Perform full pharmacovigilance triage: (1) SUSAR determination, "
            "(2) MedDRA coding of adverse event terms, (3) Regulatory routing "
            "(FDA/EMA/PMDA/MHRA), (4) Expedited reporting decision, "
            "(5) Brief ICSR narrative summary."
        ),
        "max_steps": 5,
        "cases": list(range(len(CASE_BANK))),
    },
}


# ──────────────────────────────────────────────
# Graders (return 0.0–1.0 with partial credit)
# ──────────────────────────────────────────────

def grade_seriousness(action: Action, truth: Dict) -> float:
    """Task 1 grader: Seriousness classification only."""
    if action.seriousness is None:
        return 0.0
    try:
        correct = action.seriousness.value == truth.get("seriousness", "")
        reason_bonus = 0.0
        if action.seriousness_reason and len(action.seriousness_reason) > 5:
            reason_bonus = 0.1
        return min(1.0, (0.9 if correct else 0.0) + reason_bonus)
    except Exception:
        return 0.0


def grade_susar(action: Action, truth: Dict) -> float:
    """Task 2 grader: Full SUSAR criteria with partial credit."""
    try:
        score = 0.0

        # Seriousness (weight 0.25)
        if action.seriousness is not None:
            if action.seriousness.value == truth.get("seriousness", ""):
                score += 0.25

        # Causality (weight 0.25)
        if action.causality is not None:
            truth_causal = truth.get("causality", "")
            pred_causal = action.causality.value
            if pred_causal == truth_causal:
                score += 0.25
            elif pred_causal in ("related", "possibly_related") and truth_causal in ("related", "possibly_related"):
                score += 0.15

        # Expectedness (weight 0.25)
        if action.expectedness is not None:
            if action.expectedness.value == truth.get("expectedness", ""):
                score += 0.25

        # Final SUSAR decision (weight 0.25)
        if action.triage_decision is not None:
            truth_susar = truth.get("is_susar", False)
            pred_susar = action.triage_decision.value == "SUSAR"
            if pred_susar == truth_susar:
                score += 0.25
            elif truth_susar and not pred_susar:
                score -= 0.1

        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


def grade_full_triage(action: Action, truth: Dict) -> float:
    """Task 3 grader: Complete triage with 5 sub-components."""
    try:
        score = 0.0

        # 1. SUSAR determination (weight 0.30)
        susar_score = grade_susar(action, truth)
        score += 0.30 * susar_score

        # 2. MedDRA coding (weight 0.20)
        if action.meddra_codings and len(action.meddra_codings) > 0:
            truth_terms = truth.get("meddra_terms", [])
            truth_pts = {t["pt"].lower() for t in truth_terms if "pt" in t}
            pred_pts = {c.preferred_term.lower() for c in action.meddra_codings if c.preferred_term}
            if truth_pts and pred_pts:
                intersection = truth_pts & pred_pts
                precision = len(intersection) / len(pred_pts) if pred_pts else 0
                recall = len(intersection) / len(truth_pts) if truth_pts else 0
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
                score += 0.20 * f1
            elif len(action.meddra_codings) > 0:
                score += 0.05

        # 3. Regulatory routing (weight 0.15)
        if action.regulatory_route is not None:
            if action.regulatory_route.value == truth.get("regulatory_route", ""):
                score += 0.15
            elif truth.get("is_susar", False) and action.regulatory_route.value != "NONE":
                score += 0.05

        # 4. Expedited reporting (weight 0.15)
        if action.expedited_report is not None:
            if action.expedited_report == truth.get("expedited", False):
                score += 0.15
            elif truth.get("expedited", False) and not action.expedited_report:
                score -= 0.05

        # 5. Narrative summary (weight 0.20)
        if action.narrative_summary and len(action.narrative_summary) > 20:
            narrative_score = min(1.0, len(action.narrative_summary) / 200)
            narrative_lower = action.narrative_summary.lower()
            truth_terms = truth.get("meddra_terms", [])
            key_elements = []
            if truth_terms and "pt" in truth_terms[0]:
                key_elements.append(truth_terms[0]["pt"].lower())
            key_elements.append(truth.get("seriousness", ""))
            mention_count = sum(1 for k in key_elements if k and k in narrative_lower)
            narrative_score = (narrative_score * 0.5) + (mention_count / max(len(key_elements), 1) * 0.5)
            score += 0.20 * narrative_score

        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


# Map task IDs to graders
GRADERS = {
    TaskID.SERIOUSNESS: grade_seriousness,
    TaskID.SUSAR_DETECTION: grade_susar,
    TaskID.FULL_TRIAGE: grade_full_triage,
}

























