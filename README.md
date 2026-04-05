Clinical Trial AE Triage — OpenEnv Environment
An OpenEnv-compliant RL environment simulating real-world pharmacovigilance adverse event (AE) triage and SUSAR detection.
Motivation
Pharmacovigilance professionals at pharmaceutical companies process thousands of adverse event reports daily. Each report must be triaged for seriousness, causality, expectedness, coded to MedDRA terminology, and routed to the appropriate regulatory authority — all under strict 7/15-day deadlines. This environment enables AI agents to learn this critical healthcare task.
What Makes This Novel
Asymmetric safety-first reward shaping: Missing a SUSAR gets 10× the penalty of a false alarm
Three-criteria SUSAR detection: Seriousness + Causality + Expectedness (no published RL environment does this)
MedDRA coding evaluation: Agents must map free-text to standardized medical terms
Regulatory routing: Multi-target decision (FDA/EMA/PMDA/MHRA)
Real pharmacovigilance domain: Not a game or toy — fills a genuine gap in OpenEnv
Action Space
Field
Type
Task
Description
seriousness
serious / non_serious
1,2,3
ICH E2A seriousness classification
seriousness_reason
string
1,2,3
Reason (death, hospitalization, etc.)
causality
related / possibly_related / unlikely / unrelated
2,3
Causal relationship assessment
expectedness
expected / unexpected
2,3
Based on Reference Safety Information
triage_decision
SUSAR / NOT_SUSAR / NEEDS_REVIEW
2,3
Final SUSAR determination
meddra_codings
List[{raw_term, preferred_term, soc}]
3
MedDRA Preferred Term coding
regulatory_route
FDA / EMA / PMDA / MHRA / NONE
3
Regulatory authority routing
expedited_report
boolean
3
Whether expedited reporting needed
narrative_summary
string
3
Brief ICSR narrative

Observation Space
Field
Type
Description
ae_report.narrative
string
Free-text adverse event report
ae_report.drug_name
string
Suspect drug name
ae_report.known_side_effects
List[string]
Drug's known side effects (RSI)
ae_report.reporter_type
string
HCP, Patient, or Consumer
done
boolean
Episode finished?
reward
float
Reward from last action
feedback
string
Partial progress feedback
score
float
Grader score 0.0–1.0

Tasks
Task
Difficulty
Description
Scoring
task_seriousness
Easy
Classify serious vs non-serious
0.9 for correct + 0.1 reason bonus
task_susar
Medium
Full SUSAR 3-criteria assessment
0.25 per criterion (partial credit)
task_full_triage
Hard
SUSAR + MedDRA + routing + narrative
Weighted across 5 sub-tasks

Reward Design
Asymmetric Safety-First Shaping (Novel):
Scenario
Reward Modifier
Correct SUSAR identification
+0.3 bonus
Missed SUSAR (false negative)
-0.5 penalty
False alarm (false positive)
-0.05 penalty
Correct non-SUSAR
+0.1 bonus
Step penalty (efficiency)
-0.02 per step

Setup & Usage
Local Development
pip install -r requirements.txt
# Start the environment server
python -m env.server
# In another terminal, run the baseline agent
OPENAI_API_KEY=sk-xxx python inference.py

Docker
docker build -t ae-triage-env .
docker run -p 7860:7860 ae-triage-env

Environment Variables
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4
OPENAI_API_KEY=sk-xxx
HF_TOKEN=hf_xxx
ENV_URL=http://localhost:7860

Baseline Scores (GPT-4)
Task
Avg Score
Avg Reward
task_seriousness
~0.85
~0.83
task_susar
~0.72
~0.80
task_full_triage
~0.55
~0.48
Overall
~0.71
~0.70

Case Bank
9 realistic AE cases covering:
SUSAR cases: liver failure, cardiac arrest, anaphylaxis, SJS, kidney failure
Non-SUSAR cases: mild headache, expected nausea, known rash, mild fatigue
Multiple drugs: DrugX, Pembrolizumab, Metformin, Remdesivir, Ibuprofen
Various reporters: HCP, Patient, Consumer, Nurse
Multiple sources: clinical trial site, patient app, email, call center
