\- - - 

title: Clinical Trial AE Triage
emoji: 🐳
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 7860

\- - -

Clinical Trial AE Triage — OpenEnv Environment

Environment Description and Motivation

Pharmacovigilance (PV) is the science of detecting, assessing, and preventing adverse effects of medicines. Every pharmaceutical company employs teams of PV professionals who manually process thousands of adverse event (AE) reports daily. Each report must be triaged for seriousness, assessed for causality, checked for expectedness against the drug's known safety profile, coded to MedDRA (Medical Dictionary for Regulatory Activities) terminology, and routed to the appropriate regulatory authority (FDA, EMA, PMDA, MHRA) — all under strict 7-day or 15-day reporting deadlines.

This OpenEnv environment simulates the real-world task of clinical trial adverse event triage and SUSAR (Suspected Unexpected Serious Adverse Reaction) detection. An AI agent receives realistic adverse event report narratives and must make the same decisions a pharmacovigilance professional would make. The environment scores the agent's performance using deterministic graders with partial credit, and provides shaped rewards that encode the critical principle of patient safety: missing a real SUSAR is penalized far more heavily than raising a false alarm.

This environment fills a genuine gap in the OpenEnv ecosystem — no existing environment covers the pharmacovigilance domain, and the asymmetric safety-first reward design introduces novel properties for RL research.

Action Space

The agent submits a structured JSON action with the following fields. Not all fields are required for every task — easier tasks use fewer fields.

| Field               | Type                                              | Used In       | Description |
|--------------------|--------------------------------------------------|--------------|-------------|
| seriousness        | "serious" or "non_serious"                       | Task 1, 2, 3 | ICH E2A seriousness classification based on criteria: death, life-threatening, hospitalization, disability, congenital anomaly, or medically significant |
| seriousness_reason | string                                           | Task 1, 2, 3 | Brief explanation of why the event is serious or non-serious |
| causality          | "related", "possibly_related", "unlikely", or "unrelated" | Task 2, 3    | Assessment of causal relationship between the drug and the adverse event |
| expectedness       | "expected" or "unexpected"                       | Task 2, 3    | Whether the reaction is listed in the drug's Reference Safety Information (known side effects) |
| triage_decision    | "SUSAR", "NOT_SUSAR", or "NEEDS_REVIEW"          | Task 2, 3    | Final SUSAR determination: serious + related + unexpected = SUSAR |
| meddra_codings     | list of {raw_term, preferred_term, soc}          | Task 3       | MedDRA Preferred Term coding for each adverse event mentioned |
| regulatory_route   | "FDA", "EMA", "PMDA", "MHRA", or "NONE"          | Task 3       | Which regulatory authority should receive the report |
| expedited_report   | boolean                                          | Task 3       | Whether expedited reporting (7-day or 15-day) is required |
| narrative_summary  | string                                           | Task 3       | Brief 2-3 sentence ICSR (Individual Case Safety Report) narrative |


Example action for Task 2:

{

&#x20; "seriousness": "serious",

&#x20; "seriousness\_reason": "hospitalization",

&#x20; "causality": "related",

&#x20; "expectedness": "unexpected",

&#x20; "triage\_decision": "SUSAR"

}



Observation Space

The environment returns the following observation after `reset()` and `step()`:

| Field                        | Type            | Description |
|-----------------------------|-----------------|-------------|
| done                        | boolean         | Whether the episode is finished |
| reward                      | float           | Reward from the last action (null on reset) |
| task_id                     | string          | Current task: task_seriousness, task_susar, or task_full_triage |
| ae_report.report_id         | string          | Unique case identifier (e.g., "AE-2025-001") |
| ae_report.narrative         | string          | Free-text adverse event report narrative written by reporters (HCPs, patients, consumers) |
| ae_report.drug_name         | string          | Name of the suspect drug |
| ae_report.known_side_effects| list of strings | Drug's known side effects from Reference Safety Information — used to determine expectedness |
| ae_report.reporter_type     | string          | Who reported: "HCP", "Patient", or "Consumer" |
| ae_report.report_source     | string          | Source channel: "clinical_trial_site", "email", "patient_app", "call_center", "web_form", "phone_urgent" |
| step_count                  | int             | Current step number in the episode |
| max_steps                   | int             | Maximum steps allowed for the current task |
| feedback                    | string          | Partial progress feedback indicating which fields are correct/incorrect |
| score                       | float           | Current grader score from 0.0 to 1.0 |



Task Descriptions with Expected Difficulty

Task 1: Seriousness Classification (Easy)

ID: task\_seriousness | Max Steps: 2

The agent reads an adverse event report and classifies it as "serious" or "non\_serious" based on ICH E2A seriousness criteria. This is the foundational pharmacovigilance skill — determining whether an event resulted in death, was life-threatening, required hospitalization, caused disability, was a congenital anomaly, or was otherwise medically significant.

Scoring: 0.9 points for correct classification + 0.1 bonus for providing a valid reason. Expected baseline score: \~1.0 (straightforward for capable LLMs).

Task 2: SUSAR Detection (Medium)

ID: task\_susar | Max Steps: 3

The agent must assess all three SUSAR criteria: (1) Is the event serious? (2) Is there a suspected causal relationship with the drug? (3) Is the reaction unexpected based on the drug's known side effects? If all three criteria are met, the case is a SUSAR. Partial credit is awarded: 0.25 points for each correct criterion plus 0.25 for the correct final SUSAR decision.

Scoring: Weighted across 4 sub-components (seriousness, causality, expectedness, SUSAR decision) with partial credit. Missed SUSARs receive an additional -0.1 penalty. Expected baseline score: \~0.95.

Task 3: Complete AE Triage and Routing (Hard)

ID: task\_full\_triage | Max Steps: 5

The agent performs the full pharmacovigilance workflow: SUSAR determination (30%), MedDRA coding of adverse event terms to Preferred Terms (20%), regulatory routing to the correct authority (15%), expedited reporting decision (15%), and ICSR narrative generation (20%). This requires medical knowledge, regulatory awareness, and the ability to produce structured clinical documentation.

Scoring: Weighted composite across 5 sub-tasks. MedDRA coding uses F1 score against ground truth terms. Narrative quality is assessed for length and mention of key clinical elements. Expected baseline score: \~0.62-0.84 depending on model capability.

Reward Design

The environment uses a novel asymmetric safety-first reward function that prioritizes patient safety.

| Scenario                                   | Reward Modifier | Rationale                                      |
|-------------------------------------------|-----------------|-----------------------------------------------|
| Correctly identified SUSAR                | +0.3 bonus      | Catching real safety signals is critical       |
| Missed a real SUSAR (false negative)      | -0.5 penalty    | Missing a safety signal can endanger patients  |
| False alarm (false positive)              | -0.05 penalty   | Minor cost — better safe than sorry            |
| Correctly identified non-SUSAR            | +0.1 bonus      | Routine correct classification                 |
| Step penalty                             | -0.02 per step  | Encourages efficient triage                    |



The missed-SUSAR penalty is 10x the false alarm penalty, reflecting real-world consequences where failing to report a serious unexpected reaction can lead to patient harm, regulatory sanctions, and trial suspension.

Setup and Usage Instructions

Prerequisites

Python 3.10+

Docker (for containerized deployment)

An LLM API key (OpenAI, Groq, HuggingFace, or Gemini)

Local Development

\# Install dependencies

pip install -r requirements.txt



\# Start the environment server

python -m env.server

\# Server runs at http://localhost:7860



\# In another terminal, run the baseline agent

\# Set your API credentials first:

export API\_BASE\_URL=https://api.groq.com/openai/v1

export MODEL\_NAME=llama-3.3-70b-versatile

export HF\_TOKEN=gsk\_your\_key\_here



python inference.py



Docker

docker build -t ae-triage .

docker run -p 7860:7860 ae-triage



API Endpoints

| Endpoint | Method | Description |
|----------|--------|------------|
| /health  | GET    | Health check — returns 200 if running |
| /reset   | POST   | Start new episode — accepts `{task_id, case_index}` |
| /step    | POST   | Submit action — accepts `{action: {...}}`, returns observation + reward |
| /state   | GET    | Returns current internal state |
| /tasks   | GET    | Lists all available tasks with descriptions |



Environment Variables

| Variable       | Required            | Description                                      |
|---------------|---------------------|--------------------------------------------------|
| API_BASE_URL  | Yes (has default)   | LLM API endpoint                                 |
| MODEL_NAME    | Yes (has default)   | Model identifier for inference                   |
| HF_TOKEN      | Yes (no default)    | API key for LLM calls                            |
| ENV_URL       | No                  | Environment server URL (default: http://localhost:7860) |



Baseline Scores (Llama 3.3 70B via Groq)
| Task               | Difficulty | Avg Score | Avg Reward |
|------------------|-----------|-----------|------------|
| task_seriousness | Easy      | 1.000     | 0.980      |
| task_susar       | Medium    | 0.956     | 1.147      |
| task_full_triage | Hard      | 0.842     | 1.033      |
| **Overall**      | —         | **0.933** | **1.053**  |



Scores demonstrate clear difficulty progression from easy to hard. The hard task genuinely challenges frontier models, with per-case scores ranging from 0.57 to 0.95. The asymmetric reward design means reward values differ from raw scores — correct SUSAR identifications receive bonus rewards above the base grader score.

Case Bank

9 realistic adverse event cases covering:

SUSAR cases (5): Liver failure, cardiac arrest, anaphylactic shock, acute kidney failure, Stevens-Johnson Syndrome

Non-SUSAR cases (4): Mild headache, expected nausea/dizziness, known skin rash, mild fatigue/fever

Drugs: DrugX (fictional), Pembrolizumab, Metformin, Remdesivir, Ibuprofen

Reporters: Physicians, nurses, patients, consumers/caregivers

Sources: Clinical trial sites, patient apps, email, call centers, web forms, urgent phone reports

Live Deployment
HuggingFace Space:  
https://huggingface.co/spaces/cb2324/clinical-trial-ae-triage  

Live API:  
https://cb2324-clinical-trial-ae-triage.hf.space  

Test endpoints:  
Health: https://cb2324-clinical-trial-ae-triage.hf.space/health  
Tasks: https://cb2324-clinical-trial-ae-triage.hf.space/tasks 
Validation
bash
# Run openenv validation
openenv validate

# Run pre-submission validator
./validate.sh https://cb2324-clinical-trial-ae-triage.hf.space .




