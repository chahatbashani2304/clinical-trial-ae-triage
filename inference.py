
"""
Inference Script — Clinical Trial AE Triage OpenEnv
===================================
MANDATORY
- Environment variables:
   API_BASE_URL   The API endpoint for the LLM.
   MODEL_NAME     The model identifier to use for inference.
   HF_TOKEN       Your Hugging Face / API key.

- Defaults are set only for API_BASE_URL and MODEL_NAME:
   API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
   MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

- The inference script must be named `inference.py` and placed in the root directory
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- [START] task=<task_name> env=<benchmark> model=<model_name>
- [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
- [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import json
import time
import textwrap
import requests
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# ── Configuration (defaults only for API_BASE_URL and MODEL_NAME) ──

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK = "clinical-trial-ae-triage"
MAX_STEPS = 3
TEMPERATURE = 0.1
MAX_TOKENS = 800

TASKS = [
    {"id": "task_seriousness", "name": "seriousness-classification", "max_steps": 2},
    {"id": "task_susar", "name": "susar-detection", "max_steps": 3},
    {"id": "task_full_triage", "name": "full-triage-routing", "max_steps": 5},
]
NUM_CASES = 9

# ── System Prompts ──

SYSTEM_PROMPTS = {
    "task_seriousness": textwrap.dedent("""
        You are a pharmacovigilance expert. You will receive an adverse event
        (AE) report. Classify it as 'serious' or 'non_serious' based on ICH E2A
        criteria: death, life-threatening, hospitalization, disability, congenital
        anomaly, or medically significant events.

        Respond ONLY with a JSON object (no markdown, no backticks):
        {"seriousness": "serious" or "non_serious", "seriousness_reason": "brief reason"}
    """).strip(),

    "task_susar": textwrap.dedent("""
        You are a pharmacovigilance expert performing SUSAR detection.
        Assess the adverse event report for:
        1. Seriousness: "serious" or "non_serious"
        2. Causality: "related", "possibly_related", "unlikely", or "unrelated"
        3. Expectedness: "expected" or "unexpected" (compare against the drug's
           known side effects provided)
        4. SUSAR decision: "SUSAR" if serious + related + unexpected, else "NOT_SUSAR"

        Respond ONLY with a JSON object (no markdown, no backticks):
        {"seriousness": "...", "seriousness_reason": "...", "causality": "...", "expectedness": "...", "triage_decision": "..."}
    """).strip(),

    "task_full_triage": textwrap.dedent("""
        You are a senior pharmacovigilance specialist performing complete AE triage.
        Provide all of the following:
        1. Seriousness: "serious" or "non_serious" with reason
        2. Causality: "related", "possibly_related", "unlikely", or "unrelated"
        3. Expectedness: "expected" or "unexpected"
        4. SUSAR decision: "SUSAR" or "NOT_SUSAR" or "NEEDS_REVIEW"
        5. MedDRA coding: extract AE terms and map to Preferred Terms
        6. Regulatory routing: "FDA", "EMA", "PMDA", "MHRA", or "NONE"
        7. Expedited reporting: true or false
        8. Brief ICSR narrative summary (2-3 sentences)

        Respond ONLY with a JSON object (no markdown, no backticks):
        {"seriousness": "...", "seriousness_reason": "...", "causality": "...", "expectedness": "...", "triage_decision": "...", "meddra_codings": [{"raw_term": "...", "preferred_term": "...", "soc": "..."}], "regulatory_route": "...", "expedited_report": true, "narrative_summary": "..."}
    """).strip(),
}


# ── Structured Logging (MANDATORY FORMAT) ──

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Helper Functions ──

def build_user_prompt(obs: dict) -> str:
    """Build the user prompt from the observation."""
    ae = obs["ae_report"]
    parts = [
        f"Adverse Event Report: {ae['report_id']}",
        f"Drug: {ae['drug_name']}",
        f"Reporter: {ae['reporter_type']} ({ae['report_source']})",
        f"Known Side Effects (from RSI): {', '.join(ae['known_side_effects'])}",
        f"Narrative: {ae['narrative']}",
    ]
    if obs.get("feedback"):
        parts.append(f"Environment Feedback: {obs['feedback']}")
    return "\n".join(parts)


def parse_llm_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip().rstrip("`")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {}


def get_model_action(client: OpenAI, task_id: str, obs: dict) -> str:
    """Call LLM and get action as JSON string."""
    sys_prompt = SYSTEM_PROMPTS[task_id]
    user_prompt = build_user_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        time.sleep(1)  # Rate limit protection
        return text if text else "{}"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        time.sleep(5)  # Wait longer on rate limit error
        return "{}"


# ── Main Inference Loop ──

def run_task(client: OpenAI, task_config: dict) -> float:
    """Run one task across all cases. Returns average score."""
    task_id = task_config["id"]
    task_name = task_config["name"]
    max_steps = task_config["max_steps"]

    all_scores = []

    for case_idx in range(NUM_CASES):
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False
        error_msg = None

        log_start(task=f"{task_name}-case{case_idx}", env=BENCHMARK, model=MODEL_NAME)

        try:
            # Reset environment
            reset_resp = requests.post(
                f"{ENV_URL}/reset",
                json={"task_id": task_id, "case_index": case_idx},
                timeout=30,
            )
            if reset_resp.status_code != 200:
                error_msg = f"reset-failed-{reset_resp.status_code}"
                log_step(step=1, action="reset", reward=0.0, done=True, error=error_msg)
                log_end(success=False, steps=0, score=0.0, rewards=[])
                all_scores.append(0.0)
                continue

            obs = reset_resp.json()

            for step in range(1, max_steps + 1):
                if obs.get("done", False):
                    break

                # Get action from LLM
                llm_text = get_model_action(client, task_id, obs)
                parsed = parse_llm_response(llm_text)

                # Create action string for logging
                action_str = json.dumps(parsed, separators=(",", ":"))
                if len(action_str) > 200:
                    action_str = action_str[:200] + "..."

                # Step environment
                step_resp = requests.post(
                    f"{ENV_URL}/step",
                    json={"action": parsed},
                    timeout=30,
                )

                if step_resp.status_code != 200:
                    error_msg = f"step-failed-{step_resp.status_code}"
                    log_step(step=step, action=action_str, reward=0.0, done=True, error=error_msg)
                    steps_taken = step
                    break

                obs = step_resp.json()
                reward = obs.get("reward", 0.0) or 0.0
                done = obs.get("done", False)
                step_error = None

                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=action_str, reward=reward, done=done, error=step_error)

                if done:
                    break

            # Calculate score (use last grader score if available)
            if obs.get("score") is not None:
                score = float(obs["score"])
            elif rewards:
                score = max(0.0, min(1.0, sum(rewards) / len(rewards)))
            else:
                score = 0.0

            score = min(max(score, 0.0), 1.0)
            success = score >= 0.5

        except Exception as exc:
            error_msg = str(exc)
            print(f"[DEBUG] Exception: {exc}", flush=True)

        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
            all_scores.append(score)

    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return avg_score


def main() -> None:
    """Run baseline agent on all 3 tasks."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Verify environment is running
    try:
        health = requests.get(f"{ENV_URL}/health", timeout=10)
        assert health.status_code == 200
        print(f"[DEBUG] Environment healthy at {ENV_URL}", flush=True)
    except Exception as e:
        print(f"[DEBUG] ERROR: Cannot reach environment at {ENV_URL}: {e}", flush=True)
        print("[DEBUG] Start the server: python -m env.server", flush=True)
        return

    task_scores = []
    for task_config in TASKS:
        avg = run_task(client, task_config)
        task_scores.append(avg)
        print(f"[DEBUG] {task_config['name']} avg_score={avg:.4f}", flush=True)

    overall = sum(task_scores) / len(task_scores) if task_scores else 0.0
    print(f"[DEBUG] Overall average score: {overall:.4f}", flush=True)


if __name__ == "__main__":
    main()


