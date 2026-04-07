"""
Inference Script - Clinical Trial AE Triage OpenEnv
===================================
MANDATORY
- Environment variables:
   API_BASE_URL   The API endpoint for the LLM.
   MODEL_NAME     The model identifier to use for inference.
   HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT
- [START] task=<task_name> env=<benchmark> model=<model_name>
- [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
- [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import time
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests
from openai import OpenAI

# ── Configuration (defaults only for API_BASE_URL and MODEL_NAME) ──

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

BENCHMARK = "clinical-trial-ae-triage"
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
    "task_seriousness": (
        "You are a pharmacovigilance expert. Classify the adverse event as 'serious' or 'non_serious' "
        "based on ICH E2A criteria: death, life-threatening, hospitalization, disability, congenital anomaly, "
        "or medically significant. Respond ONLY with a JSON object, no markdown, no backticks: "
        '{"seriousness": "serious", "seriousness_reason": "brief reason"}'
    ),
    "task_susar": (
        "You are a pharmacovigilance expert. Assess the adverse event for SUSAR criteria: "
        "1. Seriousness: serious or non_serious. "
        "2. Causality: related, possibly_related, unlikely, or unrelated. "
        "3. Expectedness: expected or unexpected (compare against known side effects provided). "
        "4. SUSAR decision: SUSAR if serious+related+unexpected, else NOT_SUSAR. "
        "Respond ONLY with a JSON object, no markdown, no backticks: "
        '{"seriousness": "serious", "seriousness_reason": "reason", "causality": "related", '
        '"expectedness": "unexpected", "triage_decision": "SUSAR"}'
    ),
    "task_full_triage": (
        "You are a senior pharmacovigilance specialist. Provide complete AE triage: "
        "seriousness with reason, causality, expectedness, SUSAR decision, "
        "MedDRA coding of AE terms, regulatory routing (FDA/EMA/PMDA/MHRA/NONE), "
        "expedited reporting (true/false), and brief ICSR narrative (2-3 sentences). "
        "IMPORTANT: Use exactly these enum values - seriousness must be 'serious' or 'non_serious' "
        "(with underscore, NOT hyphen). causality must be 'related', 'possibly_related', 'unlikely', "
        "or 'unrelated'. expectedness must be 'expected' or 'unexpected'. "
        "triage_decision must be 'SUSAR', 'NOT_SUSAR', or 'NEEDS_REVIEW'. "
        "regulatory_route must be 'FDA', 'EMA', 'PMDA', 'MHRA', or 'NONE'. "
        "Respond ONLY with a JSON object, no markdown, no backticks: "
        '{"seriousness": "serious", "seriousness_reason": "reason", "causality": "related", '
        '"expectedness": "unexpected", "triage_decision": "SUSAR", '
        '"meddra_codings": [{"raw_term": "term", "preferred_term": "PT", "soc": "SOC"}], '
        '"regulatory_route": "FDA", "expedited_report": true, "narrative_summary": "summary"}'
    ),
}


# ── Structured Logging ──

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    e = error if error else "null"
    d = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={d} error={e}", flush=True)

def log_end(success, steps, score, rewards):
    r = ",".join(f"{x:.2f}" for x in rewards) if rewards else "0.01"
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={r}", flush=True)


# ── Helpers ──

def wait_for_env(url, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def safe_request(method, url, **kwargs):
    kwargs.setdefault("timeout", 30)
    last_err = None
    for attempt in range(3):
        try:
            if method == "GET":
                resp = requests.get(url, **kwargs)
            else:
                resp = requests.post(url, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_err = e
            time.sleep(2)
    raise last_err


def parse_json(text):
    if not text:
        return {}
    text = text.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break
    text = text.strip().rstrip("`")
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        s, e = text.find("{"), text.rfind("}") + 1
        if s >= 0 and e > s:
            return json.loads(text[s:e])
    except Exception:
        pass
    return {}


def fix_enums(parsed):
    """Fix common LLM enum mistakes like non-serious -> non_serious."""
    if not parsed:
        return parsed
    fixes = {
        "seriousness": {"non-serious": "non_serious", "nonserious": "non_serious"},
        "causality": {"possibly related": "possibly_related", "possibly-related": "possibly_related"},
        "triage_decision": {"NOT SUSAR": "NOT_SUSAR", "NEEDS REVIEW": "NEEDS_REVIEW",
                           "not_susar": "NOT_SUSAR", "not susar": "NOT_SUSAR"},
    }
    for field, mapping in fixes.items():
        if field in parsed and isinstance(parsed[field], str):
            val = parsed[field].strip()
            if val in mapping:
                parsed[field] = mapping[val]
    return parsed


def build_prompt(obs):
    try:
        ae = obs.get("ae_report") or {}
        lines = [
            f"Report: {ae.get('report_id', 'N/A')}",
            f"Drug: {ae.get('drug_name', 'N/A')}",
            f"Reporter: {ae.get('reporter_type', 'N/A')} ({ae.get('report_source', 'N/A')})",
            f"Known Side Effects: {', '.join(ae.get('known_side_effects', []))}",
            f"Narrative: {ae.get('narrative', 'N/A')}",
        ]
        fb = obs.get("feedback")
        if fb:
            lines.append(f"Feedback: {fb}")
        return "\n".join(lines)
    except Exception:
        return "Error reading observation"


def call_llm(client, task_id, obs):
    prompt = build_prompt(obs)
    sys_prompt = SYSTEM_PROMPTS.get(task_id, "")
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            text = (resp.choices[0].message.content or "").strip()
            time.sleep(1)
            result = parse_json(text)
            if result:
                return fix_enums(result)
        except Exception as e:
            print(f"[DEBUG] LLM attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(3)
    return {}


def clamp(score):
    """Clamp score strictly between 0 and 1."""
    return max(0.001, min(0.999, score))


# ── Task Runner ──

def run_task(client, task_config):
    task_id = task_config["id"]
    task_name = task_config["name"]
    max_steps = task_config["max_steps"]
    all_scores = []

    for case_idx in range(NUM_CASES):
        rewards = []
        steps_taken = 0
        score = 0.001
        success = False

        log_start(task=f"{task_name}-case{case_idx}", env=BENCHMARK, model=MODEL_NAME)

        try:
            # Reset
            try:
                reset_resp = safe_request("POST", f"{ENV_URL}/reset",
                    json={"task_id": task_id, "case_index": case_idx})
                obs = reset_resp.json()
            except Exception as e:
                log_step(step=1, action="reset_failed", reward=0.01, done=True, error=str(e)[:100])
                log_end(success=False, steps=1, score=0.001, rewards=[0.01])
                all_scores.append(0.001)
                continue

            # Steps
            for step_num in range(1, max_steps + 1):
                if obs.get("done", False):
                    break

                parsed = call_llm(client, task_id, obs)
                action_str = json.dumps(parsed, separators=(",", ":"))
                if len(action_str) > 150:
                    action_str = action_str[:147] + "..."

                try:
                    step_resp = safe_request("POST", f"{ENV_URL}/step",
                        json={"action": parsed})
                    obs = step_resp.json()
                except Exception as e:
                    rewards.append(0.01)
                    steps_taken = step_num
                    log_step(step=step_num, action=action_str, reward=0.01, done=True, error=str(e)[:100])
                    break

                reward = float(obs.get("reward") or 0.01)
                done = bool(obs.get("done", False))
                rewards.append(reward)
                steps_taken = step_num

                log_step(step=step_num, action=action_str, reward=reward, done=done, error=None)

                if done:
                    break

            # Score
            if obs.get("score") is not None:
                score = float(obs["score"])
            elif rewards:
                score = sum(rewards) / max(len(rewards), 1)
            score = clamp(score)
            success = score >= 0.5

        except Exception as e:
            print(f"[DEBUG] Case {case_idx} error: {e}", flush=True)
            if steps_taken == 0:
                steps_taken = 1
                rewards = [0.01]

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards if rewards else [0.01])
        all_scores.append(score)

    return sum(all_scores) / len(all_scores) if all_scores else 0.001


# ── Main ──

def main():
    print(f"[DEBUG] Connecting to environment at {ENV_URL}...", flush=True)
    if not wait_for_env(ENV_URL, timeout=180):
        print(f"[DEBUG] Environment unreachable after 180s", flush=True)
        for task in TASKS:
            for ci in range(NUM_CASES):
                log_start(task=f"{task['name']}-case{ci}", env=BENCHMARK, model=MODEL_NAME)
                log_step(step=1, action="env_timeout", reward=0.01, done=True, error="env unreachable")
                log_end(success=False, steps=1, score=0.001, rewards=[0.01])
        return

    print(f"[DEBUG] Environment ready", flush=True)
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "no-key-provided")

    scores = []
    for task in TASKS:
        try:
            avg = run_task(client, task)
        except Exception as e:
            print(f"[DEBUG] Task failed: {e}", flush=True)
            avg = 0.001
        scores.append(avg)
        print(f"[DEBUG] {task['name']} score={avg:.4f}", flush=True)

    overall = sum(scores) / len(scores) if scores else 0.001
    print(f"[DEBUG] Overall: {overall:.4f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[DEBUG] Fatal: {e}", flush=True)
        for task in TASKS:
            for ci in range(NUM_CASES):
                log_start(task=f"{task['name']}-case{ci}", env=BENCHMARK, model=MODEL_NAME)
                log_step(step=1, action="fatal_error", reward=0.01, done=True, error=str(e)[:100])
                log_end(success=False, steps=1, score=0.001, rewards=[0.01])

