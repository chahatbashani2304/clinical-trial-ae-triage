
# """
# FastAPI server exposing OpenEnv HTTP API endpoints.
# POST /reset  — Start new episode
# POST /step   — Submit action, get observation + reward
# GET  /state  — Get current state
# GET  /health — Health check
# GET  /tasks  — List available tasks
# """
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Optional
# from .models import Action, Observation, State, TaskID
# from .environment import AETriageEnvironment

# app = FastAPI(
#     title="Clinical Trial AE Triage Environment",
#     description=(
#         "An OpenEnv-compliant RL environment for pharmacovigilance "
#         "adverse event triage. Agents learn to classify AE reports "
#         "for seriousness, causality, expectedness (SUSAR detection), "
#         "MedDRA coding, and regulatory routing."
#     ),
#     version="1.0.0",
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Single environment instance (per-session in production)
# env = AETriageEnvironment()


# # ── Request Models ──

# class ResetRequest(BaseModel):
#     task_id: str = "task_seriousness"
#     case_index: Optional[int] = None

# class StepRequest(BaseModel):
#     action: Action


# # ── Endpoints ──

# @app.get("/health")
# def health():
#     """Health check for HF Spaces deployment verification."""
#     return {"status": "healthy", "environment": "clinical-trial-ae-triage"}


# @app.get("/")
# def root():
#     """Root endpoint — returns environment info."""
#     return {
#         "name": "Clinical Trial AE Triage",
#         "version": "1.0.0",
#         "spec": "openenv",
#         "tasks": ["task_seriousness", "task_susar", "task_full_triage"],
#     }


# @app.post("/reset", response_model=Observation)
# def reset(request: ResetRequest = ResetRequest()):
#     """
#     Reset the environment and start a new episode.

#     Args:
#         task_id: One of 'task_seriousness', 'task_susar', 'task_full_triage'
#         case_index: Optional specific case number (0-8)

#     Returns:
#         Initial observation with the AE report to triage.
#     """
#     try:
#         obs = env.reset(
#             task_id=request.task_id,
#             case_index=request.case_index,
#         )
#         return obs
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))


# @app.post("/step", response_model=Observation)
# def step(request: StepRequest):
#     """
#     Submit an action and receive observation + reward.

#     The agent sends its triage decision, and the environment
#     returns a reward based on correctness, with asymmetric
#     penalties (missed SUSARs penalized heavily).

#     Returns:
#         Observation with reward, feedback, and done flag.
#     """
#     try:
#         obs = env.step(request.action)
#         return obs
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))


# @app.get("/state", response_model=State)
# def state():
#     """Return current environment state."""
#     return env.state()


# @app.get("/tasks")
# def list_tasks():
#     """List all available tasks with descriptions."""
#     from .tasks import TASKS
#     return {
#         tid.value: {
#             "name": info["name"],
#             "difficulty": info["difficulty"],
#             "description": info["description"],
#             "max_steps": info["max_steps"],
#             "num_cases": len(info["cases"]),
#         }
#         for tid, info in TASKS.items()
#     }


# @app.get("/summary")
# def episode_summary():
#     """Return summary of the current/last episode."""
#     return env.get_episode_summary()


# # ── Entry point ──

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=7860)


"""
FastAPI server exposing OpenEnv HTTP API endpoints.
POST /reset  — Start new episode
POST /step   — Submit action, get observation + reward
GET  /state  — Get current state
GET  /health — Health check
GET  /tasks  — List available tasks
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from .models import Action, Observation, State, TaskID
from .environment import AETriageEnvironment

app = FastAPI(
    title="Clinical Trial AE Triage Environment",
    description=(
        "An OpenEnv-compliant RL environment for pharmacovigilance "
        "adverse event triage. Agents learn to classify AE reports "
        "for seriousness, causality, expectedness (SUSAR detection), "
        "MedDRA coding, and regulatory routing."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single environment instance (per-session in production)
env = AETriageEnvironment()


# ── Request Models ──

class ResetRequest(BaseModel):
    task_id: str = "task_seriousness"
    case_index: Optional[int] = None

class StepRequest(BaseModel):
    action: Action


# ── Endpoints ──

@app.get("/health")
def health():
    """Health check for HF Spaces deployment verification."""
    return {"status": "healthy", "environment": "clinical-trial-ae-triage"}


@app.get("/")
def root():
    """Root endpoint — returns environment info."""
    return {
        "name": "Clinical Trial AE Triage",
        "version": "1.0.0",
        "spec": "openenv",
        "tasks": ["task_seriousness", "task_susar", "task_full_triage"],
    }


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest = ResetRequest()):
    """
    Reset the environment and start a new episode.

    Args:
        task_id: One of 'task_seriousness', 'task_susar', 'task_full_triage'
        case_index: Optional specific case number (0-8)

    Returns:
        Initial observation with the AE report to triage.
    """
    try:
        obs = env.reset(
            task_id=request.task_id,
            case_index=request.case_index,
        )
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=Observation)
def step(request: StepRequest):
    """
    Submit an action and receive observation + reward.

    The agent sends its triage decision, and the environment
    returns a reward based on correctness, with asymmetric
    penalties (missed SUSARs penalized heavily).

    Returns:
        Observation with reward, feedback, and done flag.
    """
    try:
        obs = env.step(request.action)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=State)
def state():
    """Return current environment state."""
    return env.state()


@app.get("/tasks")
def list_tasks():
    """List all available tasks with descriptions."""
    from .tasks import TASKS
    return {
        tid.value: {
            "name": info["name"],
            "difficulty": info["difficulty"],
            "description": info["description"],
            "max_steps": info["max_steps"],
            "num_cases": len(info["cases"]),
        }
        for tid, info in TASKS.items()
    }


@app.get("/summary")
def episode_summary():
    """Return summary of the current/last episode."""
    return env.get_episode_summary()


# ── Entry point ──

def main():
    """Entry point for `server` command via pyproject.toml scripts."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
