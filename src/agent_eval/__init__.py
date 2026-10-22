"""agent-eval: a pytest-like evaluation harness for LLM agents."""

from agent_eval.config import HarnessConfig, default_config, load_config
from agent_eval.core.evaluator import Evaluator
from agent_eval.core.runner import Runner, RunOutcome
from agent_eval.core.scoring import ScoringEngine
from agent_eval.domain.assertions import AssertionEngine
from agent_eval.domain.models import (
    AgentResult,
    AgentVersion,
    AssertionResult,
    CaseResult,
    Dataset,
    EvalCase,
    Expectation,
    RunRecord,
    Step,
    TokenUsage,
    ToolCall,
)
from agent_eval.integrations.mock import MockAgent, default_mock_agent
from agent_eval.judges.mock import HeuristicJudge, MockJudge

__version__ = "0.1.0"

__all__ = [
    "AgentResult",
    "AgentVersion",
    "AssertionEngine",
    "AssertionResult",
    "CaseResult",
    "Dataset",
    "EvalCase",
    "Evaluator",
    "Expectation",
    "HarnessConfig",
    "HeuristicJudge",
    "MockAgent",
    "MockJudge",
    "RunOutcome",
    "RunRecord",
    "Runner",
    "ScoringEngine",
    "Step",
    "TokenUsage",
    "ToolCall",
    "__version__",
    "default_config",
    "default_mock_agent",
    "load_config",
]
