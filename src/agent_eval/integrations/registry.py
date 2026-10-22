"""Agent adapter registry.

Resolves ``agent.type`` from configuration to a concrete adapter. LangGraph is an
optional extra and is imported lazily, so the core never depends on it.
"""

from __future__ import annotations

import importlib
from typing import Any, cast

from agent_eval.config import AgentConfig
from agent_eval.ports.agent import AgentAdapter

#: Attributes a custom agent must expose to satisfy :class:`AgentAdapter`.
ADAPTER_ATTRS = ("name", "version", "framework", "run")


def create_agent(config: AgentConfig) -> AgentAdapter:
    """Build the adapter described by ``config``.

    Args:
        config: The ``agent`` section of the harness configuration.

    Returns:
        A ready-to-use adapter.

    Raises:
        ValueError: If the agent type or factory path is unknown or invalid.
    """
    if config.factory:
        return _from_factory(config.factory)

    agent_type = config.type.lower()
    if agent_type == "mock":
        from agent_eval.integrations.mock import default_mock_agent

        return default_mock_agent()
    if agent_type in ("langgraph", "lang_graph", "example"):
        return _langgraph(config)
    raise ValueError(
        f"Unknown agent type {config.type!r}. Use 'mock', 'langgraph', 'example', "
        "or set agent.factory to a 'module:callable' path."
    )


def _langgraph(config: AgentConfig) -> AgentAdapter:
    """Build the LangGraph adapter, importing the extra on demand.

    ``agent.options`` accepts three ways of naming a graph:

    * ``graph``: a compiled graph object, or an import path to one.
    * ``graph_factory``: an import path to a *callable* returning a graph.
    * neither: falls back to the bundled deterministic example graph.
    """
    try:
        from agent_eval.integrations.langgraph.adapter import LangGraphAdapter
    except ImportError as exc:
        raise ValueError(
            "The langgraph agent needs the 'langgraph' extra: pip install agent-eval[langgraph]"
        ) from exc

    graph = config.options.get("graph")
    factory_path = config.options.get("graph_factory")
    if isinstance(factory_path, str):
        graph = _build(_load_object(factory_path), factory_path)
    elif isinstance(graph, str):
        graph = _load_object(graph)
    elif graph is None:
        from agent_eval.integrations.langgraph.examples import default_graph

        graph = default_graph()

    return cast(
        "AgentAdapter",
        LangGraphAdapter(
            graph=graph,
            options=config.options,
            input_key=_option(config, "input_key", "input"),
            output_key=_option(config, "output_key", "output"),
            messages_key=_option(config, "messages_key", "messages"),
            recursion_limit=_int_option(config, "recursion_limit"),
        ),
    )


def _build(factory: Any, path: str) -> Any:
    """Call a graph factory, giving an actionable error when it is not one."""
    if not callable(factory):
        raise ValueError(
            f"graph_factory {path!r} is not callable; it must return a compiled graph."
        )
    return factory()


def _option(config: AgentConfig, key: str, default: str) -> str:
    """Read a string option, ignoring non-string values such as ``None``."""
    value = config.options.get(key, default)
    return value if isinstance(value, str) else default


def _int_option(config: AgentConfig, key: str) -> int | None:
    """Read an integer option, ignoring anything that is not a usable int."""
    value = config.options.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _from_factory(factory_path: str) -> AgentAdapter:
    """Resolve a ``module:callable`` factory and use what it returns.

    A factory may return either a ready adapter or a zero-argument callable that
    builds one, because both spellings are natural in configuration and neither
    is more obviously correct.
    """
    produced = _load_object(factory_path)
    if not _is_adapter(produced) and callable(produced):
        produced = produced()
    return _ensure_adapter(produced)


def _is_adapter(candidate: Any) -> bool:
    """Whether ``candidate`` already satisfies the adapter protocol."""
    return all(hasattr(candidate, attr) for attr in ADAPTER_ATTRS)


def _ensure_adapter(candidate: Any) -> AgentAdapter:
    """Validate that ``candidate`` satisfies the adapter protocol."""
    missing = [attr for attr in ADAPTER_ATTRS if not hasattr(candidate, attr)]
    if missing:
        raise ValueError(
            f"agent.factory produced {type(candidate).__name__}, which is missing "
            f"required adapter attribute(s): {', '.join(missing)}."
        )
    return cast("AgentAdapter", candidate)


def _load_object(path: str) -> Any:
    """Import ``module:callable`` (or ``module.attr``) and return the object."""
    if ":" in path:
        module_name, _, attribute = path.partition(":")
    else:
        module_name, _, attribute = path.rpartition(".")
    if not module_name or not attribute:
        raise ValueError(f"Invalid import path {path!r}; expected 'module:callable'.")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"Could not import module {module_name!r}: {exc}") from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ValueError(f"Module {module_name!r} has no attribute {attribute!r}.") from exc
