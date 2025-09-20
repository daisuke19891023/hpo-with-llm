"""MCP interface implementation using FastMCP."""

from fastmcp import FastMCP

from clean_interfaces.hpo.configuration import default_tuning_config
from clean_interfaces.hpo.executors import DefaultTrialExecutor
from clean_interfaces.hpo.schemas import (
    CodingTask,
    HPOExecutionRequest,
    HPORunConfig,
    ReflectionMode,
)
from clean_interfaces.models.io import WelcomeMessage

from .base import BaseInterface


class MCPInterface(BaseInterface):
    """MCP Interface implementation."""

    def __init__(self) -> None:
        """Initialize the MCP interface."""
        super().__init__()
        self.mcp = FastMCP(name="clean-interfaces-mcp")
        self._setup_commands()

    @property
    def name(self) -> str:
        """Get the interface name.

        Returns:
            str: The interface name

        """
        return "MCP"

    def _setup_commands(self) -> None:
        """Set up MCP commands."""

        def welcome() -> str:
            """Display welcome message."""
            msg = WelcomeMessage()
            return f"{msg.message}\n{msg.hint}"

        self.mcp.tool()(welcome)

        def reflect_hpo(
            task_description: str,
            mode: str = "baseline",
            max_trials: int = 5,
            direction: str = "maximize",
        ) -> dict[str, object]:
            """Execute HPO trials and return a reflection summary."""
            normalized_direction = direction.lower()
            if normalized_direction not in {"minimize", "maximize"}:
                msg = "Direction must be 'minimize' or 'maximize'."
                raise ValueError(msg)

            normalized_mode = mode.strip().lower()
            if normalized_mode == "llm":
                reflection_mode = ReflectionMode.LLM_AUGMENTED
            else:
                try:
                    reflection_mode = ReflectionMode(normalized_mode)
                except ValueError as exc:  # pragma: no cover - defensive path
                    msg = "Mode must be 'baseline' or 'llm'."
                    raise ValueError(msg) from exc

            tuning_config = default_tuning_config()
            search_space = list(tuning_config.to_search_space())
            request = HPOExecutionRequest(
                task=CodingTask(task_id="mcp-task", description=task_description),
                search_space=search_space,
                config=HPORunConfig(
                    max_trials=max_trials,
                    direction=normalized_direction,
                ),
            )

            from clean_interfaces.app import run_hpo_with_reflection

            trial_executor = DefaultTrialExecutor()
            result, reflection = run_hpo_with_reflection(
                request,
                trial_executor=trial_executor,
                mode=reflection_mode,
            )

            return {
                "task": result.task.model_dump(),
                "trials": [trial.model_dump() for trial in result.trials],
                "best_trial": (
                    result.best_trial.model_dump()
                    if result.best_trial is not None
                    else None
                ),
                "reflection": {
                    "mode": reflection.mode.value,
                    "summary": reflection.summary,
                    "suggested_hyperparameters": reflection.suggested_hyperparameters,
                    "insights": [
                        insight.model_dump() for insight in reflection.insights
                    ],
                    "next_actions": list(reflection.next_actions),
                    "critique": reflection.critique,
                    "evidence": reflection.evidence,
                },
            }

        self._reflect_tool = reflect_hpo
        self.mcp.tool()(reflect_hpo)

    def reflect_hpo(
        self,
        *,
        task_description: str,
        mode: str = "baseline",
        max_trials: int = 5,
        direction: str = "maximize",
    ) -> dict[str, object]:
        """Execute the reflection tool using the configured FastMCP app."""
        return self._reflect_tool(
            task_description=task_description,
            mode=mode,
            max_trials=max_trials,
            direction=direction,
        )

    def run(self) -> None:
        """Run the MCP interface."""
        self.logger.info("Starting MCP server")
        self.mcp.run()
