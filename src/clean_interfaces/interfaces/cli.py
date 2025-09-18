"""CLI interface implementation using Typer."""

from pathlib import Path

import typer
from rich.console import Console

from clean_interfaces.hpo.configuration import (
    default_tuning_config,
    load_tuning_config,
)
from clean_interfaces.hpo.executors import default_trial_executor
from clean_interfaces.hpo.schemas import (
    CodingTask,
    HPOExecutionRequest,
    HPOOptimizationResult,
    HPOReflectionResponse,
    HPORunConfig,
    ReflectionMode,
)
from clean_interfaces.models.io import WelcomeMessage

from .base import BaseInterface

# Configure console for better test compatibility
# Force terminal mode even in non-TTY environments
console = Console(force_terminal=True, force_interactive=False)


def _normalise_direction(direction: str) -> str:
    """Validate and normalise the optimisation direction."""
    normalized_direction = direction.lower()
    if normalized_direction not in {"minimize", "maximize"}:
        message = "Direction must be 'minimize' or 'maximize'."
        raise typer.BadParameter(message)
    return normalized_direction


def _resolve_reflection_mode(mode: str) -> ReflectionMode:
    """Convert CLI input into a :class:`ReflectionMode`."""
    normalized_mode = mode.strip().lower()
    if normalized_mode == "llm":
        return ReflectionMode.LLM_AUGMENTED
    try:
        return ReflectionMode(normalized_mode)
    except ValueError as exc:  # pragma: no cover - defensive path
        message = "Mode must be 'baseline' or 'llm'."
        raise typer.BadParameter(message) from exc


def _display_reflection_result(
    task: CodingTask,
    result: HPOOptimizationResult,
    reflection: HPOReflectionResponse,
) -> None:
    """Render reflection output to the console."""
    console.print(
        f"Executed {len(result.trials)} trials for task '{task.task_id}'.",
    )
    if result.best_trial is not None:
        console.print(f"Best score: {result.best_trial.score:.3f}")
    else:
        console.print("No successful trials were recorded.")

    console.print(
        f"Reflection summary ({reflection.mode.value}): {reflection.summary}",
    )
    if reflection.insights:
        console.print("Insights:")
        for insight in reflection.insights:
            console.print(f" - {insight.title}: {insight.detail}")
    hyperparameters_msg = (
        "Suggested hyperparameters: "
        f"{reflection.suggested_hyperparameters}"
    )
    console.print(hyperparameters_msg)
    if reflection.next_actions:
        console.print("Next actions:")
        for action in reflection.next_actions:
            console.print(f" - {action}")
    if reflection.critique:
        console.print(f"Critique: {reflection.critique}")
    console.file.flush()


RUN_HPO_TASK_OPTION = typer.Option(
    ...,
    "--task",
    "-t",
    prompt=True,
    help="Description of the coding task to optimize.",
)
RUN_HPO_MAX_TRIALS_OPTION = typer.Option(
    5,
    "--max-trials",
    "-n",
    min=1,
    help="Maximum number of trials to run.",
)
RUN_HPO_DIRECTION_OPTION = typer.Option(
    "maximize",
    "--direction",
    "-d",
    help="Optimization direction (minimize or maximize).",
)
RUN_HPO_SEARCH_SPACE_OPTION = typer.Option(
    None,
    "--search-space-config",
    "-s",
    help="Path to a YAML file describing the hyperparameters to tune.",
)

REFLECT_TASK_OPTION = typer.Option(
    ...,
    "--task",
    "-t",
    prompt=True,
    help="Description of the coding task to optimise.",
)
REFLECT_MAX_TRIALS_OPTION = typer.Option(
    5,
    "--max-trials",
    "-n",
    min=1,
    help="Maximum number of trials to run before reflecting.",
)
REFLECT_DIRECTION_OPTION = typer.Option(
    "maximize",
    "--direction",
    "-d",
    help="Optimisation direction (minimize or maximize).",
)
REFLECT_MODE_OPTION = typer.Option(
    "baseline",
    "--mode",
    "-m",
    help="Reflection mode to apply (baseline or llm).",
)
REFLECT_SEARCH_SPACE_OPTION = typer.Option(
    None,
    "--search-space-config",
    "-s",
    help="Optional YAML file describing the hyperparameters to tune.",
)


class CLIInterface(BaseInterface):
    """Command Line Interface implementation."""

    def __init__(self) -> None:
        """Initialize the CLI interface."""
        super().__init__()  # Call BaseComponent's __init__ for logger initialization
        self.app = typer.Typer(
            name="clean-interfaces",
            help="Clean Interfaces CLI",
            add_completion=False,
        )
        self._setup_commands()

    @property
    def name(self) -> str:
        """Get the interface name.

        Returns:
            str: The interface name

        """
        return "CLI"

    def _setup_commands(self) -> None:
        """Set up CLI commands."""
        # Set the default command to welcome
        self.app.command(name="welcome")(self.welcome)
        self.app.command(name="run-hpo")(self.run_hpo)
        self.app.command(name="reflect-hpo")(self.reflect_hpo)

        # Add a callback that shows welcome when no command is specified
        self.app.callback(invoke_without_command=True)(self._main_callback)

    def _main_callback(self, ctx: typer.Context) -> None:  # pragma: no cover
        """Run when no subcommand is provided."""
        if ctx.invoked_subcommand is None:
            self.welcome()
            # Ensure we exit cleanly after showing welcome
            raise typer.Exit(0)

    def welcome(self) -> None:
        """Display welcome message."""
        msg = WelcomeMessage()
        # Use console for output (configured for E2E test compatibility)
        console.print(msg.message)
        console.print(msg.hint)
        # Force flush to ensure output is visible
        console.file.flush()

    def run_hpo(
        self,
        task_description: str = RUN_HPO_TASK_OPTION,
        max_trials: int = RUN_HPO_MAX_TRIALS_OPTION,
        direction: str = RUN_HPO_DIRECTION_OPTION,
        search_space_config: Path | None = RUN_HPO_SEARCH_SPACE_OPTION,
    ) -> None:
        """Execute a simulated HPO run using the default orchestrator."""
        normalized_direction = _normalise_direction(direction)

        if search_space_config is not None:
            tuning_config = load_tuning_config(search_space_config)
        else:
            tuning_config = default_tuning_config()

        search_space = tuning_config.to_search_space()

        task = CodingTask(task_id="cli-task", description=task_description)
        config = HPORunConfig(max_trials=max_trials, direction=normalized_direction)

        from clean_interfaces.app import create_hpo_orchestrator

        orchestrator = create_hpo_orchestrator(trial_executor=default_trial_executor)
        result = orchestrator.optimize(task, search_space, config)

        console.print(
            f"Executed {len(result.trials)} trials for task '{task.task_id}'.",
        )
        if result.best_trial is not None:
            console.print(
                f"Best score: {result.best_trial.score:.3f}",
            )
            console.print(
                f"Best hyperparameters: {result.best_trial.hyperparameters}",
            )
        else:
            console.print("No successful trials were recorded.")

        console.file.flush()

    def reflect_hpo(
        self,
        task_description: str = REFLECT_TASK_OPTION,
        max_trials: int = REFLECT_MAX_TRIALS_OPTION,
        direction: str = REFLECT_DIRECTION_OPTION,
        mode: str = REFLECT_MODE_OPTION,
        search_space_config: Path | None = REFLECT_SEARCH_SPACE_OPTION,
    ) -> None:
        """Run an HPO loop and produce a reflection summary."""
        normalized_direction = _normalise_direction(direction)
        reflection_mode = _resolve_reflection_mode(mode)

        if search_space_config is not None:
            tuning_config = load_tuning_config(search_space_config)
        else:
            tuning_config = default_tuning_config()

        search_space = tuning_config.to_search_space()

        task = CodingTask(task_id="cli-task", description=task_description)
        config = HPORunConfig(max_trials=max_trials, direction=normalized_direction)

        from clean_interfaces.app import run_hpo_with_reflection

        execution_request = HPOExecutionRequest(
            task=task,
            search_space=list(search_space),
            config=config,
        )
        result, reflection = run_hpo_with_reflection(
            execution_request,
            trial_executor=default_trial_executor,
            mode=reflection_mode,
        )

        _display_reflection_result(task, result, reflection)

    def run(self) -> None:
        """Run the CLI interface."""
        # Let Typer handle the command parsing
        self.app()
