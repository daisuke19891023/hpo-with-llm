"""Application builder for clean interfaces.

This module handles the construction and configuration of the application,
including interface selection, storage configuration, and future database setup.
"""

from pathlib import Path

from dotenv import load_dotenv

from clean_interfaces.hpo.backends import HPOSearchBackend, InMemorySearchBackend
from clean_interfaces.hpo.logging import HPOTrialLogger
from clean_interfaces.hpo.orchestrator import HPOOrchestrator, TrialExecutorProtocol
from clean_interfaces.hpo.reflection import ReflectionAgent
from clean_interfaces.hpo.schemas import (
    HPOExecutionRequest,
    HPOOptimizationResult,
    HPOReflectionRequest,
    HPOReflectionResponse,
    ReflectionMode,
)
from clean_interfaces.llm import LLMClientFactory
from clean_interfaces.interfaces.factory import InterfaceFactory
from clean_interfaces.utils.logger import configure_logging, get_logger
from clean_interfaces.utils.settings import get_interface_settings, get_settings


class Application:
    """Main application class that orchestrates components."""

    def __init__(self, dotenv_path: Path | None = None) -> None:
        """Initialize the application.

        Args:
            dotenv_path: Optional path to .env file to load

        """
        # Load environment variables from .env file if provided
        if dotenv_path:
            load_dotenv(dotenv_path, override=True)
        else:
            # Load from default .env file if it exists
            load_dotenv(override=True)

        # Configure logging first
        settings = get_settings()

        # Configure logging
        configure_logging(
            log_level=settings.log_level,
            log_format=settings.log_format,
            log_file=settings.log_file_path,
        )
        self.logger = get_logger(__name__)

        # Initialize interface
        self.interface_factory = InterfaceFactory()
        self.interface = self.interface_factory.create_from_settings()

        self.logger.info(
            "Application initialized",
            interface=self.interface.name,
            settings=get_interface_settings().model_dump(),
            dotenv_loaded=str(dotenv_path) if dotenv_path else "default",
        )

    def run(self) -> None:
        """Run the application."""
        self.logger.info("Starting application", interface=self.interface.name)

        try:
            self.interface.run()
        except Exception as e:
            self.logger.error("Application error", error=str(e))
            raise
        finally:
            self.logger.info("Application shutting down")

    def create_hpo_orchestrator(
        self,
        trial_executor: TrialExecutorProtocol,
        backend: HPOSearchBackend | None = None,
        trial_logger: HPOTrialLogger | None = None,
    ) -> HPOOrchestrator:
        """Create an HPO orchestrator using application defaults."""
        return create_hpo_orchestrator(
            trial_executor=trial_executor,
            backend=backend,
            trial_logger=trial_logger,
        )

    def create_reflection_agent(
        self, *, llm_factory: LLMClientFactory | None = None,
    ) -> ReflectionAgent:
        """Create a reflection agent with default configuration."""
        return create_reflection_agent(llm_factory=llm_factory)


def create_hpo_orchestrator(
    *,
    trial_executor: TrialExecutorProtocol,
    backend: HPOSearchBackend | None = None,
    trial_logger: HPOTrialLogger | None = None,
) -> HPOOrchestrator:
    """Construct a configured HPO orchestrator."""
    selected_backend = backend or InMemorySearchBackend()
    return HPOOrchestrator(
        search_backend=selected_backend,
        trial_executor=trial_executor,
        trial_logger=trial_logger,
    )


def create_reflection_agent(
    llm_factory: LLMClientFactory | None = None,
) -> ReflectionAgent:
    """Construct the default reflection agent."""
    return ReflectionAgent(llm_factory=llm_factory)


def run_hpo_experiment(
    request: HPOExecutionRequest,
    *,
    trial_executor: TrialExecutorProtocol,
    backend: HPOSearchBackend | None = None,
    trial_logger: HPOTrialLogger | None = None,
) -> HPOOptimizationResult:
    """Execute a full HPO experiment for the provided request."""
    orchestrator = create_hpo_orchestrator(
        trial_executor=trial_executor,
        backend=backend,
        trial_logger=trial_logger,
    )
    return orchestrator.optimize(
        task=request.task,
        search_space=request.search_space,
        config=request.config,
    )


def run_hpo_with_reflection(
    request: HPOExecutionRequest,
    *,
    trial_executor: TrialExecutorProtocol,
    backend: HPOSearchBackend | None = None,
    mode: ReflectionMode = ReflectionMode.BASELINE,
    trial_logger: HPOTrialLogger | None = None,
) -> tuple[HPOOptimizationResult, HPOReflectionResponse]:
    """Execute an HPO experiment and produce a reflection summary."""
    result = run_hpo_experiment(
        request,
        trial_executor=trial_executor,
        backend=backend,
        trial_logger=trial_logger,
    )
    agent = create_reflection_agent()
    reflection_request = HPOReflectionRequest(
        task=request.task,
        config=request.config,
        search_space=request.search_space,
        history=result.trials,
        mode=mode,
    )
    reflection = agent.reflect(reflection_request)
    return result, reflection


def create_app(dotenv_path: Path | None = None) -> Application:
    """Create an application instance.

    Args:
        dotenv_path: Optional path to .env file to load

    Returns:
        Application: Configured application instance

    """
    return Application(dotenv_path=dotenv_path)


def run_app(dotenv_path: Path | None = None) -> None:
    """Create and run the application.

    Args:
        dotenv_path: Optional path to .env file to load

    """
    app = create_app(dotenv_path=dotenv_path)
    app.run()
