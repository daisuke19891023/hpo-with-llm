"""Input/Output models for interfaces."""

from pydantic import BaseModel, Field


class WelcomeMessage(BaseModel):
    """Welcome message model for CLI interface."""

    message: str = Field(
        default="Welcome to Clean Interfaces!",
        description="The welcome message to display",
    )
    hint: str = Field(
        default="Type --help for more information",
        description="Hint message for users",
    )
