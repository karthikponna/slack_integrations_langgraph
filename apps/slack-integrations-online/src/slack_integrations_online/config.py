import os
from loguru import logger
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
    )

    # OpenAI Configuration
    OPENAI_API_KEY: str = Field(
        description="API key for OpenAI service authentication."
    )

    # Slack Configuration
    SLACK_BOT_TOKEN: str = Field(
        description="Bot token for slack."
    )

    SLACK_APP_TOKEN: str = Field(
        description="App token of Socket model for slack."
    )

    # MongoDB Configuration
    MONGODB_DATABASE_NAME: str = Field(
        default="slack_integrations",
        description="Name of the MongoDB database.",
    )

    MONGODB_URI: str = Field(
        default="mongodb://slack:slack@localhost:27017/?directConnection=true",
        description="Connection URI for the local MongoDB Atlas instance.",
    )

    # Langsmit Configuration
    LANGCHAIN_TRACING_V2: str = Field(
        default="true",
        description="Enabling tracking for langsmith"
    )

    LANGCHAIN_API_KEY: str = Field(
        description="Langsmith api key for traces"
    )

    LANGCHAIN_PROJECT: str = Field(
        description="Project name"
    )


    @field_validator("OPENAI_API_KEY")
    @classmethod
    def check_not_empty(cls, value:str, info) -> str:
        if not value or value.strip() == "":
            logger.error(f"{info.field_name} cannot be empty")
            raise ValueError(f"{info.field_name} cannot be empty.")
        
        return value
    

try:
    settings = Settings()
    
    # Export LangSmith environment variables for LangChain runtime
    os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT

except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    raise SystemExit(e)