from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    HF_TOKEN: str
