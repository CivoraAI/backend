from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='config/.env', env_file_encoding='utf-8')

    APP_ENV: str = 'dev'
    APP_PORT: int = 8000
    LOG_LEVEL: str = 'info'
    ALLOW_ORIGINS: str = 'http://localhost:5173'
    DATABASE_URL: str = 'postgresql://user:password@localhost:5432/appdb'

    def allow_origins_list(self) -> List[str]:
        return [o.strip() for o in self.ALLOW_ORIGINS.split(',') if o.strip()]

settings = Settings()
