from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Base de Datos
    DATABASE_URL: str = "sqlite:///esports_bot.db"

    # Ingesta y API Keys
    BETSAPI_KEY: str | None = None
    PANDASCORE_KEY: str | None = None
    SPORTRADAR_KEY: str | None = None

    # Parámetros del Modelo y Blending Bayesiano
    RECENT_MATCHES_COUNT: int = 15
    TIME_DECAY_XI: float = 0.005             # Decaimiento temporal continuo e^(-xi * t)
    MIN_H2H_SAMPLE_SIZE: int = 5             # Umbral mínimo de enfrentamientos H2H
    SHRINKAGE_INTENSITY_M: float = 5.0       # Pseudo-cuentas m en el shrinkage Bayesiano
    BASE_ELO_RATING: float = 1500.0
    ELO_RD_INIT: float = 350.0
    ELO_K_FACTOR: float = 32.0
    ELO_GOAL_SCALE: float = 0.45             # Factor de conversión ELO diff a goles esperados

    # Ventaja de Saque / Local
    FORCE_GAMMA_ZERO: bool = False           # Si True, fuerza gamma = 0 (sin ventaja de saque)
    AUTO_TEST_GAMMA: bool = True             # Si True, corre Likelihood Ratio Test para validar gamma

    # Backtesting Walk-Forward
    BACKTEST_WINDOW_TYPE: str = "rolling"    # "rolling" o "expanding"
    BACKTEST_WINDOW_SIZE: int = 200          # Partidos en la ventana rodante
    BACKTEST_RETRAIN_BATCH_SIZE: int = 10    # Reentrenar Dixon-Coles cada B partidos (ELO es continuo)
    MIN_EDGE_THRESHOLD: float = 0.03         # Edge mínimo (+3% EV)
    KELLY_FRACTION: float = 0.25             # Cuarto de Kelly
    MAX_STAKE_PERCENT: float = 0.05          # Límite 5% de banca por apuesta
    DEFAULT_BANKROLL: float = 1000.0

    # Rango de Split Out-of-Sample (OOS Holdout)
    OOS_SPLIT_DATE: str = "2026-07-26T00:00:00Z"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
