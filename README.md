# ⚽ e-Soccer / FIFA Predictive Engine & Value Bet Analyzer

Sistema modular de análisis cuantitativo, modelado predictivo y backtesting para torneos de FIFA Esports / e-Soccer (**GT Leagues**, **Esports Battle**: formatos `4x2`, `5x2` y `6x2` minutos).

---

## 🔒 Protocolo de Validación Out-of-Sample (OOS Holdout)

Para garantizar un proceso cuantitativo riguroso y evitar el sobreajuste (*data snooping / lookahead bias*), el histórico total se divide estrictamente en dos tramos cronológicos:

1. **Tramo In-Sample (Entrenamiento, Calibración y Backtesting de Desarrollo)**:
   - **Rango**: Desde el inicio del histórico hasta `2026-07-25T23:59:59Z`.
   - **Uso**: Estimación de parámetros $\alpha_i, \beta_j$, sintonización del parámetro de shrinkage $m$, K-Factor del ELO, parámetro de decaimiento $\xi$ y umbrales de +EV.
2. **Tramo Out-of-Sample (OOS Ciego / Blind Test)**:
   - **Rango**: Desde `2026-07-26T00:00:00Z` en adelante.
   - **Uso EXCLUSIVO**: Validación final out-of-sample del modelo completo y dimensionamiento de stakes con Kelly fraccional. **Este tramo no se toca ni se inspecciona durante el desarrollo.**

---

## 🏛️ Arquitectura Modular

```
esoccer_predictor/
├── .env.example                      # Configuración y variables de entorno
├── requirements.txt                  # Dependencias del proyecto
├── main.py                           # CLI principal
├── src/
│   ├── config.py                     # Parámetros centralizados con pydantic-settings
│   ├── db/
│   │   ├── models.py                 # SQLAlchemy 2.0 (Player, Match, OddsSnapshot, RatingHistory)
│   │   ├── session.py                # Conexión SQLite y SessionLocal
│   │   └── queries.py                # get_h2h_matches, get_recent_form, etc.
│   ├── analytics/
│   │   ├── metrics.py                # Goles ponderados, % Over 2.5/3.5, % BTTS con decaimiento
│   │   ├── elo.py                    # Rating dinámico ELO/Glicko + Blending Bayesiano (Shrinkage)
│   │   ├── dixon_coles.py            # Motor Poisson Bivariado con corrección Dixon-Coles
│   │   └── odds.py                   # Eliminación de overround (Shin/Power/Proporcional) y +EV
│   ├── ingestion/
│   │   ├── base.py                   # Interfaz abstracta BaseIngestionAdapter
│   │   ├── validators.py             # Validación estricta de goles, cuotas y mercados
│   │   ├── betsapi_adapter.py        # Adaptador API con rate limiting
│   │   └── mock_generator.py         # Generador de datos sintéticos realistas
│   ├── backtesting/
│   │   ├── engine.py                 # Simulación Walk-Forward por lotes configurables
│   │   └── metrics.py                # Brier score, ROI, PnL, Max Drawdown y Sharpe Ratio
│   └── ui/
│       ├── cli.py                    # Interfaz CLI con Rich
│       └── app_streamlit.py          # Dashboard interactivo Streamlit
└── tests/
    ├── test_dixon_coles.py
    ├── test_elo.py
    ├── test_odds.py
    ├── test_backtesting.py
    └── test_pipeline_integration.py  # Test de integración End-to-End
```

---

## 📐 Decisiones de Diseño y Fórmulas Matemáticas

### 1. Shrinkage Bayesiano (Dixon-Coles + Prior ELO/Glicko)
Para dos gamers $i$ y $j$ en un formato de duración dado:
- Sean $\lambda_{ij}^{MLE}, \mu_{ij}^{MLE}$ las tasas esperadas estimadas por Dixon-Coles.
- Sean $\lambda_{ij}^{ELO}, \mu_{ij}^{ELO}$ las tasas esperadas deducidas del diferencial de rating ELO del formato.
- Sea $n_{ij}$ el número de partidos H2H previos finalizados en ese formato específico.
- Sea $m$ la constante de intensidad del prior (pseudo-cuentas, por defecto $m = 5.0$).

**Pesos:**
$$w_{ij}^{MLE} = \frac{n_{ij}}{n_{ij} + m}, \quad w_{ij}^{ELO} = \frac{m}{n_{ij} + m}$$

**Intensidades combinadas:**
$$\lambda_{ij}^{blended} = \frac{n_{ij} \cdot \lambda_{ij}^{MLE} + m \cdot \lambda_{ij}^{ELO}}{n_{ij} + m}$$
$$\mu_{ij}^{blended} = \frac{n_{ij} \cdot \mu_{ij}^{MLE} + m \cdot \mu_{ij}^{ELO}}{n_{ij} + m}$$

### 2. Walk-Forward Backtesting (Ventana y Reentrenamiento)
- **Ventana Rodante (Rolling Window)**: $W = 200$ partidos para capturar cambios de meta y parches de FIFA sin arrastrar datos obsoletos.
- **Frecuencia**: Reentrenamiento MLE de Dixon-Coles por lotes de $B = 10$ partidos; actualización de ratings ELO online continua partido a partido ($B = 1$).

### 3. Validación Estadística de $\gamma$ (Ventaja de Local/Saque)
Test de Razón de Verosimilitud (Likelihood Ratio Test):
$$\Lambda_{LR} = 2 \left( \ln \mathcal{L}_{\text{unrestricted}}(\alpha, \beta, \gamma, \rho) - \ln \mathcal{L}_{\text{restricted}}(\alpha, \beta, \gamma = 0, \rho) \right) \sim \chi^2(1)$$
Si $p\text{-value} > 0.05$, se fuerza $\gamma \equiv 0$ para evitar sobreajuste.
