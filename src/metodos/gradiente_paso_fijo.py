"""Gradiente descendente con paso fijo (método intencionalmente frágil).

Minimizamos una función de costo:

    J(phi, beta) = 1 - (n(phi,beta) · u)

- u: vector unitario hacia el sol (panel -> sol)
- n: normal unitario del panel

Objetivo: J -> 0  (equivalente a incidencia ~ 0°)

Por qué puede fallar
--------------------
Este gradiente es deliberadamente "no robusto":
- Usa paso fijo alpha (sin búsqueda de línea).
- Usa gradiente numérico (diferencias finitas).
- Con alpha y kmax modestos, a veces oscila o no alcanza el mínimo.

"""

from __future__ import annotations

import numpy as np

from src.cinematica import normal_panel


def _envolver_angulo(a: float) -> float:
    """Envuelve un ángulo a [-pi, pi] para mantenerlo acotado."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def costo_alineacion(phi: float, beta: float, u: np.ndarray) -> float:
    """Costo J = 1 - n·u (mínimo en 0)."""
    n = normal_panel(phi, beta)
    # n y u son unitarios (por construcción), pero se protege el producto.
    c = float(np.clip(np.dot(n, u), -1.0, 1.0))
    return 1.0 - c


def gradiente_numerico(phi: float, beta: float, u: np.ndarray, h: float = 1e-4) -> tuple[float, float]:
    """Gradiente numérico (central) de J respecto a (phi, beta)."""
    j_ph = costo_alineacion(phi + h, beta, u)
    j_mh = costo_alineacion(phi - h, beta, u)
    dphi = (j_ph - j_mh) / (2.0 * h)

    j_ph = costo_alineacion(phi, beta + h, u)
    j_mh = costo_alineacion(phi, beta - h, u)
    dbeta = (j_ph - j_mh) / (2.0 * h)

    return float(dphi), float(dbeta)


def resolver_gradiente_paso_fijo(
    u: np.ndarray,
    phi0: float,
    beta0: float,
    *,
    alpha: float = 0.45,
    h: float = 1e-4,
    tol: float = 1e-8,
    kmax: int = 25,
    limitar_angulos: bool = True,
    semilla_fija: bool = True,

) -> tuple[float, float, dict]:
    
    """Resuelve por gradiente descendente con paso fijo.

    Parámetros clave (para contraste):
    - alpha fijo y kmax moderado => puede NO converger.
    - semilla_fija=True fuerza (phi0,beta0)=(0,0) para aumentar fallas.

    Retorna:
      phi, beta, info
    """

    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)

    if semilla_fija:
        phi, beta = np.pi, 0.0
        motivo = "Semilla fija (0,0) para contraste" 
    else:
        phi, beta = float(phi0), float(beta0)
        motivo = "Semilla heredada" 

    # límites físicos simples (panel tipo gimbal)
    beta_lo, beta_hi = -np.pi / 2, np.pi / 2

    j_prev = costo_alineacion(phi, beta, u)

    convergio = False
    for k in range(1, int(kmax) + 1):
        dphi, dbeta = gradiente_numerico(phi, beta, u, h=h)

        # descenso (paso fijo)
        phi = phi - alpha * dphi
        beta = beta - alpha * dbeta

        # mantener estable numéricamente
        phi = _envolver_angulo(phi)
        if limitar_angulos:
            beta = _clamp(beta, beta_lo, beta_hi)

        j = costo_alineacion(phi, beta, u)

        if j < tol:
            convergio = True
            j_prev = j
            break

        # si empieza a aumentar mucho, suele ser signo de "rebotar"
        j_prev = j

    info = {
        "iteraciones": int(k if 'k' in locals() else 0),
        "convergio": bool(convergio),
        "motivo": "OK" if convergio else f"No converge con paso fijo (alpha={alpha}, kmax={kmax}). {motivo}",
        "alpha": float(alpha),
        "kmax": int(kmax),
        "semilla_fija": bool(semilla_fija),
        "costo_final": float(j_prev),
    }

    return float(phi), float(beta), info
