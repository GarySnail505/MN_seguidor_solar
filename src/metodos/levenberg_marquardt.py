import numpy as np
from src.cinematica import normal_panel

def resolver_levenberg_marquardt(
    s: np.ndarray,
    phi0: float,
    beta0: float,
    eps_r: float = 1e-12,
    eps_delta: float = 1e-12,
    kmax: int = 60,
    lambda0: float = 1e-3
) -> tuple[float, float, dict]:
    """
    Levenberg–Marquardt para minimizar:
      1/2 ||r||^2, r = n(phi,beta) - s

    Paso:
      (J^T J + lambda I) Delta = -J^T r
    """
    phi, beta = float(phi0), float(beta0)
    lam = float(lambda0)

    info = {
        "metodo": "levenberg_marquardt",
        "iteraciones": 0,
        "convergio": False,
        "motivo": "",
        "lambda_final": None
    }

    def jacobiano_n(phi_, beta_):
        cp, sp = np.cos(phi_), np.sin(phi_)
        cb, sb = np.cos(beta_), np.sin(beta_)

        dndphi = np.array([cp * cb, 0.0, -sp * cb], dtype=float)
        dndbet = np.array([-sp * sb, -cb, -cp * sb], dtype=float)
        return np.column_stack([dndphi, dndbet])  # 3x2

    def costo(phi_, beta_):
        n = normal_panel(phi_, beta_)
        r = n - s
        return 0.5 * float(r @ r), r

    C, r = costo(phi, beta)

    for k in range(kmax):
        info["iteraciones"] = k + 1

        if np.linalg.norm(r, 2) < eps_r:
            info["convergio"] = True
            info["motivo"] = "norma(r) < eps_r"
            info["lambda_final"] = lam
            return phi, beta, info

        J = jacobiano_n(phi, beta)
        A = (J.T @ J) + lam * np.eye(2)
        g = J.T @ r

        try:
            Delta = np.linalg.solve(A, -g)
        except np.linalg.LinAlgError:
            info["motivo"] = "No se pudo resolver (A singular)"
            info["lambda_final"] = lam
            return phi, beta, info

        if np.linalg.norm(Delta, 2) < eps_delta:
            info["convergio"] = True
            info["motivo"] = "norma(Delta) < eps_delta"
            info["lambda_final"] = lam
            return phi, beta, info

        phi_n = phi + float(Delta[0])
        beta_n = beta + float(Delta[1])
        Cn, rn = costo(phi_n, beta_n)

        if Cn < C:
            phi, beta, C, r = phi_n, beta_n, Cn, rn
            lam = max(lam / 2.0, 1e-12)
        else:
            lam = min(lam * 2.0, 1e12)

    info["motivo"] = "kmax alcanzado"
    info["lambda_final"] = lam
    return phi, beta, info
