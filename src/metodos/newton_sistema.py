import numpy as np

def resolver_newton_sistema(
    s: np.ndarray,
    phi0: float,
    beta0: float,
    eps_f: float = 1e-12,
    eps_delta: float = 1e-12,
    kmax: int = 30,
    tau: float = 1e-10
) -> tuple[float, float, dict]:
    """
    Newton–Raphson para sistema 2x2:

      F1(phi,beta) = sin(phi)*cos(beta) - s_x
      F2(phi,beta) = -sin(beta) - s_y

    Jacobiano:
      J = [[cos(phi)*cos(beta), -sin(phi)*sin(beta)],
           [0,                 -cos(beta)]]

    Se resuelve J*Delta = -F (no se invierte J).
    """
    sx, sy = float(s[0]), float(s[1])
    phi, beta = float(phi0), float(beta0)

    info = {"metodo": "newton_sistema", "iteraciones": 0, "convergio": False, "motivo": ""}

    for k in range(kmax):
        F1 = np.sin(phi) * np.cos(beta) - sx
        F2 = -np.sin(beta) - sy
        F = np.array([F1, F2], dtype=float)

        info["iteraciones"] = k + 1

        if np.linalg.norm(F, 2) < eps_f:
            info["convergio"] = True
            info["motivo"] = "norma(F) < eps_f"
            return phi, beta, info

        cb = np.cos(beta)
        if abs(cb) < tau:
            info["motivo"] = "Jacobiano mal condicionado: |cos(beta)| < tau"
            return phi, beta, info

        J = np.array([
            [np.cos(phi) * cb, -np.sin(phi) * np.sin(beta)],
            [0.0,              -cb]
        ], dtype=float)

        try:
            Delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            info["motivo"] = "No se pudo resolver el sistema lineal (J singular)"
            return phi, beta, info

        if np.linalg.norm(Delta, 2) < eps_delta:
            info["convergio"] = True
            info["motivo"] = "norma(Delta) < eps_delta"
            return phi, beta, info

        phi += float(Delta[0])
        beta += float(Delta[1])

    info["motivo"] = "kmax alcanzado"
    return phi, beta, info
