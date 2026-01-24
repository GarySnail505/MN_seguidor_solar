import numpy as np

def vector_incidencia_desde_az_el(azimut_rad: float, elevacion_rad: float) -> np.ndarray:
    """
    Vector unitario HACIA el sol (panel -> sol) en ENU.

    ENU:
      x = Este, y = Norte, z = Arriba

    Este vector se usa para comparar con la normal del panel n (que por
    convención apunta hacia arriba, es decir, hacia el cielo).

    Para una elevación positiva, el componente z será positivo.

    Retorna las componentes x, y, z para el vector solar.
    """
    cos_el = np.cos(elevacion_rad)
    sen_el = np.sin(elevacion_rad)
    sen_az = np.sin(azimut_rad)
    cos_az = np.cos(azimut_rad)

    u = np.array([cos_el * sen_az, cos_el * cos_az, sen_el], dtype=float) 
    return u

def rotacion_x(beta: float) -> np.ndarray:
    """Rotación alrededor del eje x (Este)."""
    cb, sb = np.cos(beta), np.sin(beta)
    return np.array([[1, 0, 0],
                     [0, cb, -sb],
                     [0, sb, cb]], dtype=float)

def rotacion_y(phi: float) -> np.ndarray:
    """Rotación alrededor del eje y (Norte)."""
    cp, sp = np.cos(phi), np.sin(phi)
    return np.array([[cp, 0, sp],
                     [0,  1, 0],
                     [-sp, 0, cp]], dtype=float)

def normal_panel(phi: float, beta: float) -> np.ndarray:
    """
    Normal del panel después de aplicar:
      1) pitch (beta) en eje Este (x)
      2) roll  (phi)  en eje Norte (y)

    n(phi,beta) = Ry(phi) * Rx(beta) * n0
    con n0 = [0, 0, 1] (panel horizontal)
    """
    n0 = np.array([0.0, 0.0, 1.0], dtype=float)
    R = rotacion_y(phi) @ rotacion_x(beta)
    return R @ n0

def angulo_incidencia_grados(n: np.ndarray, s: np.ndarray) -> float:
    """
    Ángulo entre la normal del panel (n) y la incidencia solar (s).
    Objetivo: n paralelo a s => n·s = 1 => ángulo = 0°.
    """
    n = n / np.linalg.norm(n)
    s = s / np.linalg.norm(s)
    c = np.clip(np.dot(n, s), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def solucion_cerrada_semilla(s: np.ndarray) -> tuple[float, float]:
    """
    Solución analítica útil como semilla/verificación:

      beta = -asin(s_y)
      phi  = atan2(s_x, s_z)

    Esta semilla suele hacer que Newton converja en muy pocas iteraciones.
    """
    sx, sy, sz = float(s[0]), float(s[1]), float(s[2])
    beta0 = -np.arcsin(np.clip(sy, -1.0, 1.0))
    phi0 = np.arctan2(sx, sz)
    # beta0 = np.arctan2(-sy, sz) 
    # phi0 = np.arctan2(sx, np.sqrt(sz * sz + sy * sy))
    return phi0, beta0
