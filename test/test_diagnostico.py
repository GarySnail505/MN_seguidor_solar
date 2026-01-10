import numpy as np
from src.cinematica import (
    vector_incidencia_desde_az_el,
    normal_panel,
    solucion_cerrada_semilla,
    angulo_incidencia_grados
)
from src.metodos.newton_sistema import resolver_newton_sistema
from src.metodos.levenberg_marquardt import resolver_levenberg_marquardt

def test_ambos_metodos_dejan_incidencia_casi_cero():
    rng = np.random.default_rng(123)

    for _ in range(200):
        az = np.deg2rad(rng.uniform(0, 360))
        el = np.deg2rad(rng.uniform(5, 85))  # evita extremos
        s = vector_incidencia_desde_az_el(az, el)

        phi0, beta0 = solucion_cerrada_semilla(s)

        phi_n, beta_n, info_n = resolver_newton_sistema(s, phi0, beta0)
        err_n = angulo_incidencia_grados(normal_panel(phi_n, beta_n), s)

        phi_lm, beta_lm, info_lm = resolver_levenberg_marquardt(s, phi0, beta0)
        err_lm = angulo_incidencia_grados(normal_panel(phi_lm, beta_lm), s)

        assert err_n < 1e-6 or info_n["convergio"]
        assert err_lm < 1e-6 or info_lm["convergio"]
