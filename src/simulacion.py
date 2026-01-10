import os
import numpy as np
import pandas as pd

from src.posicion_solar import obtener_azimut_elevacion
from src.cinematica import (
    vector_incidencia_desde_az_el,
    normal_panel,
    angulo_incidencia_grados,
    solucion_cerrada_semilla
)
from src.metodos.newton_sistema import resolver_newton_sistema
from src.metodos.levenberg_marquardt import resolver_levenberg_marquardt

def simular_y_guardar(
    inicio_iso: str,
    horas: float,
    paso_seg: int,
    lat: float,
    lon: float,
    alt_m: float,
    zona_horaria: str,
    backend: str,
    carpeta_salida: str
) -> str:
    os.makedirs(carpeta_salida, exist_ok=True)

    t0 = pd.Timestamp(inicio_iso).tz_localize(zona_horaria)
    n_pasos = int((horas * 3600) // paso_seg) + 1
    tiempos = [t0 + pd.Timedelta(seconds=i * paso_seg) for i in range(n_pasos)]

    filas = []
    phi_prev, beta_prev = None, None

    for t in tiempos:
        az_deg, el_deg = obtener_azimut_elevacion(
            fecha_hora=t,
            lat=lat,
            lon=lon,
            alt_m=alt_m,
            zona_horaria=zona_horaria,
            backend=backend
        )

        az = np.radians(az_deg)
        el = np.radians(el_deg)

        s = vector_incidencia_desde_az_el(az, el)

        # Semilla: si hay paso anterior, usarlo. Si no, solución cerrada.
        if phi_prev is None:
            phi0, beta0 = solucion_cerrada_semilla(s)
        else:
            phi0, beta0 = phi_prev, beta_prev

        # Newton (sistema)
        phi_n, beta_n, info_n = resolver_newton_sistema(s, phi0, beta0)
        n_n = normal_panel(phi_n, beta_n)
        err_n = angulo_incidencia_grados(n_n, s)

        # LM (mínimos cuadrados)
        phi_lm, beta_lm, info_lm = resolver_levenberg_marquardt(s, phi0, beta0)
        n_lm = normal_panel(phi_lm, beta_lm)
        err_lm = angulo_incidencia_grados(n_lm, s)

        # Semilla para siguiente paso
        if info_n["convergio"]:
            phi_prev, beta_prev = phi_n, beta_n
        else:
            phi_prev, beta_prev = phi_lm, beta_lm

        filas.append({
            "tiempo": t.isoformat(),
            "azimut_deg": az_deg,
            "elevacion_deg": el_deg,

            "phi_newton_rad": phi_n,
            "beta_newton_rad": beta_n,
            "error_newton_deg": err_n,
            "iter_newton": info_n["iteraciones"],
            "ok_newton": info_n["convergio"],
            "motivo_newton": info_n["motivo"],

            "phi_lm_rad": phi_lm,
            "beta_lm_rad": beta_lm,
            "error_lm_deg": err_lm,
            "iter_lm": info_lm["iteraciones"],
            "ok_lm": info_lm["convergio"],
            "motivo_lm": info_lm["motivo"],
            "lambda_lm": info_lm.get("lambda_final", None),
        })

    df = pd.DataFrame(filas)
    ruta_csv = os.path.join(carpeta_salida, "simulacion.csv")
    df.to_csv(ruta_csv, index=False)
    return ruta_csv
