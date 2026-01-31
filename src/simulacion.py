import os
from typing import Optional

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
from src.metodos.gradiente_paso_fijo import resolver_gradiente_paso_fijo

def _normalizar_fecha(fecha_iso: str, zona_horaria: str) -> pd.Timestamp:
    t = pd.Timestamp(fecha_iso)
    if t.tzinfo is None:
        return t.tz_localize(zona_horaria)
    return t.tz_convert(zona_horaria)


def _generar_tiempos(
    inicio_iso: str,
    paso_seg: int,
    zona_horaria: str,
    horas: Optional[float] = None,
    fin_iso: Optional[str] = None
) -> list[pd.Timestamp]:
    t0 = _normalizar_fecha(inicio_iso, zona_horaria)

    if fin_iso is not None:
        t1 = _normalizar_fecha(fin_iso, zona_horaria)
        if t1 < t0:
            raise ValueError("La fecha final no puede ser anterior a la inicial.")
        segundos = (t1 - t0).total_seconds()
    else:
        if horas is None:
            raise ValueError("Debe especificar 'horas' o 'fin_iso'.")
        segundos = horas * 3600

    n_pasos = int(segundos // paso_seg) + 1
    return [t0 + pd.Timedelta(seconds=i * paso_seg) for i in range(n_pasos)]


def simular_dataframe(
    inicio_iso: str,
    paso_seg: int,
    lat: float,
    lon: float,
    alt_m: float,
    zona_horaria: str,
    backend: str,
    horas: Optional[float] = None,
    fin_iso: Optional[str] = None
) -> pd.DataFrame:
    tiempos = _generar_tiempos(
        inicio_iso=inicio_iso,
        paso_seg=paso_seg,
        zona_horaria=zona_horaria,
        horas=horas,
        fin_iso=fin_iso
    )

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
        if el_deg < 1.0:
            phi0, beta0 = solucion_cerrada_semilla(s)
        elif phi_prev is None:
            phi0, beta0 = solucion_cerrada_semilla(s)
        else:
            phi0, beta0 = phi_prev, beta_prev
        # if phi_prev is None:
        #     phi0, beta0 = solucion_cerrada_semilla(s)
        # else:
        #     phi0, beta0 = phi_prev, beta_prev

        # Newton (sistema)
        phi_n, beta_n, info_n = resolver_newton_sistema(s, phi0, beta0)
        n_n = normal_panel(phi_n, beta_n)

        # Corrección física: la normal del panel debe apuntar hacia el sol.
        # (Evita soluciones "invertidas" que son válidas en ecuaciones parciales,
        # pero no tienen sentido para captación de energía.)
        if float(np.dot(n_n, s)) < 0.0:
            phi_n += np.pi
            n_n = normal_panel(phi_n, beta_n)

        err_n = angulo_incidencia_grados(n_n, s)

        # Gradiente con paso fijo (método contrastante: puede fallar)
        # Nota: por defecto usa semilla fija (0,0) y kmax moderado para que
        # en algunas horas NO alcance el mínimo. Esto permite contrastar con Newton.
        phi_g, beta_g, info_g = resolver_gradiente_paso_fijo(
            s,
            phi0,
            beta0,
            alpha=0.45,
            kmax=25,
            semilla_fija=True,
        )
        n_g = normal_panel(phi_g, beta_g)
        err_g = angulo_incidencia_grados(n_g, s)

        # Semilla para siguiente paso (solo Newton)
        if info_n["convergio"]:
            phi_prev, beta_prev = phi_n, beta_n
        else:
            phi_prev, beta_prev = phi0, beta0

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

            "phi_grad_rad": phi_g,
            "beta_grad_rad": beta_g,
            "error_grad_deg": err_g,
            "iter_grad": info_g["iteraciones"],
            "ok_grad": info_g["convergio"],
            "motivo_grad": info_g["motivo"],
            "alpha_grad": info_g.get("alpha", None),
        })

    return pd.DataFrame(filas)


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

    df = simular_dataframe(
        inicio_iso=inicio_iso,
        paso_seg=paso_seg,
        lat=lat,
        lon=lon,
        alt_m=alt_m,
        zona_horaria=zona_horaria,
        backend=backend,
        horas=horas
    )

    ruta_csv = os.path.join(carpeta_salida, "simulacion.csv")
    df.to_csv(ruta_csv, index=False)
    return ruta_csv