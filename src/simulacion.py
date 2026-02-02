import os
import time
from datetime import time as time_of_day
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

VENTANA_DIURNA_INICIO = time_of_day(6, 0)
VENTANA_DIURNA_FIN = time_of_day(18, 0)

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


def _clasificar_estabilidad(ok_series: pd.Series) -> str:
    tasa = float(ok_series.mean()) if len(ok_series) else 0.0
    if tasa >= 0.9:
        return "Alta"
    if tasa >= 0.6:
        return "Media"
    return "Baja"


def calcular_estadisticas(df: pd.DataFrame, tiempos_metodos: dict) -> dict:
    df_diurno = df[df["es_diurno"]] if "es_diurno" in df else df
    if df_diurno.empty:
        return {}

    stats = {
        "total_pasos": int(len(df_diurno)),
        "newton": {
            "complejidad": "O(k·D^3) total (D=2)",
            "iteraciones_promedio": float(df_diurno["iter_newton"].mean()),
            "tiempo_total_s": float(tiempos_metodos.get("newton", 0.0)),
            "precision_final_deg": float(df_diurno["error_newton_deg"].iloc[-1]),
            "estabilidad": _clasificar_estabilidad(df_diurno["ok_newton"]),
        },
        "gradiente": {
            "complejidad": "O(k·D) total (D=2)",
            "iteraciones_promedio": float(df_diurno["iter_grad"].mean()),
            "tiempo_total_s": float(tiempos_metodos.get("gradiente", 0.0)),
            "precision_final_deg": float(df_diurno["error_grad_deg"].iloc[-1]),
            "estabilidad": _clasificar_estabilidad(df_diurno["ok_grad"]),
        },
    }
    return stats


def simular_dataframe(
    inicio_iso: str,
    paso_seg: int,
    lat: float,
    lon: float,
    alt_m: float,
    zona_horaria: str,
    backend: str,
    horas: Optional[float] = None,
    fin_iso: Optional[str] = None,
    return_stats: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    tiempos = _generar_tiempos(
        inicio_iso=inicio_iso,
        paso_seg=paso_seg,
        zona_horaria=zona_horaria,
        horas=horas,
        fin_iso=fin_iso
    )

    filas = []
    phi_prev, beta_prev = None, None
    tiempo_newton = 0.0
    tiempo_gradiente = 0.0

    for t in tiempos:
        es_diurno = VENTANA_DIURNA_INICIO <= t.time() <= VENTANA_DIURNA_FIN
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

        # Semilla: si hay paso anterior, usarlo. En horario nocturno fijamos la posición.
        if phi_prev is None:
            if es_diurno:
                phi0, beta0 = solucion_cerrada_semilla(s)
            else:
                phi0, beta0 = 0.0, 0.0
        else:
            phi0, beta0 = phi_prev, beta_prev

        if not es_diurno:
            phi_n = phi0
            beta_n = beta0
            err_n = np.nan
            info_n = {
                "iteraciones": 0,
                "convergio": False,
                "motivo": "Horario nocturno (panel fijo)",
            }
            phi_g = phi0
            beta_g = beta0
            err_g = np.nan
            info_g = {
                "iteraciones": 0,
                "convergio": False,
                "motivo": "Horario nocturno (panel fijo)",
                "alpha": np.nan,
            }
        else:
            if el_deg < 1.0:
                phi0, beta0 = solucion_cerrada_semilla(s)

            # Newton (sistema)
            inicio_newton = time.perf_counter()
            phi_n, beta_n, info_n = resolver_newton_sistema(s, phi0, beta0)
            tiempo_newton += time.perf_counter() - inicio_newton
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
            inicio_grad = time.perf_counter()
            phi_g, beta_g, info_g = resolver_gradiente_paso_fijo(
                s,
                phi0,
                beta0,
                alpha=0.45,
                kmax=25,
                semilla_fija=True,
            )
            tiempo_gradiente += time.perf_counter() - inicio_grad
            n_g = normal_panel(phi_g, beta_g)
            err_g = angulo_incidencia_grados(n_g, s)

        # Semilla para siguiente paso (solo Newton en horario diurno)
        if es_diurno and info_n["convergio"]:
            phi_prev, beta_prev = phi_n, beta_n

        filas.append({
            "tiempo": t.isoformat(),
            "es_diurno": es_diurno,
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

    df = pd.DataFrame(filas)
    if return_stats:
        stats = calcular_estadisticas(
            df,
            {"newton": tiempo_newton, "gradiente": tiempo_gradiente},
        )
        return df, stats

    return df


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

    df, stats = simular_dataframe(
        inicio_iso=inicio_iso,
        paso_seg=paso_seg,
        lat=lat,
        lon=lon,
        alt_m=alt_m,
        zona_horaria=zona_horaria,
        backend=backend,
        horas=horas,
        return_stats=True,
    )

    ruta_csv = os.path.join(carpeta_salida, "simulacion.csv")
    df.to_csv(ruta_csv, index=False)
    ruta_stats = os.path.join(carpeta_salida, "estadisticas.txt")
    with open(ruta_stats, "w", encoding="utf-8") as f:
        f.write("Resumen de métodos numéricos (solo horas diurnas 06:00 - 18:00)\n")
        if not stats:
            f.write("No hay pasos diurnos en el rango seleccionado.\n")
            return ruta_csv
        f.write(f"Total de pasos diurnos: {stats['total_pasos']}\n\n")
        f.write("Newton-Raphson\n")
        f.write(f"  Complejidad: {stats['newton']['complejidad']}\n")
        f.write(f"  Iteraciones promedio: {stats['newton']['iteraciones_promedio']:.2f}\n")
        f.write(f"  Tiempo total (s): {stats['newton']['tiempo_total_s']:.4f}\n")
        f.write(f"  Precisión final (deg): {stats['newton']['precision_final_deg']:.4f}\n")
        f.write(f"  Estabilidad: {stats['newton']['estabilidad']}\n\n")
        f.write("Gradiente (paso fijo)\n")
        f.write(f"  Complejidad: {stats['gradiente']['complejidad']}\n")
        f.write(f"  Iteraciones promedio: {stats['gradiente']['iteraciones_promedio']:.2f}\n")
        f.write(f"  Tiempo total (s): {stats['gradiente']['tiempo_total_s']:.4f}\n")
        f.write(f"  Precisión final (deg): {stats['gradiente']['precision_final_deg']:.4f}\n")
        f.write(f"  Estabilidad: {stats['gradiente']['estabilidad']}\n")
    return ruta_csv
