from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from pysolar.solar import get_altitude, get_azimuth


def obtener_azimut_elevacion(
    fecha_hora,
    lat: float,
    lon: float,
    alt_m: float = 0.0,
    zona_horaria: str = "America/Guayaquil",
    backend: str = "pysolar",
) -> tuple[float, float]:
    """Calcula la posición solar usando únicamente ``pysolar``.

    Parámetros ``alt_m`` y ``backend`` se mantienen solo para compatibilidad con
    el resto del proyecto (no se usan).

    Retorna:
        (azimut_deg, elevacion_deg)

    Convención de azimut (pysolar): desde el Norte, positivo hacia el Este.
    """

    # Convertir entrada a datetime con zona horaria
    if isinstance(fecha_hora, str):
        dt = datetime.fromisoformat(fecha_hora)
    else:
        dt = fecha_hora

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(zona_horaria))
    else:
        dt = dt.astimezone(ZoneInfo(zona_horaria))

    elevacion_deg = float(get_altitude(lat, lon, dt))
    azimut_deg = float(get_azimuth(lat, lon, dt)) #% 360.0
    return azimut_deg, elevacion_deg