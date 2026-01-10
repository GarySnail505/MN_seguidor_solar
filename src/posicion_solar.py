from __future__ import annotations

def obtener_azimut_elevacion(
    fecha_hora,
    lat: float,
    lon: float,
    alt_m: float = 0.0,
    zona_horaria: str = "America/Guayaquil",
    backend: str = "pvlib"
) -> tuple[float, float]:
    """
    Retorna (azimut_deg, elevacion_deg).

    pvlib usa la convención de azimut como grados al Este desde el Norte
    (N=0, E=90, S=180, W=270). :contentReference[oaicite:0]{index=0}
    """
    backend = backend.lower()

    if backend == "pvlib":
        import pandas as pd
        import pvlib

        if isinstance(fecha_hora, str):
            t = pd.Timestamp(fecha_hora).tz_localize(zona_horaria)
        else:
            t = pd.Timestamp(fecha_hora)
            if t.tzinfo is None:
                t = t.tz_localize(zona_horaria)
            else:
                t = t.tz_convert(zona_horaria)

        loc = pvlib.location.Location(latitude=lat, longitude=lon, altitude=alt_m, tz=zona_horaria)
        sp = loc.get_solarposition(times=pd.DatetimeIndex([t]))

        az = float(sp["azimuth"].iloc[0])
        zen = float(sp["apparent_zenith"].iloc[0])
        elev = 90.0 - zen
        return az, elev

    if backend == "pysolar":
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from pysolar.solar import get_altitude, get_azimuth

        if isinstance(fecha_hora, str):
            dt = datetime.fromisoformat(fecha_hora)
        else:
            dt = fecha_hora

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(zona_horaria))

        elev = float(get_altitude(lat, lon, dt))
        az = float(get_azimuth(lat, lon, dt)) % 360.0
        return az, elev

    raise ValueError("backend debe ser 'pvlib' o 'pysolar'")
