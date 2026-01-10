# Seguidor Solar 2GDL (Roll/Pitch) con Métodos Numéricos + Animación 3D

Este proyecto calcula la orientación de un panel solar con 2 grados de libertad:
- roll (φ): giro alrededor del eje Norte (eje y en ENU)
- pitch (β): giro alrededor del eje Este (eje x en ENU)

Objetivo: alinear la normal del panel con la dirección de incidencia solar para maximizar la captación
(ángulo de incidencia cercano a 0°).

Métodos numéricos:
1) Newton–Raphson (sistema no lineal 2×2): método principal (preciso cuando converge).
2) Levenberg–Marquardt (mínimos cuadrados no lineales): método alternativo (más robusto).

Salidas:
- CSV con resultados (ambos métodos)
- Gráficas de diagnóstico
- Animación 3D comparando ambos métodos (GIF y MP4)

Convención:
- Azimut: desde Norte hacia Este (N=0°, E=90°, S=180°, W=270°)
- Elevación: 0° en horizonte, positiva hacia arriba

Instalación:
  pip install -r requisitos.txt

Ejecución (Quito por defecto):
  python main.py --inicio 2026-01-09T06:00 --horas 12 --paso 60 --backend pvlib --gif --mp4

Pruebas:
  pytest -q

Nota FFmpeg:
Para exportar MP4, Matplotlib requiere ffmpeg instalado y en PATH. Si no está, el GIF sí se genera.
