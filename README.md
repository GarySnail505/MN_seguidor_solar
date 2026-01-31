# Seguidor Solar 2GDL (roll/pitch) con 2 métodos numéricos + animación 3D

Este proyecto simula un **seguidor solar de 2 grados de libertad** (2GDL) que orienta un panel para que su **normal** quede alineada con la dirección del sol.

- **roll (φ)**: giro alrededor del eje **Norte** (eje *y* en ENU)
- **pitch (β)**: giro alrededor del eje **Este** (eje *x* en ENU)

> Objetivo: **incidencia ≈ 0°** (el ángulo entre la normal del panel y el vector hacia el sol).

## Métodos numéricos

1) **Newton–Raphson (sistema no lineal 2×2)**
   - Método principal (rápido y preciso cuando converge).

2) **Gradiente descendente con paso fijo (método contrastante)**
   - Método intencionalmente *más frágil*.
   - Usa **paso fijo** (sin búsqueda de línea) y normalmente **pocas iteraciones**, así que puede quedarse con error.
   - Se usa para **comparar** contra Newton.

## Requisitos

Instale dependencias:

```bash
pip install -r requisitos.txt
```

> Nota: para exportar MP4, Matplotlib requiere **FFmpeg** instalado y en el `PATH`.

## Ejecución

### Opción A: Modo consola (genera CSV + gráficas y opcionalmente animación)

```bash
python main.py --inicio 2026-01-09T06:00 --horas 12 --paso 60 --backend pysolar --gif --mp4
```

Salidas (por defecto en `salidas/`):
- `simulacion.csv`
- `angulos.png`
- `error_incidencia.png`
- `animacion_3d.gif` / `animacion_3d.mp4` (si se activan)

### Opción B: Interfaz gráfica (GUI)

```bash
python main.py --gui
```

## Pruebas

```bash
pytest -q
```

## ¿Qué significa “incidencia”?

La **incidencia** es el ángulo (en grados) entre:
- la **normal del panel** `n(φ,β)`
- el **vector hacia el sol** `u` (panel → sol)

- Si `incidencia = 0°` ⇒ el panel está **perfectamente apuntando al sol**.
- Si `incidencia` es grande ⇒ el panel está **mal orientado** (pierde captación).
