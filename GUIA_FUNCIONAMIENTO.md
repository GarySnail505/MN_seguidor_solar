# Guía de funcionamiento

## 1) ¿Qué hace el programa?
Para cada instante de tiempo:
1. Calcula **azimut** y **elevación** del Sol usando **pysolar**.
2. Construye el vector solar unitario `u` (**panel → sol**) en coordenadas ENU.
3. Calcula los ángulos del panel `(φ, β)` con **dos métodos numéricos**:
   - **Newton–Raphson** (sistema no lineal 2×2)
   - **Gradiente con paso fijo** (método contrastante)
4. Evalúa la **incidencia** (en grados):

   \[
   \text{incidencia} = \arccos\big( n(\phi,\beta) \cdot u \big)
   \]

   Objetivo: **incidencia ≈ 0°**.
5. Guarda todo en `salidas/simulacion.csv`.
6. Si se pide, genera gráficas y animación 3D.

## 2) Gráficas
- `angulos.png`: roll y pitch para Newton y Gradiente.
- `error_incidencia.png`: diagnóstico; mientras más cerca de 0°, mejor seguimiento.

## 3) Animación 3D
Se dibuja:
- Panel (rectángulo rotado)
- Vector hacia el sol `u`
- Normal del panel `n`

Izquierda: **Newton**
Derecha: **Gradiente (paso fijo)**

> Nota: el gradiente se configura con paso fijo y pocas iteraciones para que, en algunos instantes, **no alcance** incidencia cero. Eso es intencional para comparar.
