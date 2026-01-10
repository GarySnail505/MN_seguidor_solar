# Guía de funcionamiento

## 1) ¿Qué hace el programa?
Para cada instante:
1. Calcula azimut y elevación del Sol (pvlib o pysolar).
2. Construye el vector de incidencia s (Sol → panel).
3. Calcula (roll, pitch) por dos métodos:
   - Newton (sistema no lineal)
   - Levenberg–Marquardt (mínimos cuadrados)
4. Calcula error de incidencia en grados:
   error = arccos(n · s)
   Objetivo: error ≈ 0°
5. Guarda resultados en CSV.
6. Genera gráficas y animación 3D (si se solicita).

## 2) Gráficas
- angulos.png: roll y pitch para ambos métodos
- error_incidencia.png: si está cerca de 0°, el seguimiento está correcto

## 3) Animación 3D
Se dibuja:
- Panel (rectángulo rotado)
- Vector de luz incidente (s)
- Normal del panel (n)

Izquierda: Newton
Derecha: Levenberg–Marquardt
