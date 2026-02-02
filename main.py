import argparse
from src.simulacion import simular_y_guardar
from src.graficas import generar_graficas
from src.animacion_3d import crear_animacion_3d
from src.gui import launch_gui

def main():
    parser = argparse.ArgumentParser(
        description="Seguidor solar 2GDL con dos métodos numéricos (Newton + Gradiente) + animación 3D."
    )

    parser.add_argument("--inicio", type=str, default="2026-01-09T06:00",
                        help="Fecha/hora de inicio ISO (ej: 2026-01-09T06:00)")
    parser.add_argument("--horas", type=float, default=12.0, help="Duración en horas")
    parser.add_argument("--paso", type=int, default=60, help="Paso de simulación en segundos")

    parser.add_argument("--lat", type=float, default=-0.1807, help="Latitud (Quito por defecto)")
    parser.add_argument("--lon", type=float, default=-78.4678, help="Longitud (Quito por defecto)")
    parser.add_argument("--alt", type=float, default=2850.0, help="Altitud en metros (Quito por defecto)")
    parser.add_argument("--zona_horaria", type=str, default="America/Guayaquil",
                        help="Zona horaria IANA (Ecuador: America/Guayaquil)")

    parser.add_argument("--backend", type=str, choices=["pysolar"], default="pysolar",
                        help="Librería para posición solar (se usa pysolar)")
    parser.add_argument("--salida", type=str, default="salidas", help="Carpeta de salida")

    parser.add_argument("--gif", action="store_true", help="Generar animación GIF 3D")
    parser.add_argument("--mp4", action="store_true", help="Generar animación MP4 3D (requiere ffmpeg)")
    parser.add_argument("--fps", type=int, default=25, help="FPS del video/gif")

    parser.add_argument("--gui", action="store_true", help="Abrir interfaz gráfica (GUI)")

    args = parser.parse_args()

    if args.gui:
        launch_gui()
        return

    ruta_csv, ruta_metricas = simular_y_guardar(
        inicio_iso=args.inicio,
        horas=args.horas,
        paso_seg=args.paso,
        lat=args.lat,
        lon=args.lon,
        alt_m=args.alt,
        zona_horaria=args.zona_horaria,
        backend=args.backend,
        carpeta_salida=args.salida
    )

    generar_graficas(ruta_csv, args.salida)

    if args.gif or args.mp4:
        crear_animacion_3d(
            ruta_csv=ruta_csv,
            carpeta_salida=args.salida,
            fps=args.fps,
            guardar_gif=args.gif,
            guardar_mp4=args.mp4
        )

    print("\nListo. Revise la carpeta:", args.salida)
    print("CSV:", ruta_csv)
    print("Métricas:", ruta_metricas)

if __name__ == "__main__":
    main()
