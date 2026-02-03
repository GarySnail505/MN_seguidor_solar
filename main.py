import argparse
from datetime import datetime
from src.simulacion import simular_y_guardar
from src.graficas import generar_graficas
from src.animacion_3d import crear_animacion_3d
from src.gui import launch_gui

def main():
    # Calcula la fecha de "hoy" cada vez que se ejecuta el script
    hoy_default = datetime.now().strftime("%Y-%m-%d") + "T06:00"
    
    parser = argparse.ArgumentParser(
        description=(
            "Seguidor solar 2GDL con dos métodos numéricos (Newton + Gradiente) "
            "+ animación 3D. El panel se mantiene fijo fuera de 06:00–18:00."
        )
    )

    # Si el usuario no pone --inicio, se usa hoy_default
    parser.add_argument("--inicio", type=str, default=hoy_default,
                        help=f"Fecha/hora inicio ISO. Por defecto: HOY ({hoy_default})")
    
    parser.add_argument(
        "--horas",
        type=float,
        default=12.0,
        help="Duración en horas (el panel solo se mueve entre 06:00–18:00).",
    )
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
    # ELIMINADO: Argumento --mp4 para que no aparezca en la ayuda ni se pueda usar
    
    parser.add_argument("--fps", type=int, default=25, help="FPS del video/gif")

    parser.add_argument("--gui", action="store_true", help="Abrir interfaz gráfica (GUI)")

    args = parser.parse_args()

    if args.gui:
        launch_gui()
        return

    # Ejecución de la simulación
    ruta_csv = simular_y_guardar(
        inicio_iso=args.inicio, # Si no se especificó, usa hoy_default
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

    # Solo generamos GIF si se pide. MP4 está desactivado forzosamente.
    if args.gif:
        crear_animacion_3d(
            ruta_csv=ruta_csv,
            carpeta_salida=args.salida,
            fps=args.fps,
            guardar_gif=args.gif,
            guardar_mp4=False  # <--- CORRECCIÓN: Forzado a False
        )

    print("\nListo. Revise la carpeta:", args.salida)
    print("CSV:", ruta_csv)
    print("Resumen:", f"{args.salida}/estadisticas.txt")

if __name__ == "__main__":
    main()