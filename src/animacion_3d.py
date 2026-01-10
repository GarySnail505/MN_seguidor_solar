import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.cinematica import (
    normal_panel,
    vector_incidencia_desde_az_el,
    rotacion_x,
    rotacion_y,
    angulo_incidencia_grados
)

def _panel_vertices(phi: float, beta: float, ancho=1.2, alto=0.7):
    hw, hh = ancho / 2, alto / 2
    verts_local = np.array([
        [-hw, -hh, 0],
        [ hw, -hh, 0],
        [ hw,  hh, 0],
        [-hw,  hh, 0],
    ], dtype=float)

    R = rotacion_y(phi) @ rotacion_x(beta)
    verts = (R @ verts_local.T).T
    return verts

def _configurar_ejes(ax, titulo: str):
    ax.set_title(titulo)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_xlabel("Este (x)")
    ax.set_ylabel("Norte (y)")
    ax.set_zlabel("Arriba (z)")

def crear_animacion_3d(
    ruta_csv: str,
    carpeta_salida: str,
    fps: int = 25,
    guardar_gif: bool = True,
    guardar_mp4: bool = True
):
    os.makedirs(carpeta_salida, exist_ok=True)
    df = pd.read_csv(ruta_csv)

    # Vector solar por frame
    az = np.radians(df["azimut_deg"].to_numpy())
    el = np.radians(df["elevacion_deg"].to_numpy())
    S = np.array([vector_incidencia_desde_az_el(az[i], el[i]) for i in range(len(df))])

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    _configurar_ejes(ax1, "Método 1: Newton (sistema)")
    _configurar_ejes(ax2, "Método 2: Levenberg–Marquardt")

    poly1 = Poly3DCollection([], alpha=0.6)
    poly2 = Poly3DCollection([], alpha=0.6)
    ax1.add_collection3d(poly1)
    ax2.add_collection3d(poly2)

    sol_line1, = ax1.plot([], [], [], linewidth=2)
    nor_line1, = ax1.plot([], [], [], linewidth=2)

    sol_line2, = ax2.plot([], [], [], linewidth=2)
    nor_line2, = ax2.plot([], [], [], linewidth=2)

    texto1 = ax1.text2D(0.05, 0.95, "", transform=ax1.transAxes)
    texto2 = ax2.text2D(0.05, 0.95, "", transform=ax2.transAxes)

    def init():
        return poly1, poly2, sol_line1, nor_line1, sol_line2, nor_line2, texto1, texto2

    def update(i):
        s = S[i]

        # Newton
        phi_n = float(df.loc[i, "phi_newton_rad"])
        beta_n = float(df.loc[i, "beta_newton_rad"])
        n_n = normal_panel(phi_n, beta_n)
        v1 = _panel_vertices(phi_n, beta_n)
        poly1.set_verts([v1])

        # LM
        phi_lm = float(df.loc[i, "phi_lm_rad"])
        beta_lm = float(df.loc[i, "beta_lm_rad"])
        n_lm = normal_panel(phi_lm, beta_lm)
        v2 = _panel_vertices(phi_lm, beta_lm)
        poly2.set_verts([v2])

        # Vector solar
        sol_scale = 1.2
        sol_line1.set_data([0, sol_scale*s[0]], [0, sol_scale*s[1]])
        sol_line1.set_3d_properties([0, sol_scale*s[2]])

        sol_line2.set_data([0, sol_scale*s[0]], [0, sol_scale*s[1]])
        sol_line2.set_3d_properties([0, sol_scale*s[2]])

        # Normales
        n_scale = 1.2
        nor_line1.set_data([0, n_scale*n_n[0]], [0, n_scale*n_n[1]])
        nor_line1.set_3d_properties([0, n_scale*n_n[2]])

        nor_line2.set_data([0, n_scale*n_lm[0]], [0, n_scale*n_lm[1]])
        nor_line2.set_3d_properties([0, n_scale*n_lm[2]])

        err_n = angulo_incidencia_grados(n_n, s)
        err_lm = angulo_incidencia_grados(n_lm, s)

        texto1.set_text(f"Frame {i}/{len(df)-1}\nIncidencia: {err_n:.6f}°")
        texto2.set_text(f"Frame {i}/{len(df)-1}\nIncidencia: {err_lm:.6f}°")

        return poly1, poly2, sol_line1, nor_line1, sol_line2, nor_line2, texto1, texto2

    anim = FuncAnimation(fig, update, frames=len(df), init_func=init, interval=1000/fps, blit=False)

    if guardar_gif:
        ruta_gif = os.path.join(carpeta_salida, "animacion_3d.gif")
        anim.save(ruta_gif, writer=PillowWriter(fps=fps))
        print("GIF guardado en:", ruta_gif)

    if guardar_mp4:
        ruta_mp4 = os.path.join(carpeta_salida, "animacion_3d.mp4")
        try:
            anim.save(ruta_mp4, writer=FFMpegWriter(fps=fps, codec="libx264", bitrate=1800))
            print("MP4 guardado en:", ruta_mp4)
        except Exception as e:
            print("No se pudo guardar MP4. Verifique que ffmpeg esté instalado y en PATH.")
            print("Detalle:", e)

    plt.close(fig)
