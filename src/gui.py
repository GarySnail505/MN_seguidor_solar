import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.simulacion import simular_df
from src.cinematica import (
    vector_incidencia_desde_az_el,
    normal_panel,
    rotacion_x,
    rotacion_y,
    angulo_incidencia_grados
)

DEFAULT_LAT = -0.1807
DEFAULT_LON = -78.4678
DEFAULT_ALT = 2850.0
DEFAULT_TZ = "America/Guayaquil"


def _panel_vertices(phi: float, beta: float, ancho=1.2, alto=0.7):
    hw, hh = ancho / 2, alto / 2
    verts_local = np.array([
        [-hw, -hh, 0],
        [ hw, -hh, 0],
        [ hw,  hh, 0],
        [-hw,  hh, 0],
    ], dtype=float)

    rot = rotacion_y(phi) @ rotacion_x(beta)
    return (rot @ verts_local.T).T


def _parse_datetime(value: str, zona_horaria: str) -> pd.Timestamp:
    if not value:
        raise ValueError("La fecha no puede estar vacía.")
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(zona_horaria)
    return timestamp.tz_convert(zona_horaria)


def _formatear_timestamp(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y-%m-%d %H:%M")


class SeguidorSolarGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Seguidor Solar 2GDL - GUI")
        self.root.geometry("1400x800")

        self.animacion = None
        self.frame_index = 0
        self.df = None
        self.metodo = "newton"

        self._crear_widgets()
        self._configurar_figura()

    def _crear_widgets(self):
        panel_superior = ttk.Frame(self.root, padding=10)
        panel_superior.pack(side=tk.TOP, fill=tk.X)

        entradas_frame = ttk.LabelFrame(panel_superior, text="Rango de simulación", padding=10)
        entradas_frame.pack(side=tk.LEFT, padx=10)

        ttk.Label(entradas_frame, text="Inicio (YYYY-MM-DD HH:MM)").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(entradas_frame, text="Fin (opcional)").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(entradas_frame, text="Paso (s)").grid(row=2, column=0, sticky=tk.W)
        ttk.Label(entradas_frame, text="FPS (time-lapse)").grid(row=3, column=0, sticky=tk.W)

        ahora = pd.Timestamp.now(tz=DEFAULT_TZ)
        inicio_default = ahora.normalize() + pd.Timedelta(hours=6)

        self.inicio_var = tk.StringVar(value=_formatear_timestamp(inicio_default))
        self.fin_var = tk.StringVar(value="")
        self.paso_var = tk.StringVar(value="60")
        self.fps_var = tk.StringVar(value="20")

        ttk.Entry(entradas_frame, textvariable=self.inicio_var, width=22).grid(row=0, column=1, padx=5)
        ttk.Entry(entradas_frame, textvariable=self.fin_var, width=22).grid(row=1, column=1, padx=5)
        ttk.Entry(entradas_frame, textvariable=self.paso_var, width=8).grid(row=2, column=1, padx=5, sticky=tk.W)
        ttk.Entry(entradas_frame, textvariable=self.fps_var, width=8).grid(row=3, column=1, padx=5, sticky=tk.W)

        botones_frame = ttk.Frame(panel_superior, padding=10)
        botones_frame.pack(side=tk.LEFT)

        self.boton_newton = ttk.Button(
            botones_frame,
            text="Método 1 (Newton)",
            command=lambda: self._iniciar_simulacion("newton")
        )
        self.boton_newton.grid(row=0, column=0, padx=5, pady=5)

        self.boton_lm = ttk.Button(
            botones_frame,
            text="Método 2 (Levenberg-Marquardt)",
            command=lambda: self._iniciar_simulacion("lm")
        )
        self.boton_lm.grid(row=0, column=1, padx=5, pady=5)

        ttk.Button(
            botones_frame,
            text="Detener",
            command=self._detener_animacion
        ).grid(row=0, column=2, padx=5, pady=5)

        info_frame = ttk.LabelFrame(panel_superior, text="Ubicación (Quito, EPN)", padding=10)
        info_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(info_frame, text=f"Lat: {DEFAULT_LAT}").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(info_frame, text=f"Lon: {DEFAULT_LON}").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(info_frame, text=f"Alt: {DEFAULT_ALT} m").grid(row=2, column=0, sticky=tk.W)
        ttk.Label(info_frame, text=f"TZ: {DEFAULT_TZ}").grid(row=3, column=0, sticky=tk.W)

        self.estado_var = tk.StringVar(value="Listo para simular.")
        ttk.Label(panel_superior, textvariable=self.estado_var, foreground="#1f77b4").pack(side=tk.RIGHT)

    def _configurar_figura(self):
        self.fig = Figure(figsize=(14, 6), constrained_layout=True)
        gs = self.fig.add_gridspec(2, 3, width_ratios=[1.1, 1.4, 1.4])

        self.ax_solar = self.fig.add_subplot(gs[0, 0], projection="polar")
        self.ax_panel = self.fig.add_subplot(gs[:, 1], projection="3d")
        self.ax_grafica = self.fig.add_subplot(gs[:, 2])

        self.ax_solar.set_title("Posición solar")
        self.ax_solar.set_theta_zero_location("N")
        self.ax_solar.set_theta_direction(-1)
        self.ax_solar.set_rlim(90, 0)
        self.ax_solar.set_rlabel_position(135)

        self.ax_panel.set_title("Panel solar")
        self.ax_panel.set_xlim(-1.5, 1.5)
        self.ax_panel.set_ylim(-1.5, 1.5)
        self.ax_panel.set_zlim(-1.5, 1.5)
        self.ax_panel.set_xlabel("Este (x)")
        self.ax_panel.set_ylabel("Norte (y)")
        self.ax_panel.set_zlabel("Arriba (z)")

        self.ax_grafica.set_title("Ángulos del panel vs sol")
        self.ax_grafica.set_xlabel("Paso de tiempo")
        self.ax_grafica.set_ylabel("Ángulo (deg)")
        self.ax_grafica.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _detener_animacion(self):
        if self.animacion:
            self.animacion.event_source.stop()
        self.estado_var.set("Simulación detenida.")

    def _obtener_rango(self):
        inicio = _parse_datetime(self.inicio_var.get().strip(), DEFAULT_TZ)
        fin_texto = self.fin_var.get().strip()

        if fin_texto:
            fin = _parse_datetime(fin_texto, DEFAULT_TZ)
        else:
            fin = inicio.normalize() + pd.Timedelta(hours=23, minutes=59)

        return inicio, fin

    def _iniciar_simulacion(self, metodo: str):
        try:
            inicio, fin = self._obtener_rango()
            paso = int(self.paso_var.get())
            fps = max(1, int(self.fps_var.get()))
        except Exception as exc:
            messagebox.showerror("Entrada inválida", str(exc))
            return

        self.metodo = metodo
        self.estado_var.set("Calculando simulación...")
        self.root.update_idletasks()

        try:
            self.df = simular_df(
                inicio_iso=inicio.isoformat(),
                fin_iso=fin.isoformat(),
                paso_seg=paso,
                lat=DEFAULT_LAT,
                lon=DEFAULT_LON,
                alt_m=DEFAULT_ALT,
                zona_horaria=DEFAULT_TZ,
                backend="pvlib"
            )
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo simular: {exc}")
            self.estado_var.set("Error en la simulación.")
            return

        if self.df.empty:
            messagebox.showwarning("Sin datos", "La simulación no generó datos.")
            return

        self._preparar_graficos()
        self._iniciar_animacion(fps)

    def _preparar_graficos(self):
        df = self.df
        self.ax_solar.cla()
        self.ax_panel.cla()
        self.ax_grafica.cla()

        self.ax_solar.set_title("Posición solar")
        self.ax_solar.set_theta_zero_location("N")
        self.ax_solar.set_theta_direction(-1)
        self.ax_solar.set_rlim(90, 0)
        self.ax_solar.set_rlabel_position(135)

        self.ax_panel.set_title(
            "Panel solar - Método 1 (Newton)" if self.metodo == "newton" else "Panel solar - Método 2 (LM)"
        )
        self.ax_panel.set_xlim(-1.5, 1.5)
        self.ax_panel.set_ylim(-1.5, 1.5)
        self.ax_panel.set_zlim(-1.5, 1.5)
        self.ax_panel.set_xlabel("Este (x)")
        self.ax_panel.set_ylabel("Norte (y)")
        self.ax_panel.set_zlabel("Arriba (z)")

        self.ax_grafica.set_title("Ángulos del panel vs sol")
        self.ax_grafica.set_xlabel("Paso de tiempo")
        self.ax_grafica.set_ylabel("Ángulo (deg)")
        self.ax_grafica.grid(True)

        pasos = np.arange(len(df))
        az = df["azimut_deg"].to_numpy()
        el = df["elevacion_deg"].to_numpy()

        phi_newton = np.degrees(df["phi_newton_rad"].to_numpy())
        beta_newton = np.degrees(df["beta_newton_rad"].to_numpy())
        phi_lm = np.degrees(df["phi_lm_rad"].to_numpy())
        beta_lm = np.degrees(df["beta_lm_rad"].to_numpy())

        self.ax_grafica.plot(pasos, az, label="Azimut sol", color="#ff7f0e")
        self.ax_grafica.plot(pasos, el, label="Elevación sol", color="#f1c40f")
        self.ax_grafica.plot(pasos, phi_newton, label="Roll (Newton)")
        self.ax_grafica.plot(pasos, beta_newton, label="Pitch (Newton)")
        self.ax_grafica.plot(pasos, phi_lm, "--", label="Roll (LM)")
        self.ax_grafica.plot(pasos, beta_lm, "--", label="Pitch (LM)")
        self.ax_grafica.legend(loc="upper right", fontsize=8)

        theta = np.radians(az)
        r = 90 - el
        self.ax_solar.plot(theta, r, color="#2980b9", linewidth=1.5)
        self.sun_point, = self.ax_solar.plot([], [], "o", color="#e74c3c")

        self.panel_poly = Poly3DCollection([], alpha=0.6, facecolor="#2ecc71")
        self.ax_panel.add_collection3d(self.panel_poly)

        self.sol_line, = self.ax_panel.plot([], [], [], color="#f39c12", linewidth=2)
        self.normal_line, = self.ax_panel.plot([], [], [], color="#2c3e50", linewidth=2)

        self.marker_line = self.ax_grafica.axvline(0, color="black", linestyle=":")
        self.status_text = self.ax_panel.text2D(0.02, 0.95, "", transform=self.ax_panel.transAxes)

    def _iniciar_animacion(self, fps: int):
        if self.animacion:
            self.animacion.event_source.stop()

        self.frame_index = 0
        intervalo = int(1000 / fps)

        def actualizar(frame):
            df = self.df
            if frame >= len(df):
                return

            az = float(df.loc[frame, "azimut_deg"])
            el = float(df.loc[frame, "elevacion_deg"])

            theta = np.radians(az)
            r = 90 - el
            self.sun_point.set_data([theta], [r])

            az_rad = np.radians(az)
            el_rad = np.radians(el)
            s = vector_incidencia_desde_az_el(az_rad, el_rad)

            if self.metodo == "newton":
                phi = float(df.loc[frame, "phi_newton_rad"])
                beta = float(df.loc[frame, "beta_newton_rad"])
            else:
                phi = float(df.loc[frame, "phi_lm_rad"])
                beta = float(df.loc[frame, "beta_lm_rad"])

            n_vec = normal_panel(phi, beta)
            verts = _panel_vertices(phi, beta)
            self.panel_poly.set_verts([verts])

            sol_scale = 1.2
            self.sol_line.set_data([0, sol_scale * s[0]], [0, sol_scale * s[1]])
            self.sol_line.set_3d_properties([0, sol_scale * s[2]])

            n_scale = 1.2
            self.normal_line.set_data([0, n_scale * n_vec[0]], [0, n_scale * n_vec[1]])
            self.normal_line.set_3d_properties([0, n_scale * n_vec[2]])

            error = angulo_incidencia_grados(n_vec, s)
            tiempo = df.loc[frame, "tiempo"]
            self.status_text.set_text(f"{tiempo}\nIncidencia: {error:.4f}°")

            self.marker_line.set_xdata(frame)
            self.canvas.draw_idle()

        self.animacion = FuncAnimation(
            self.fig,
            actualizar,
            frames=len(self.df),
            interval=intervalo,
            repeat=False
        )
        self.estado_var.set("Simulación en curso...")


def main():
    root = tk.Tk()
    app = SeguidorSolarGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
