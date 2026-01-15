<<<<<<< HEAD
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

=======
import threading
import tkinter as tk
from datetime import datetime
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.cinematica import normal_panel, rotacion_x, rotacion_y, vector_incidencia_desde_az_el
from src.simulacion import simular_dataframe

LAT_Quito = -0.1807
LON_Quito = -78.4678
ALT_Quito = 2850.0
ZONA_Quito = "America/Guayaquil"


def _panel_vertices(phi: float, beta: float, ancho: float = 1.2, alto: float = 0.7) -> np.ndarray:
    hw, hh = ancho / 2, alto / 2
    verts_local = np.array(
        [
            [-hw, -hh, 0],
            [hw, -hh, 0],
            [hw, hh, 0],
            [-hw, hh, 0],
        ],
        dtype=float,
    )
>>>>>>> rama_gui
    rot = rotacion_y(phi) @ rotacion_x(beta)
    return (rot @ verts_local.T).T


<<<<<<< HEAD
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
=======
class SolarTrackerGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Simulador de Seguidor Solar 2GDL")
        self.animation = None
        self.df = None
        self._animation_lock = threading.Lock()

        self._setup_inputs()
        self._setup_figures()

    def _setup_inputs(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(side=tk.TOP, fill=tk.X)

        frame.columnconfigure(0, weight=2)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)

        range_frame = ttk.LabelFrame(frame, text="Rango de simulación", padding=8)
        range_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        today = datetime.now()
        self.start_date = tk.StringVar(value=today.strftime("%Y-%m-%d"))
        self.start_time = tk.StringVar(value="06:00")
        self.end_date = tk.StringVar(value=today.strftime("%Y-%m-%d"))
        self.end_time = tk.StringVar(value="18:00")
        self.full_day = tk.BooleanVar(value=False)

        ttk.Label(range_frame, text="Inicio (YYYY-MM-DD)").grid(row=0, column=0, sticky="w")
        ttk.Label(range_frame, text="HH:MM").grid(row=0, column=1, sticky="w", padx=(4, 0))
        ttk.Label(range_frame, text="Fin (opcional)").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(range_frame, text="HH:MM").grid(
            row=1, column=1, sticky="w", padx=(4, 0), pady=(6, 0)
        )

        ttk.Entry(range_frame, textvariable=self.start_date, width=12).grid(row=0, column=0)
        ttk.Entry(range_frame, textvariable=self.start_time, width=8).grid(
            row=0, column=1, padx=(4, 0)
        )
        ttk.Entry(range_frame, textvariable=self.end_date, width=12).grid(
            row=1, column=0, pady=(6, 0)
        )
        ttk.Entry(range_frame, textvariable=self.end_time, width=8).grid(
            row=1, column=1, padx=(4, 0), pady=(6, 0)
        )

        ttk.Label(range_frame, text="Paso (s)").grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.step_seconds = tk.IntVar(value=300)
        ttk.Entry(range_frame, textvariable=self.step_seconds, width=6).grid(
            row=0, column=3, padx=(4, 0)
        )

        ttk.Label(range_frame, text="FPS (time-lapse)").grid(
            row=1, column=2, sticky="w", padx=(12, 0), pady=(6, 0)
        )
        self.fps = tk.IntVar(value=20)
        ttk.Entry(range_frame, textvariable=self.fps, width=6).grid(
            row=1, column=3, padx=(4, 0), pady=(6, 0)
        )

        ttk.Checkbutton(range_frame, text="Simular día completo", variable=self.full_day).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        buttons_frame = ttk.Frame(frame)
        buttons_frame.grid(row=0, column=1, sticky="n", pady=(10, 0))

        self.btn_newton = ttk.Button(
            buttons_frame,
            text="Método 1 (Newton)",
            command=lambda: self._run_simulation("newton"),
        )
        self.btn_newton.pack(side=tk.LEFT, padx=6)

        self.btn_lm = ttk.Button(
            buttons_frame,
            text="Método 2 (Levenberg–Marquardt)",
            command=lambda: self._run_simulation("lm"),
        )
        self.btn_lm.pack(side=tk.LEFT, padx=6)

        self.btn_stop = ttk.Button(
            buttons_frame,
            text="Detener animación",
            command=self._stop_animation,
        )
        self.btn_stop.pack(side=tk.LEFT, padx=6)

        location_frame = ttk.LabelFrame(frame, text="Ubicación (Quito, EPN)", padding=8)
        location_frame.grid(row=0, column=2, sticky="nsew")
        ttk.Label(location_frame, text="Lat: -0.1807").grid(row=0, column=0, sticky="w")
        ttk.Label(location_frame, text="Lon: -78.4678").grid(row=1, column=0, sticky="w")
        ttk.Label(location_frame, text="Alt: 2850.0 m").grid(row=2, column=0, sticky="w")
        ttk.Label(location_frame, text="TZ: America/Guayaquil").grid(row=3, column=0, sticky="w")

        self.status = tk.StringVar(value="Listo para simular.")
        ttk.Label(frame, textvariable=self.status).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )

    def _setup_figures(self) -> None:
        self.figure = Figure(figsize=(13, 4.8), dpi=100)
        gs = self.figure.add_gridspec(1, 2, wspace=0.3)

        self.ax_panel = self.figure.add_subplot(gs[0, 0], projection="3d")
        self.ax_plot = self.figure.add_subplot(gs[0, 1])

        self.ax_panel.set_title("Movimiento del panel y posición del sol")
>>>>>>> rama_gui
        self.ax_panel.set_xlim(-1.5, 1.5)
        self.ax_panel.set_ylim(-1.5, 1.5)
        self.ax_panel.set_zlim(-1.5, 1.5)
        self.ax_panel.set_xlabel("Este (x)")
        self.ax_panel.set_ylabel("Norte (y)")
        self.ax_panel.set_zlabel("Arriba (z)")

<<<<<<< HEAD
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
=======
        self.ax_plot.set_title("Ángulos vs. posición solar")
        self.ax_plot.set_xlabel("Paso de tiempo")
        self.ax_plot.set_ylabel("Ángulo (deg)")
        self.ax_plot.grid(True)

        self.panel_poly = Poly3DCollection([], alpha=0.6)
        self.ax_panel.add_collection3d(self.panel_poly)
        self.sun_vec_line, = self.ax_panel.plot(
            [], [], [], color="#F5A623", linewidth=2, label="Vector del sol"
        )
        self.normal_vec_line, = self.ax_panel.plot(
            [], [], [], color="#4A90E2", linewidth=2, label="Normal del panel"
        )
        self.sun_sphere = self.ax_panel.scatter([], [], [], s=80, color="#F5A623", label="Sol")
        self.time_text = self.ax_panel.text2D(0.05, 0.92, "", transform=self.ax_panel.transAxes)

        self.roll_line, = self.ax_plot.plot([], [], label="roll (panel)")
        self.pitch_line, = self.ax_plot.plot([], [], label="pitch (panel)")
        self.elev_line, = self.ax_plot.plot([], [], label="elevación (sol)")
        self.ax_plot.legend(loc="upper right")

        self.ax_panel.legend(loc="upper left")

        canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = canvas

    def _toggle_buttons(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for btn in (self.btn_newton, self.btn_lm):
            btn.configure(state=state)

    def _stop_animation(self) -> None:
        with self._animation_lock:
            if self.animation is not None:
                # En algunos casos (por ejemplo, cuando la animación ya terminó),
                # event_source puede ser None. Se protege el stop() para evitar
                # errores al volver a ejecutar.
                event_source = getattr(self.animation, "event_source", None)
                if event_source is not None:
                    try:
                        event_source.stop()
                    except Exception:
                        pass

                self.animation = None
                self.status.set("Animación detenida.")

        self._reset_artists()

    def _reset_artists(self) -> None:
        self.sun_vec_line.set_data([], [])
        self.sun_vec_line.set_3d_properties([])
        self.normal_vec_line.set_data([], [])
        self.normal_vec_line.set_3d_properties([])
        self.panel_poly.set_verts([])
        self.roll_line.set_data([], [])
        self.pitch_line.set_data([], [])
        self.elev_line.set_data([], [])
        self.sun_sphere._offsets3d = ([], [], [])
        self.time_text.set_text("")
        self.canvas.draw_idle()

    def _parse_datetime(self, date_str: str, time_str: str) -> str:
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        return dt.isoformat()

    def _build_time_range(self) -> tuple[str, str]:
        if self.full_day.get():
            date = self.start_date.get().strip()
            return f"{date}T00:00", f"{date}T23:59"

        start_date = self.start_date.get().strip()
        start_time = self.start_time.get().strip()
        end_date = self.end_date.get().strip()
        end_time = self.end_time.get().strip()

        if not start_date or not start_time:
            raise ValueError("Complete fecha y hora de inicio.")

        if not end_date or not end_time:
            end_date = start_date
            end_time = "23:59"

        return (
            self._parse_datetime(start_date, start_time),
            self._parse_datetime(end_date, end_time),
        )

    def _run_simulation(self, method: str) -> None:
        try:
            inicio_iso, fin_iso = self._build_time_range()
        except ValueError as exc:
            messagebox.showerror("Datos inválidos", str(exc))
            return

        step = self.step_seconds.get()
        if step <= 0:
            messagebox.showerror("Datos inválidos", "El paso debe ser un entero positivo.")
            return

        fps = self.fps.get()
        if fps <= 0:
            messagebox.showerror("Datos inválidos", "El FPS debe ser un entero positivo.")
            return

        self._toggle_buttons(False)
        self.status.set("Calculando simulación...")

        thread = threading.Thread(
            target=self._simulate_worker,
            args=(inicio_iso, fin_iso, step, method),
            daemon=True,
        )
        thread.start()

    def _simulate_worker(self, inicio_iso: str, fin_iso: str, paso_seg: int, method: str) -> None:
        try:
            df = simular_dataframe(
                inicio_iso=inicio_iso,
                fin_iso=fin_iso,
                paso_seg=paso_seg,
                lat=LAT_Quito,
                lon=LON_Quito,
                alt_m=ALT_Quito,
                zona_horaria=ZONA_Quito,
                backend="pysolar",
            )
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("Error", str(exc)))
            self.root.after(0, lambda: self.status.set("Error en la simulación."))
            self.root.after(0, lambda: self._toggle_buttons(True))
            return

        self.root.after(0, lambda: self._start_animation(df, method))

    def _start_animation(self, df, method: str) -> None:
        self.df = df.reset_index(drop=True)
        self._stop_animation()
        self._reset_artists()

        az = np.radians(self.df["azimut_deg"].to_numpy())
        el = np.radians(self.df["elevacion_deg"].to_numpy())

        if method == "newton":
            phi = self.df["phi_newton_rad"].to_numpy()
            beta = self.df["beta_newton_rad"].to_numpy()
            method_label = "Método 1 (Newton)"
        else:
            phi = self.df["phi_lm_rad"].to_numpy()
            beta = self.df["beta_lm_rad"].to_numpy()
            method_label = "Método 2 (Levenberg–Marquardt)"

        roll_deg = np.degrees(phi)
        pitch_deg = np.degrees(beta)
        elev_deg = np.degrees(el)

        self.ax_plot.set_title(f"Ángulos vs. posición solar - {method_label}")
        self.ax_plot.set_xlim(0, len(df))
        min_angle = min(np.min(roll_deg), np.min(pitch_deg), np.min(elev_deg))
        max_angle = max(np.max(roll_deg), np.max(pitch_deg), np.max(elev_deg))
        pad = max(5, (max_angle - min_angle) * 0.1)
        self.ax_plot.set_ylim(min_angle - pad, max_angle + pad)

        def update(i: int):
            s_vec = vector_incidencia_desde_az_el(az[i], el[i])
            verts = _panel_vertices(phi[i], beta[i])
            self.panel_poly.set_verts([verts])

            scale = 1.2
            self.sun_vec_line.set_data([0, scale * s_vec[0]], [0, scale * s_vec[1]])
            self.sun_vec_line.set_3d_properties([0, scale * s_vec[2]])
            self.sun_sphere._offsets3d = (
                [scale * s_vec[0]],
                [scale * s_vec[1]],
                [scale * s_vec[2]],
            )

            n_vec = normal_panel(phi[i], beta[i])
            self.normal_vec_line.set_data([0, scale * n_vec[0]], [0, scale * n_vec[1]])
            self.normal_vec_line.set_3d_properties([0, scale * n_vec[2]])

            x = np.arange(i + 1)
            self.roll_line.set_data(x, roll_deg[: i + 1])
            self.pitch_line.set_data(x, pitch_deg[: i + 1])
            self.elev_line.set_data(x, elev_deg[: i + 1])

            tiempo = self.df.loc[i, "tiempo"]
            self.time_text.set_text(f"{method_label}\n{tiempo}")

            return (
                self.panel_poly,
                self.sun_vec_line,
                self.normal_vec_line,
                self.sun_sphere,
                self.roll_line,
                self.pitch_line,
                self.elev_line,
                self.time_text,
            )

        with self._animation_lock:
            self.animation = FuncAnimation(
                self.figure,
                update,
                frames=len(self.df),
                interval=max(1, int(1000 / self.fps.get())),
                blit=False,
                repeat=False,
            )

        self.status.set("Animación en marcha.")
        self._toggle_buttons(True)
        self.canvas.draw()


def launch_gui() -> None:
    root = tk.Tk()
    app = SolarTrackerGUI(root)
    root.mainloop()
>>>>>>> rama_gui
