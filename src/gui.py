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
    rot = rotacion_y(phi) @ rotacion_x(beta)
    return (rot @ verts_local.T).T


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
        self.ax_panel.set_xlim(-1.5, 1.5)
        self.ax_panel.set_ylim(-1.5, 1.5)
        self.ax_panel.set_zlim(-1.5, 1.5)
        self.ax_panel.set_xlabel("Este (x)")
        self.ax_panel.set_ylabel("Norte (y)")
        self.ax_panel.set_zlabel("Arriba (z)")

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
                self.animation.event_source.stop()
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
                backend="pvlib",
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
