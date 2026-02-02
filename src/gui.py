import threading
import tkinter as tk
from datetime import datetime
from tkinter import ttk, messagebox
from tkcalendar import DateEntry

import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.cinematica import normal_panel, rotacion_x, rotacion_y, vector_incidencia_desde_az_el
from src.simulacion import simular_dataframe

LOCATIONS = {
    "Quito (EPN) — Referencia": {
        "lat": -0.1807,
        "lon": -78.4678,
        "alt_m": 2850.0,
        "tz": "America/Guayaquil",
        "divergence": "Sin divergencia teórica esperada (caso base).",
    },
    "Macapá — Newton (cenit/equinoccios)": {
        "lat": 0.035,
        "lon": -51.07,
        "alt_m": 0.0,
        "tz": "America/Belem",
        "divergence": "Newton diverge (cenit en equinoccios, ~12:00).",
    },
    "Kisangani — Newton (paso ≥ 60 s)": {
        "lat": 0.52,
        "lon": 25.2,
        "alt_m": 0.0,
        "tz": "Africa/Lubumbashi",
        "divergence": "Newton diverge (paso temporal finito cerca del cenit).",
    },
    "Longyearbyen — Gradiente (sol de medianoche)": {
        "lat": 78.22,
        "lon": 15.65,
        "alt_m": 0.0,
        "tz": "Arctic/Longyearbyen",
        "divergence": "Gradiente puede no converger (valle plano / azimut cambia sin fin).",
    },
    "Amundsen–Scott — Gradiente (Polo Sur)": {
        "lat": -90.0,
        "lon": 0.0,
        "alt_m": 2835.0,
        "tz": "Antarctica/South_Pole",
        "divergence": "Gradiente puede estancarse (muchas soluciones equivalentes).",
    },
    "Tromsø — Gradiente (día perpetuo)": {
        "lat": 69.65,
        "lon": 18.96,
        "alt_m": 0.0,
        "tz": "Europe/Oslo",
        "divergence": "Gradiente puede oscilar/no converger (valle plano con soluciones equivalentes).",
    },
}


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
        top_container = ttk.Frame(self.root)
        top_container.pack(side=tk.TOP, fill=tk.X)

        frame = ttk.Frame(top_container, padding=12)
        frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        frame.columnconfigure(0, weight=2)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)

        range_frame = ttk.LabelFrame(frame, text="Día de simulación", padding=8)
        range_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        today = datetime.now()
        self.sim_date = tk.StringVar(value=today.strftime("%Y-%m-%d"))
        self.start_time = tk.StringVar(value="06:00")
        self.end_time = tk.StringVar(value="18:00")

        ttk.Label(range_frame, text="Fecha").grid(row=0, column=0, sticky="w")
        ttk.Label(range_frame, text="Inicio (HH:MM)").grid(row=0, column=1, sticky="w", padx=(4, 0))
        ttk.Label(range_frame, text="Fin (HH:MM)").grid(row=0, column=2, sticky="w", padx=(4, 0))
        ttk.Label(range_frame, text="Paso (s)").grid(row=0, column=3, sticky="w", padx=(12, 0))

        DateEntry(
            range_frame,
            textvariable=self.sim_date,
            width=12,
            date_pattern="yyyy-mm-dd",
        ).grid(row=1, column=0, sticky="w")
        ttk.Entry(range_frame, textvariable=self.start_time, width=8).grid(
            row=1, column=1, padx=(4, 0), sticky="w"
        )
        ttk.Entry(range_frame, textvariable=self.end_time, width=8).grid(
            row=1, column=2, padx=(4, 0), sticky="w"
        )

        self.step_seconds = tk.IntVar(value=300)
        ttk.Entry(range_frame, textvariable=self.step_seconds, width=6).grid(
            row=1, column=3, padx=(4, 0), sticky="w"
        )

        ttk.Label(range_frame, text="FPS (time-lapse)").grid(
            row=2, column=2, sticky="w", padx=(12, 0), pady=(6, 0)
        )
        self.fps = tk.IntVar(value=20)
        ttk.Entry(range_frame, textvariable=self.fps, width=6).grid(
            row=2, column=3, padx=(4, 0), pady=(6, 0), sticky="w"
        )

        ttk.Label(
            range_frame,
            text="Movimiento activo solo entre 06:00 y 18:00",
            foreground="#555",
        ).grid(row=3, column=0, columnspan=4, sticky="w", pady=(8, 0))

        buttons_frame = ttk.Frame(frame)
        buttons_frame.grid(row=0, column=1, sticky="n", pady=(10, 0))

        self.btn_newton = ttk.Button(
            buttons_frame,
            text="Método 1 (Newton)",
            command=lambda: self._run_simulation("newton"),
        )
        self.btn_newton.grid(row=0, column=0, padx=6, pady=2, sticky="ew")

        self.btn_gradiente = ttk.Button(
            buttons_frame,
            text="Método 2 (Gradiente paso fijo)",
            command=lambda: self._run_simulation("gradiente"),
        )
        self.btn_gradiente.grid(row=0, column=1, padx=6, pady=2, sticky="ew")

        self.btn_dual_3d = ttk.Button(
            buttons_frame,
            text="Comparar 3D",
            command=lambda: self._run_simulation("dual_3d"),
        )
        self.btn_dual_3d.grid(row=1, column=0, padx=6, pady=2, sticky="ew")

        self.btn_dual_plot = ttk.Button(
            buttons_frame,
            text="Comparar gráficas",
            command=lambda: self._run_simulation("dual_plot"),
        )
        self.btn_dual_plot.grid(row=1, column=1, padx=6, pady=2, sticky="ew")

        self.btn_stop = ttk.Button(
            buttons_frame,
            text="Detener animación",
            command=self._stop_animation,
        )
        self.btn_stop.grid(row=2, column=0, columnspan=2, padx=6, pady=(6, 2), sticky="ew")

        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=1)

        location_frame = ttk.LabelFrame(frame, text="Ubicación", padding=8)
        location_frame.grid(row=0, column=2, sticky="nsew")
        self.location_var = tk.StringVar(value="Quito (EPN) — Referencia")
        self.location_combo = ttk.Combobox(
            location_frame,
            textvariable=self.location_var,
            values=list(LOCATIONS.keys()),
            state="readonly",
            width=36,
        )
        self.location_combo.grid(row=0, column=0, sticky="w")
        self.location_combo.bind("<<ComboboxSelected>>", self._on_location_change)

        self.location_lat = tk.StringVar()
        self.location_lon = tk.StringVar()
        self.location_alt = tk.StringVar()
        self.location_tz = tk.StringVar()
        self.location_divergence = tk.StringVar()

        ttk.Label(location_frame, textvariable=self.location_lat).grid(row=1, column=0, sticky="w")
        ttk.Label(location_frame, textvariable=self.location_lon).grid(row=2, column=0, sticky="w")
        ttk.Label(location_frame, textvariable=self.location_alt).grid(row=3, column=0, sticky="w")
        ttk.Label(location_frame, textvariable=self.location_tz).grid(row=4, column=0, sticky="w")
        ttk.Label(location_frame, textvariable=self.location_divergence, wraplength=260).grid(
            row=5, column=0, sticky="w", pady=(6, 0)
        )

        self._set_location_display(self.location_var.get())

        self.status = tk.StringVar(value="Listo para simular.")
        ttk.Label(frame, textvariable=self.status).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )

        stats_frame = ttk.LabelFrame(top_container, text="Datos de métodos", padding=8)
        stats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 12), pady=12)

        stats_table_frame = ttk.Frame(stats_frame)
        stats_table_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.stats_table = ttk.Treeview(
            stats_table_frame,
            columns=("metric", "newton", "gradiente"),
            show="headings",
            height=5,
        )
        self.stats_table.heading("metric", text="Métrica")
        self.stats_table.heading("newton", text="Newton-Raphson")
        self.stats_table.heading("gradiente", text="Gradiente (paso fijo)")
        self.stats_table.column("metric", width=220, anchor="w", stretch=False)
        self.stats_table.column("newton", width=180, anchor="center", stretch=False)
        self.stats_table.column("gradiente", width=180, anchor="center", stretch=False)
        self.stats_table.pack(side=tk.TOP, fill=tk.X, expand=True)

        stats_scroll_x = ttk.Scrollbar(
            stats_table_frame,
            orient="horizontal",
            command=self.stats_table.xview,
        )
        self.stats_table.configure(xscrollcommand=stats_scroll_x.set)
        stats_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

    def _setup_figures(self) -> None:
        self.figure = Figure(figsize=(13, 4.8), dpi=100)
        canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = canvas
        self.axes = {}
        self.artists = {}
        self._configure_axes("newton")

    def _configure_axes(self, mode: str) -> None:
        self.figure.clear()
        gs = self.figure.add_gridspec(1, 2, wspace=0.3)
        self.axes = {}
        self.artists = {}

        if mode == "dual_3d":
            ax_left = self.figure.add_subplot(gs[0, 0], projection="3d")
            ax_right = self.figure.add_subplot(gs[0, 1], projection="3d")
            self.axes["left"] = ax_left
            self.axes["right"] = ax_right
            self.artists["left_3d"] = self._init_3d_axis(ax_left, "Panel (Newton)", "Normal (Newton)")
            self.artists["right_3d"] = self._init_3d_axis(
                ax_right, "Panel (Gradiente)", "Normal (Gradiente)"
            )
        elif mode == "dual_plot":
            ax_left = self.figure.add_subplot(gs[0, 0])
            ax_right = self.figure.add_subplot(gs[0, 1])
            self.axes["left"] = ax_left
            self.axes["right"] = ax_right
            self.artists["left_plot"] = self._init_plot_axis(ax_left, "Ángulos - Newton")
            self.artists["right_plot"] = self._init_plot_axis(ax_right, "Ángulos - Gradiente")
        else:
            ax_left = self.figure.add_subplot(gs[0, 0], projection="3d")
            ax_right = self.figure.add_subplot(gs[0, 1])
            self.axes["left"] = ax_left
            self.axes["right"] = ax_right
            normal_label = "Normal (Newton)" if mode == "newton" else "Normal (Gradiente)"
            title_left = "Movimiento del panel y posición del sol"
            title_right = "Ángulos vs. posición solar"
            self.artists["left_3d"] = self._init_3d_axis(ax_left, title_left, normal_label)
            self.artists["right_plot"] = self._init_plot_axis(ax_right, title_right)

        self.canvas.draw_idle()

    def _init_3d_axis(self, ax, title: str, normal_label: str) -> dict:
        ax.set_title(title)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_xlabel("Este (x)")
        ax.set_ylabel("Norte (y)")
        ax.set_zlabel("Arriba (z)")

        panel_poly = Poly3DCollection([], alpha=0.6, facecolor="#4A90E2")
        ax.add_collection3d(panel_poly)
        sun_vec_line, = ax.plot([], [], [], color="#F5A623", linewidth=2, label="Vector del sol")
        normal_vec_line, = ax.plot([], [], [], color="#4A90E2", linewidth=2, label=normal_label)
        sun_sphere = ax.scatter([], [], [], s=80, color="#F5A623", label="Sol")
        time_text = ax.text2D(0.05, 0.92, "", transform=ax.transAxes)

        ax.legend(loc="upper right")

        return {
            "panel_poly": panel_poly,
            "sun_vec_line": sun_vec_line,
            "normal_vec_line": normal_vec_line,
            "sun_sphere": sun_sphere,
            "time_text": time_text,
        }

    def _init_plot_axis(self, ax, title: str) -> dict:
        ax.set_title(title)
        ax.set_xlabel("Paso de tiempo")
        ax.set_ylabel("Ángulo (deg)")
        ax.grid(True)

        roll_line, = ax.plot([], [], label="roll (panel)")
        pitch_line, = ax.plot([], [], label="pitch (panel)")
        elev_line, = ax.plot([], [], label="elevación (sol)")

        ax.legend(loc="upper right")

        return {
            "roll_line": roll_line,
            "pitch_line": pitch_line,
            "elev_line": elev_line,
        }

    def _set_plot_limits(self, ax, roll_deg: np.ndarray, pitch_deg: np.ndarray, elev_deg: np.ndarray) -> None:
        ax.set_xlim(0, len(roll_deg))
        min_angle = min(np.min(roll_deg), np.min(pitch_deg), np.min(elev_deg))
        max_angle = max(np.max(roll_deg), np.max(pitch_deg), np.max(elev_deg))
        pad = max(5, (max_angle - min_angle) * 0.1)
        ax.set_ylim(min_angle - pad, max_angle + pad)

    def _toggle_buttons(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for btn in (self.btn_newton, self.btn_gradiente, self.btn_dual_3d, self.btn_dual_plot):
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
        if hasattr(self, "figure"):
            self.figure.clear()
        if hasattr(self, "canvas"):
            self.canvas.draw_idle()

    def _parse_datetime(self, date_str: str, time_str: str) -> str:
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        return dt.isoformat()

    def _build_time_range(self) -> tuple[str, str]:
        start_date = self.sim_date.get().strip()
        start_time = self.start_time.get().strip()
        end_time = self.end_time.get().strip()

        if not start_date or not start_time:
            raise ValueError("Complete fecha y hora de inicio.")

        if not end_time:
            end_time = "18:00"

        inicio = self._parse_datetime(start_date, start_time)
        fin = self._parse_datetime(start_date, end_time)

        if fin < inicio:
            raise ValueError("La fecha final no puede ser anterior a la inicial.")

        return (inicio, fin)

    def _run_simulation(self, mode: str) -> None:
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
            args=(inicio_iso, fin_iso, step, mode),
            daemon=True,
        )
        thread.start()

    def _on_location_change(self, _event: tk.Event) -> None:
        self._set_location_display(self.location_var.get())

    def _set_location_display(self, location_key: str) -> None:
        location = LOCATIONS[location_key]
        self.location_lat.set(f"Lat: {location['lat']:.3f}")
        self.location_lon.set(f"Lon: {location['lon']:.3f}")
        self.location_alt.set(f"Alt: {location['alt_m']:.1f} m")
        self.location_tz.set(f"TZ: {location['tz']}")
        self.location_divergence.set(f"Divergencia: {location['divergence']}")

    def _simulate_worker(self, inicio_iso: str, fin_iso: str, paso_seg: int, mode: str) -> None:
        location = LOCATIONS[self.location_var.get()]
        try:
            df, stats = simular_dataframe(
                inicio_iso=inicio_iso,
                fin_iso=fin_iso,
                paso_seg=paso_seg,
                lat=location["lat"],
                lon=location["lon"],
                alt_m=location["alt_m"],
                zona_horaria=location["tz"],
                backend="pysolar",
                return_stats=True,
            )
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("Error", str(exc)))
            self.root.after(0, lambda: self.status.set("Error en la simulación."))
            self.root.after(0, lambda: self._toggle_buttons(True))
            return

        self.root.after(0, lambda: self._start_animation(df, stats, mode))

    def _start_animation(self, df, stats: dict, mode: str) -> None:
        self.df = df.reset_index(drop=True)
        self._stop_animation()
        self._reset_artists()
        self._update_stats_table(stats)
        self._configure_axes(mode)

        az = np.radians(self.df["azimut_deg"].to_numpy())
        el = np.radians(self.df["elevacion_deg"].to_numpy())

        phi_newton = self.df["phi_newton_rad"].to_numpy()
        beta_newton = self.df["beta_newton_rad"].to_numpy()
        phi_grad = self.df["phi_grad_rad"].to_numpy()
        beta_grad = self.df["beta_grad_rad"].to_numpy()

        roll_newton = np.degrees(phi_newton)
        pitch_newton = np.degrees(beta_newton)
        roll_grad = np.degrees(phi_grad)
        pitch_grad = np.degrees(beta_grad)
        elev_deg = np.degrees(el)

        if "right_plot" in self.artists and mode in ("newton", "gradiente"):
            roll = roll_newton if mode == "newton" else roll_grad
            pitch = pitch_newton if mode == "newton" else pitch_grad
            self._set_plot_limits(self.axes["right"], roll, pitch, elev_deg)
        if mode == "dual_plot":
            self._set_plot_limits(self.axes["left"], roll_newton, pitch_newton, elev_deg)
            self._set_plot_limits(self.axes["right"], roll_grad, pitch_grad, elev_deg)

        def update_3d(artists: dict, phi: np.ndarray, beta: np.ndarray, label: str, i: int):
            s_vec = vector_incidencia_desde_az_el(az[i], el[i])
            verts = _panel_vertices(phi[i], beta[i])
            artists["panel_poly"].set_verts([verts])

            scale = 1.2
            artists["sun_vec_line"].set_data([0, scale * s_vec[0]], [0, scale * s_vec[1]])
            artists["sun_vec_line"].set_3d_properties([0, scale * s_vec[2]])
            artists["sun_sphere"]._offsets3d = (
                [scale * s_vec[0]],
                [scale * s_vec[1]],
                [scale * s_vec[2]],
            )

            n_vec = normal_panel(phi[i], beta[i])
            artists["normal_vec_line"].set_data([0, scale * n_vec[0]], [0, scale * n_vec[1]])
            artists["normal_vec_line"].set_3d_properties([0, scale * n_vec[2]])

            tiempo = self.df.loc[i, "tiempo"]
            artists["time_text"].set_text(f"{label}\n{tiempo}")

            return (
                artists["panel_poly"],
                artists["sun_vec_line"],
                artists["normal_vec_line"],
                artists["sun_sphere"],
                artists["time_text"],
            )

        def update_plot(artists: dict, roll_deg: np.ndarray, pitch_deg: np.ndarray, elev: np.ndarray, i: int):
            x = np.arange(i + 1)
            artists["roll_line"].set_data(x, roll_deg[: i + 1])
            artists["pitch_line"].set_data(x, pitch_deg[: i + 1])
            artists["elev_line"].set_data(x, elev[: i + 1])
            return (
                artists["roll_line"],
                artists["pitch_line"],
                artists["elev_line"],
            )

        def update(i: int):
            updated = []
            if mode == "dual_3d":
                updated += update_3d(
                    self.artists["left_3d"], phi_newton, beta_newton, "Método 1 (Newton)", i
                )
                updated += update_3d(
                    self.artists["right_3d"],
                    phi_grad,
                    beta_grad,
                    "Método 2 (Gradiente)",
                    i,
                )
            elif mode == "dual_plot":
                updated += update_plot(self.artists["left_plot"], roll_newton, pitch_newton, elev_deg, i)
                updated += update_plot(self.artists["right_plot"], roll_grad, pitch_grad, elev_deg, i)
            elif mode == "gradiente":
                updated += update_3d(
                    self.artists["left_3d"], phi_grad, beta_grad, "Método 2 (Gradiente)", i
                )
                updated += update_plot(self.artists["right_plot"], roll_grad, pitch_grad, elev_deg, i)
            else:
                updated += update_3d(
                    self.artists["left_3d"], phi_newton, beta_newton, "Método 1 (Newton)", i
                )
                updated += update_plot(self.artists["right_plot"], roll_newton, pitch_newton, elev_deg, i)

            return tuple(updated)

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

    def _update_stats_table(self, stats: dict) -> None:
        for row in self.stats_table.get_children():
            self.stats_table.delete(row)

        if not stats:
            return

        self.stats_table.insert(
            "",
            "end",
            values=("Complejidad", stats["newton"]["complejidad"], stats["gradiente"]["complejidad"]),
        )
        self.stats_table.insert(
            "",
            "end",
            values=(
                "Iteraciones promedio",
                f"{stats['newton']['iteraciones_promedio']:.2f}",
                f"{stats['gradiente']['iteraciones_promedio']:.2f}",
            ),
        )
        self.stats_table.insert(
            "",
            "end",
            values=(
                "Tiempo total (s)",
                f"{stats['newton']['tiempo_total_s']:.4f}",
                f"{stats['gradiente']['tiempo_total_s']:.4f}",
            ),
        )
        self.stats_table.insert(
            "",
            "end",
            values=(
                "Precisión final (deg)",
                f"{stats['newton']['precision_final_deg']:.4f}",
                f"{stats['gradiente']['precision_final_deg']:.4f}",
            ),
        )
        self.stats_table.insert(
            "",
            "end",
            values=("Estabilidad numérica", stats["newton"]["estabilidad"], stats["gradiente"]["estabilidad"]),
        )


def launch_gui() -> None:
    root = tk.Tk()
    app = SolarTrackerGUI(root)
    root.mainloop()
