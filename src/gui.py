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

        ttk.Label(
            range_frame,
            text="Horario diurno obligatorio: 06:00–18:00",
            foreground="#555",
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))

        buttons_frame = ttk.Frame(frame)
        buttons_frame.grid(row=0, column=1, sticky="n", pady=(10, 0))

        self.btn_newton = ttk.Button(
            buttons_frame,
            text="Método 1 (Newton)",
            command=lambda: self._run_simulation("newton"),
        )
        self.btn_newton.pack(side=tk.LEFT, padx=6)

        self.btn_gradiente = ttk.Button(
            buttons_frame,
            text="Método 2 (Gradiente paso fijo)",
            command=lambda: self._run_simulation("gradiente"),
        )
        self.btn_gradiente.pack(side=tk.LEFT, padx=6)

        self.btn_dual_3d = ttk.Button(
            buttons_frame,
            text="Comparar 3D",
            command=lambda: self._run_simulation("dual_3d"),
        )
        self.btn_dual_3d.pack(side=tk.LEFT, padx=6)

        self.btn_dual_plot = ttk.Button(
            buttons_frame,
            text="Comparar gráficas",
            command=lambda: self._run_simulation("dual_plot"),
        )
        self.btn_dual_plot.pack(side=tk.LEFT, padx=6)

        self.btn_stop = ttk.Button(
            buttons_frame,
            text="Detener animación",
            command=self._stop_animation,
        )
        self.btn_stop.pack(side=tk.LEFT, padx=6)

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

        stats_frame = ttk.LabelFrame(self.root, text="Datos de métodos", padding=8)
        stats_frame.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(0, 8))

        self.stats_table = ttk.Treeview(
            stats_frame,
            columns=("metric", "newton", "gradiente"),
            show="headings",
            height=5,
        )
        self.stats_table.heading("metric", text="Métrica")
        self.stats_table.heading("newton", text="Newton-Raphson")
        self.stats_table.heading("gradiente", text="Gradiente (paso fijo)")
        self.stats_table.column("metric", width=220, anchor="w")
        self.stats_table.column("newton", width=180, anchor="center")
        self.stats_table.column("gradiente", width=180, anchor="center")
        self.stats_table.pack(side=tk.LEFT, fill=tk.X, expand=True)

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

        self.panel_poly = Poly3DCollection([], alpha=0.6, facecolor="#4A90E2")
        self.panel_poly_alt = Poly3DCollection([], alpha=0.4, facecolor="#7ED321")
        self.ax_panel.add_collection3d(self.panel_poly)
        self.ax_panel.add_collection3d(self.panel_poly_alt)
        self.sun_vec_line, = self.ax_panel.plot(
            [], [], [], color="#F5A623", linewidth=2, label="Vector del sol"
        )
        self.normal_vec_line, = self.ax_panel.plot(
            [], [], [], color="#4A90E2", linewidth=2, label="Normal (Newton)"
        )
        self.normal_vec_line_alt, = self.ax_panel.plot(
            [], [], [], color="#7ED321", linewidth=2, label="Normal (Gradiente)"
        )
        self.sun_sphere = self.ax_panel.scatter([], [], [], s=80, color="#F5A623", label="Sol")
        self.time_text = self.ax_panel.text2D(0.05, 0.92, "", transform=self.ax_panel.transAxes)

        self.roll_line, = self.ax_plot.plot([], [], label="roll (Newton)")
        self.pitch_line, = self.ax_plot.plot([], [], label="pitch (Newton)")
        self.roll_line_grad, = self.ax_plot.plot([], [], "--", label="roll (Gradiente)")
        self.pitch_line_grad, = self.ax_plot.plot([], [], "--", label="pitch (Gradiente)")
        self.elev_line, = self.ax_plot.plot([], [], label="elevación (sol)")

        self.ax_plot.legend(loc="upper right")

        self.ax_panel.legend(loc="upper right")

        canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = canvas

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
        self.sun_vec_line.set_data([], [])
        self.sun_vec_line.set_3d_properties([])
        self.normal_vec_line.set_data([], [])
        self.normal_vec_line.set_3d_properties([])
        self.panel_poly.set_verts([])
        self.panel_poly_alt.set_verts([])
        self.normal_vec_line_alt.set_data([], [])
        self.normal_vec_line_alt.set_3d_properties([])
        self.roll_line.set_data([], [])
        self.pitch_line.set_data([], [])
        self.roll_line_grad.set_data([], [])
        self.pitch_line_grad.set_data([], [])
        self.elev_line.set_data([], [])
        self.sun_sphere._offsets3d = ([], [], [])
        self.time_text.set_text("")
        self.canvas.draw_idle()

    def _parse_datetime(self, date_str: str, time_str: str) -> str:
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        return dt.isoformat()

    def _build_time_range(self) -> tuple[str, str]:
        start_date = self.start_date.get().strip()
        start_time = self.start_time.get().strip()
        end_date = self.end_date.get().strip()
        end_time = self.end_time.get().strip()

        if not start_date or not start_time:
            raise ValueError("Complete fecha y hora de inicio.")

        if not end_date or not end_time:
            end_date = start_date
            end_time = "18:00"

        inicio = self._parse_datetime(start_date, start_time)
        fin = self._parse_datetime(end_date, end_time)

        hora_inicio = datetime.strptime(start_time, "%H:%M").time()
        hora_fin = datetime.strptime(end_time, "%H:%M").time()
        limite_inicio = datetime.strptime("06:00", "%H:%M").time()
        limite_fin = datetime.strptime("18:00", "%H:%M").time()
        if hora_inicio < limite_inicio:
            raise ValueError("El horario debe iniciar desde las 06:00.")
        if hora_fin > limite_fin:
            raise ValueError("El horario debe terminar hasta las 18:00.")
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

        az = np.radians(self.df["azimut_deg"].to_numpy())
        el = np.radians(self.df["elevacion_deg"].to_numpy())

        phi_newton = self.df["phi_newton_rad"].to_numpy()
        beta_newton = self.df["beta_newton_rad"].to_numpy()
        phi_grad = self.df["phi_grad_rad"].to_numpy()
        beta_grad = self.df["beta_grad_rad"].to_numpy()

        if mode == "newton":
            phi = phi_newton
            beta = beta_newton
            method_label = "Método 1 (Newton)"
            plot_title = "Ángulos vs. posición solar - Newton"
        elif mode == "gradiente":
            phi = phi_grad
            beta = beta_grad
            method_label = "Método 2 (Gradiente paso fijo)"
            plot_title = "Ángulos vs. posición solar - Gradiente"
        elif mode == "dual_3d":
            phi = phi_newton
            beta = beta_newton
            method_label = "Comparación 3D (Newton vs Gradiente)"
            plot_title = "Ángulos vs. posición solar - Newton"
        else:
            phi = phi_newton
            beta = beta_newton
            method_label = "Comparación de gráficas (Newton vs Gradiente)"
            plot_title = "Ángulos vs. posición solar - Comparación"

        roll_deg = np.degrees(phi)
        pitch_deg = np.degrees(beta)
        elev_deg = np.degrees(el)

        self.ax_plot.set_title(plot_title)
        self.ax_plot.set_xlim(0, len(df))
        if mode == "dual_plot":
            roll_grad_deg = np.degrees(phi_grad)
            pitch_grad_deg = np.degrees(beta_grad)
            min_angle = min(
                np.min(roll_deg),
                np.min(pitch_deg),
                np.min(roll_grad_deg),
                np.min(pitch_grad_deg),
                np.min(elev_deg),
            )
            max_angle = max(
                np.max(roll_deg),
                np.max(pitch_deg),
                np.max(roll_grad_deg),
                np.max(pitch_grad_deg),
                np.max(elev_deg),
            )
        else:
            min_angle = min(np.min(roll_deg), np.min(pitch_deg), np.min(elev_deg))
            max_angle = max(np.max(roll_deg), np.max(pitch_deg), np.max(elev_deg))
        pad = max(5, (max_angle - min_angle) * 0.1)
        self.ax_plot.set_ylim(min_angle - pad, max_angle + pad)

        def update(i: int):
            s_vec = vector_incidencia_desde_az_el(az[i], el[i])
            verts = _panel_vertices(phi[i], beta[i])
            self.panel_poly.set_verts([verts])
            if mode == "dual_3d":
                verts_alt = _panel_vertices(phi_grad[i], beta_grad[i])
                self.panel_poly_alt.set_verts([verts_alt])
            else:
                self.panel_poly_alt.set_verts([])

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
            if mode == "dual_3d":
                n_vec_alt = normal_panel(phi_grad[i], beta_grad[i])
                self.normal_vec_line_alt.set_data(
                    [0, scale * n_vec_alt[0]], [0, scale * n_vec_alt[1]]
                )
                self.normal_vec_line_alt.set_3d_properties([0, scale * n_vec_alt[2]])
            else:
                self.normal_vec_line_alt.set_data([], [])
                self.normal_vec_line_alt.set_3d_properties([])

            x = np.arange(i + 1)
            self.roll_line.set_data(x, roll_deg[: i + 1])
            self.pitch_line.set_data(x, pitch_deg[: i + 1])
            if mode == "dual_plot":
                self.roll_line_grad.set_data(x, roll_grad_deg[: i + 1])
                self.pitch_line_grad.set_data(x, pitch_grad_deg[: i + 1])
            else:
                self.roll_line_grad.set_data([], [])
                self.pitch_line_grad.set_data([], [])
            self.elev_line.set_data(x, elev_deg[: i + 1])

            tiempo = self.df.loc[i, "tiempo"]
            self.time_text.set_text(f"{method_label}\n{tiempo}")

            return (
                self.panel_poly,
                self.panel_poly_alt,
                self.sun_vec_line,
                self.normal_vec_line,
                self.normal_vec_line_alt,
                self.sun_sphere,
                self.roll_line,
                self.pitch_line,
                self.roll_line_grad,
                self.pitch_line_grad,
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
