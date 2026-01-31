import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generar_graficas(ruta_csv: str, carpeta_salida: str):
    os.makedirs(carpeta_salida, exist_ok=True)
    df = pd.read_csv(ruta_csv)
    t = range(len(df))

    # Angulos (deg)
    plt.figure()
    plt.plot(t, np.degrees(df["phi_newton_rad"]), label="roll (Newton)")
    plt.plot(t, np.degrees(df["beta_newton_rad"]), label="pitch (Newton)")
    plt.plot(t, np.degrees(df["phi_grad_rad"]), "--", label="roll (Gradiente)")
    plt.plot(t, np.degrees(df["beta_grad_rad"]), "--", label="pitch (Gradiente)")
    plt.xlabel("Paso de tiempo")
    plt.ylabel("Ángulo (deg)")
    plt.title("Ángulos del panel")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_salida, "angulos.png"), dpi=150)
    plt.close()

    # Error de incidencia
    plt.figure()
    plt.plot(t, df["error_newton_deg"], label="Error (Newton)")
    plt.plot(t, df["error_grad_deg"], label="Error (Gradiente)")
    plt.xlabel("Paso de tiempo")
    plt.ylabel("Ángulo de incidencia (deg)")
    plt.title("Diagnóstico: ángulo entre normal del panel y luz")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_salida, "error_incidencia.png"), dpi=150)
    plt.close()