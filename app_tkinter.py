"""
App Tkinter - Predictor Sector Escuela de Talentos
UPB - Analitica de Datos - Practica 4
Ejecutar:  python app_tkinter.py
"""
import joblib
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

# Cargar artefactos
artefactos = joblib.load("modelo_final.joblib")
modelo            = artefactos["modelo"]
scaler            = artefactos["scaler"]
label_encoder     = artefactos["label_encoder"]
feature_columns   = artefactos["feature_columns"]
requires_scaling  = artefactos["requires_scaling"]
modelo_nombre     = artefactos["modelo_nombre"]

GRUPOS_ETARIOS = [
    "Primera Infancia 0-5", "Infancia 6-11", "Adolescencia 12-18",
    "Juventud 19-26", "Adultez 27-59",
]
VIGENCIAS = [2022, 2023, 2024]


def predecir(vigencia, grupo_etario, inversion):
    fila = pd.DataFrame([{c: 0 for c in feature_columns}])
    fila["VIGENCIA"] = vigencia
    fila["Inversion_por_deportista"] = inversion
    col = f"Grupo_Etario_{grupo_etario.replace(' ', '_').replace('-', '_')}"
    if col not in fila.columns:
        raise ValueError(f"Grupo etario invalido: {grupo_etario}")
    fila[col] = 1
    fila = fila[feature_columns]
    X = scaler.transform(fila) if requires_scaling else fila
    idx = modelo.predict(X)[0]
    clase = label_encoder.inverse_transform([idx])[0]
    proba = modelo.predict_proba(X)[0][int(label_encoder.transform(["PARALIMPICO"])[0])]
    return clase, float(proba)


class PredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Predictor Sector - Escuela de Talentos")
        self.root.geometry("520x520")
        self.root.configure(bg="#F5F5F5")

        style = ttk.Style(); style.theme_use("clam")
        style.configure("TLabel", background="#F5F5F5", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"),
                       background="#2E75B6", foreground="white")

        ttk.Label(root, text="  Predictor de Sector Deportivo",
                  style="Header.TLabel", anchor="w").pack(fill="x", ipady=12)
        ttk.Label(root,
                  text=f"Modelo: {modelo_nombre}  |  Clases: OLIMPICO / PARALIMPICO",
                  font=("Segoe UI", 9, "italic"), background="#F5F5F5",
                  foreground="#666").pack(pady=(8, 16))

        frame = ttk.Frame(root, padding=20); frame.pack(fill="x", padx=20)

        ttk.Label(frame, text="Vigencia (anio):").grid(row=0, column=0, sticky="w", pady=8)
        self.vig_var = tk.StringVar(value=str(VIGENCIAS[-1]))
        ttk.Combobox(frame, textvariable=self.vig_var, values=[str(v) for v in VIGENCIAS],
                     state="readonly", width=25).grid(row=0, column=1, pady=8)

        ttk.Label(frame, text="Grupo etario:").grid(row=1, column=0, sticky="w", pady=8)
        self.gru_var = tk.StringVar(value=GRUPOS_ETARIOS[2])
        ttk.Combobox(frame, textvariable=self.gru_var, values=GRUPOS_ETARIOS,
                     state="readonly", width=25).grid(row=1, column=1, pady=8)

        ttk.Label(frame, text="Inversion por deportista (COP):").grid(row=2, column=0,
                                                                       sticky="w", pady=8)
        self.inv_var = tk.StringVar(value="3310519.10")
        ttk.Entry(frame, textvariable=self.inv_var, width=27).grid(row=2, column=1, pady=8)

        btn = ttk.Frame(root, padding=10); btn.pack(pady=10)
        ttk.Button(btn, text="Predecir",
                   command=self.predecir).pack(side="left", padx=8, ipadx=14, ipady=4)
        ttk.Button(btn, text="Limpiar",
                   command=self.limpiar).pack(side="left", padx=8, ipadx=14, ipady=4)

        self.res = ttk.Label(root, text="", font=("Segoe UI", 13, "bold"),
                             background="#F5F5F5"); self.res.pack(pady=12)
        self.pro = ttk.Label(root, text="", font=("Segoe UI", 10),
                             background="#F5F5F5", foreground="#444"); self.pro.pack()
        ttk.Label(root, text="UPB - Analitica de Datos - Practica 4",
                  font=("Segoe UI", 8), background="#F5F5F5",
                  foreground="#888").pack(side="bottom", pady=8)

    def predecir(self):
        try:
            vig = int(self.vig_var.get())
            gru = self.gru_var.get()
            inv = float(self.inv_var.get())
            if inv < 0:
                raise ValueError("La inversion no puede ser negativa")
            clase, p = predecir(vig, gru, inv)
            color = "#2E75B6" if clase == "OLIMPICO" else "#E97132"
            self.res.config(text=f"Prediccion: {clase}", foreground=color)
            self.pro.config(
                text=f"P(PARALIMPICO) = {p:.4f}   |   P(OLIMPICO) = {1-p:.4f}")
        except ValueError as e:
            messagebox.showerror("Error de entrada", f"Verifica los datos:\n{e}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def limpiar(self):
        self.vig_var.set(str(VIGENCIAS[-1]))
        self.gru_var.set(GRUPOS_ETARIOS[2])
        self.inv_var.set("3310519.10")
        self.res.config(text=""); self.pro.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    PredictorApp(root)
    root.mainloop()
