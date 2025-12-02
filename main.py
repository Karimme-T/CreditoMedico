#Librerías
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
import pdfplumber
import re
import io


# Carga del modelo y preprocesamiento

model = keras.models.load_model("modelo_credito_medico.h5")

model = keras.models.load_model("modelo_credito_medico.h5")
preprocess = joblib.load("prepro_credito_medico.pkl")


imputer_num = preprocess["imputer_num"]
imputer_cat = preprocess["imputer_cat"]
label_encoders = preprocess["label_encoders"]
scaler = preprocess["scaler"]
categorical_cols = preprocess["categorical_cols"]
numerical_cols = preprocess["numerical_cols"]
le_target = preprocess["le_target"]

if "feature_order" in preprocess:
    feature_order = preprocess["feature_order"]
    all_feature_cols = feature_order  # Usar el orden guardado
    print(f"\n✅ Usando orden de columnas guardado del entrenamiento")
else:
    all_feature_cols = numerical_cols + categorical_cols  # Fallback
    print(f"\n⚠️ No se encontró orden guardado, usando numerical + categorical")

print(f"Orden de columnas: {all_feature_cols}")

all_feature_cols = numerical_cols + categorical_cols

print("=" * 60)
print("CONFIGURACIÓN DEL MODELO CARGADA")
print("=" * 60)
print(f"Columnas numéricas ({len(numerical_cols)}):")
print(numerical_cols)
print(f"\nColumnas categóricas ({len(categorical_cols)}):")
print(categorical_cols)
print(f"\nTotal features esperadas: {len(all_feature_cols)}")
print("=" * 60)

# Configuración de fastapi

app = FastAPI(title="API Crédito Médico", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# MODELOS Pydantic

class Solicitud(BaseModel):
    Annual_Income: float
    Monthly_Inhand_Salary: float
    Num_Bank_Accounts: int
    Num_Credit_Card: int
    Interest_Rate: float
    Num_of_Loan: int
    Type_of_Loan: str
    Delay_from_due_date: int
    Num_of_Delayed_Payment: int
    Changed_Credit_Limit: float
    Credit_Mix: str
    Outstanding_Debt: float
    Credit_Utilization_Ratio: float
    Credit_History_Age: float   # en meses
    Payment_of_Min_Amount: str  # "Yes", "No", etc.
    Total_EMI_per_month: float
    Amount_invested_monthly: float
    Payment_Behaviour: str
    Monthly_Balance: float


@app.get("/")
def root():
    return {"mensaje": "API de scoring de tarjeta médica funcionando 🚑"}


# Funciones auxiliares pdf

def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extrae texto de un PDF a partir de bytes usando pdfplumber."""
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            text += "\n" + txt
    return text


# PDF "MI SCORE" 

def parse_buro_score_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Parser del PDF tipo 'MI SCORE'.
    Busca el número grande (ej. 652) y el segmento (REGULAR/BUENO/MALO).
    """

    text = _extract_text_from_pdf_bytes(pdf_bytes)

    # Intento 1: número + palabra (652 REGULAR, 720 BUENO, etc.)
    m = re.search(r"\b([4-8]\d{2})\b\s*(REGULAR|BUENO|MALO)", text, re.IGNORECASE)
    if m:
        score = int(m.group(1))
        segmento = m.group(2).upper()
    else:
        # Intento 2: buscar cualquier número 400–850 después de "MI SCORE" o "SCORE"
        score = None
        segmento = None
        m2 = re.search(r"(MI\s+SCORE|SCORE)[^\d]*([4-8]\d{2})", text, re.IGNORECASE)
        if m2:
            score = int(m2.group(2))

        # si aún no, tomamos el "más probable" entre 400 y 850
        if score is None:
            candidatos = re.findall(r"\b([4-8]\d{2})\b", text)
            candidatos_int = [int(c) for c in candidatos]
            if candidatos_int:
                # heurística: tomar el que esté más cerca del promedio
                promedio = sum(candidatos_int) / len(candidatos_int)
                score = min(candidatos_int, key=lambda x: abs(x - promedio))

        # segmento aproximado por rangos
        if score is not None:
            if score >= 690:
                segmento = "BUENO"
            elif score >= 620:
                segmento = "REGULAR"
            else:
                segmento = "MALO"

    return {
        "buro_score_raw": score,
        "buro_segmento": segmento,  # BUENO / REGULAR / MALO
    }


# 2) PDF "Reporte de Crédito Especial"

def parse_buro_detalle_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Parser del 'Reporte de Crédito Especial'
    Extrae info de:
    - DOMICILIOS DE EMPLEO -> salario
    - RESUMEN DE CRÉDITOS -> tarjetas, créditos, atrasos, deuda
    """

    text = _extract_text_from_pdf_bytes(pdf_bytes)
    lines = text.splitlines()

    # Salarios en 'DOMICILIOS DE EMPLEO' 
    # Se busca la palabra 'Salario' y luego números en las líneas siguientes.
    salarios = []
    for i, line in enumerate(lines):
        if re.search(r"Salario", line, re.IGNORECASE):
            # buscar números en esta línea y en las 3 siguientes
            joined = " ".join(lines[i : i + 4])
            matches = re.findall(r"(\d[\d,\.]*)", joined)
            for m in matches:
                try:
                    val = float(m.replace(",", ""))
                    if val > 500:  # filtración decosas pequeñas tipo códigos
                        salarios.append(val)
                except Exception:
                    continue

    if salarios:
        salary_from_buro = float(np.mean(salarios))  # promedio salario de empleos
    else:
        salary_from_buro = 0.0

    # Contar tarjetas de crédito y otros créditos 
    num_credit_cards = len(
        re.findall(r"TARJETA\s+DE\s+CR[ÉE]DITO", text, re.IGNORECASE)
    )

    # créditos no bancarios 
    num_loans_non_bank = len(
        re.findall(
            r"CR[ÉE]DITOS\s+NO\s+BANCARIOS|MUEBLER[IÍ]A|APARATOS/MUEBLES",
            text,
            re.IGNORECASE,
        )
    )

    if num_credit_cards == 0:
        num_credit_cards = 1  # se asume al menos 1 si va a pedir la tarjeta

    num_of_loan = max(num_loans_non_bank, 0)

    # Atrasos 
    # En la leyenda aparecen textos como:
    # "CUENTA AL CORRIENTE", "ATRASO DE 1 A 89 DIAS",
    # "ATRASO MAYOR A 90 DIAS O DEUDA SIN RECUPERAR".
    num_delayed = 0
    delay_from_due_date = 0

    if re.search(r"ATRASO\s+DE\s+1\s+A\s+89\s+DIAS", text, re.IGNORECASE):
        num_delayed += 1
        delay_from_due_date = max(delay_from_due_date, 60)

    if re.search(
        r"ATRASO\s+MAYOR\s+A\s+90\s+DIAS|DEUDA\s+SIN\s+RECUPERAR",
        text,
        re.IGNORECASE,
    ):
        num_delayed += 2
        delay_from_due_date = max(delay_from_due_date, 90)

    # Deuda ('Saldo actual')
    saldo_matches = re.findall(
        r"Saldo\s+actual\s*[:\-]?\s*\$?\s*([\d,]+\.\d+|\d+)",
        text,
        re.IGNORECASE,
    )
    outstanding_debt = 0.0
    for m in saldo_matches:
        try:
            value = float(m.replace(",", ""))
            outstanding_debt += value
        except Exception:
            continue

    # Utilización de crédito
    # Si no hay límite, se asume 30% de utilización.
    if outstanding_debt > 0:
        credit_util_ratio = 30.0
    else:
        credit_util_ratio = 0.0

    # Antigüedad del historial
    # Si el PDF no trae campo directo, se deja 12 meses por defecto.
    credit_history_age = 12.0

    # Behaviour / Payment_of_Min_Amount --------
    if num_delayed == 0:
        payment_min_amount = "Yes"
        payment_behaviour = "High_spent_Small_value_payments"
    else:
        payment_min_amount = "No"
        payment_behaviour = "Low_spent_Large_value_payments"

    # Num_Bank_Accounts 
    # aproximamos como (tarjetas + créditos no bancarios)
    num_bank_accounts = max(1, num_credit_cards + num_of_loan)

    # Type_of_Loan 
    if num_of_loan > 0:
        type_of_loan = "Credit-Card"
    else:
        type_of_loan = "Not Specified"

    # Credit_Mix: heurística
    if num_delayed == 0 and outstanding_debt > 0:
        credit_mix = "Good"
    elif num_delayed > 0:
        credit_mix = "Bad"
    else:
        credit_mix = "Standard"

    return {
        "Salary_from_buro": salary_from_buro,
        "Num_Credit_Card": num_credit_cards,
        "Num_of_Loan": num_of_loan,
        "Num_of_Delayed_Payment": num_delayed,
        "Delay_from_due_date": delay_from_due_date,
        "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": credit_util_ratio,
        "Credit_History_Age": credit_history_age,
        "Payment_of_Min_Amount": payment_min_amount,
        "Payment_Behaviour": payment_behaviour,
        "Num_Bank_Accounts": num_bank_accounts,
        "Type_of_Loan": type_of_loan,
        "Credit_Mix": credit_mix,
    }


# PDF Estado de Cuenta (opcional)

def parse_estado_cuenta_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Parser del estado de cuenta bancario
    Busca:
    - saldo promedio
    - depósitos mensuales
    - tipo de cuenta
    """

    text = _extract_text_from_pdf_bytes(pdf_bytes)

    # Buscar saldo 
    saldo_match = re.search(
        r"Saldo\s+(final|promedio|actual)\s*:?\s*\$?\s*([\d,]+\.\d+|\d+)",
        text,
        re.IGNORECASE,
    )
    monthly_balance = 0.0
    if saldo_match:
        try:
            monthly_balance = float(saldo_match.group(2).replace(",", ""))
        except Exception:
            pass

    # Buscar ingresos
    monthly_salary = 0.0
    ingreso_match = re.search(
        r"(Ingreso|Depósito|Abono)\s+(mensual|total)?\s*:?\s*\$?\s*([\d,]+\.\d+|\d+)",
        text,
        re.IGNORECASE,
    )
    if ingreso_match:
        try:
            monthly_salary = float(ingreso_match.group(3).replace(",", ""))
        except Exception:
            pass

    return {
        "Monthly_Balance": monthly_balance,
        "Monthly_Inhand_Salary": monthly_salary,
    }


# PREPROCESAMIENTO 
def preprocesar_solicitud(s: Solicitud) -> np.ndarray:
    """Aplica el mismo preprocesamiento que en el entrenamiento y devuelve X_scaled."""

    data_dict = {
        "Annual_Income": s.Annual_Income,
        "Monthly_Inhand_Salary": s.Monthly_Inhand_Salary,
        "Num_Bank_Accounts": s.Num_Bank_Accounts,
        "Num_Credit_Card": s.Num_Credit_Card,
        "Interest_Rate": s.Interest_Rate,
        "Num_of_Loan": s.Num_of_Loan,
        "Type_of_Loan": s.Type_of_Loan,
        "Delay_from_due_date": s.Delay_from_due_date,
        "Num_of_Delayed_Payment": s.Num_of_Delayed_Payment,
        "Changed_Credit_Limit": s.Changed_Credit_Limit,
        "Credit_Mix": s.Credit_Mix,
        "Outstanding_Debt": s.Outstanding_Debt,
        "Credit_Utilization_Ratio": s.Credit_Utilization_Ratio,
        "Credit_History_Age": s.Credit_History_Age,
        "Payment_of_Min_Amount": s.Payment_of_Min_Amount,
        "Total_EMI_per_month": s.Total_EMI_per_month,
        "Amount_invested_monthly": s.Amount_invested_monthly,
        "Payment_Behaviour": s.Payment_Behaviour,
        "Monthly_Balance": s.Monthly_Balance,
        "Num_Credit_Inquiries": 0.0  
    }

    df = pd.DataFrame([data_dict])

    # Mapear Payment_of_Min_Amount a número
    def map_payment(x):
        if x is None or pd.isna(x):
            return -1
        t = str(x).lower().strip()
        if t in ["yes", "y", "1", "si", "sí"]:
            return 1
        elif t in ["no", "n", "0"]:
            return 0
        else:
            return -1
    
    df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].apply(map_payment)

    # Codificar variables categóricas
    for col in categorical_cols:
        if col in label_encoders:
            le = label_encoders[col]
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    # Asegurar que todas las columnas numéricas sean float
    for col in all_feature_cols:
        if col not in categorical_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Reordenar según el orden exacto del entrenamiento
    df_ordered = df[all_feature_cols]
    
    # Escalar (sin pasar por imputers)
    X_scaled = scaler.transform(df_ordered.values)
    
    return X_scaled

# ENDPOINT 
@app.post("/predict")
async def predict(
    # TEXTO (Nombres EXACTOS como en React)
    nombre: str = Form(...),
    direccion: str = Form(...),
    correo: str = Form(...),
    telefono: str = Form(...),
    rfc: str = Form(...),
    ingresoMensual: float = Form(...),   # React lo manda como texto, FastAPI lo convierte a float
    ingresoAnual: float = Form(...),
    inversionMensual: float = Form(...),

    # ARCHIVOS (Nombres EXACTOS como en React)
    pdfBuro: UploadFile = File(...),
    pdfDetallado: UploadFile = File(...),
    ine: UploadFile = File(...),              
    comprobanteDomicilio: UploadFile = File(...),
    estadoCuenta: Optional[UploadFile] = File(None)
):
    print(f"--- Procesando solicitud de: {nombre} ---")
    
    try:
        # LEER Y PARSEAR LOS PDFS DEL BURÓ
        # Usamos los archivos que llegaron de React
        score_bytes = await pdfBuro.read()
        detalle_bytes = await pdfDetallado.read()

        buro_score_info = parse_buro_score_pdf(score_bytes)
        buro_detalle_info = parse_buro_detalle_pdf(detalle_bytes)

        # PROCESAR ESTADO DE CUENTA (Si existe)
        estado_features = {}
        if estadoCuenta is not None:
            estado_bytes = await estadoCuenta.read()
            estado_features = parse_estado_cuenta_pdf(estado_bytes)

        # DEFINIR VARIABLES FINANCIERAS
        monthly_salary = float(ingresoMensual)
        annual_income = float(ingresoAnual)
        amount_invested_monthly = float(inversionMensual)
        
        # Monthly_Balance (Calculado del estado de cuenta o default)
        if "Monthly_Balance" in estado_features:
            monthly_balance = float(estado_features["Monthly_Balance"])
        else:
            monthly_balance = 0.0


        outstanding_debt = float(buro_detalle_info.get("Outstanding_Debt", 0.0))
        total_emi = outstanding_debt * 0.05  # Regla del 5%

        # Credit Mix
        segmento = buro_score_info.get("buro_segmento")
        if segmento == "BUENO":
            credit_mix = "Good"
        elif segmento == "REGULAR":
            credit_mix = "Standard"
        elif segmento == "MALO":
            credit_mix = "Bad"
        else:
            credit_mix = str(buro_detalle_info.get("Credit_Mix", "Unknown"))

        # CONSTRUIR EL OBJETO SOLICITUD
        # Mapeamos todos los datos recolectados 
        solicitud_data = {
            "Annual_Income": annual_income,
            "Monthly_Inhand_Salary": monthly_salary,
            "Num_Bank_Accounts": int(buro_detalle_info.get("Num_Bank_Accounts", 1)),
            "Num_Credit_Card": int(buro_detalle_info.get("Num_Credit_Card", 1)),
            "Interest_Rate": 25.0,
            "Num_of_Loan": int(buro_detalle_info.get("Num_of_Loan", 1)),
            "Type_of_Loan": "Credit-Card",
            "Delay_from_due_date": int(buro_detalle_info.get("Delay_from_due_date", 0)),
            "Num_of_Delayed_Payment": int(buro_detalle_info.get("Num_of_Delayed_Payment", 0)),
            "Changed_Credit_Limit": 0.0,
            "Credit_Mix": credit_mix,
            "Outstanding_Debt": outstanding_debt,
            "Credit_Utilization_Ratio": float(buro_detalle_info.get("Credit_Utilization_Ratio", 0.0)),
            "Credit_History_Age": float(buro_detalle_info.get("Credit_History_Age", 36.0)),
            "Payment_of_Min_Amount": str(buro_detalle_info.get("Payment_of_Min_Amount", "Yes")),
            "Total_EMI_per_month": total_emi,
            "Amount_invested_monthly": amount_invested_monthly,
            "Payment_Behaviour": str(buro_detalle_info.get("Payment_Behaviour", "Unknown")),
            "Monthly_Balance": monthly_balance,
        }

        solicitud_obj = Solicitud(**solicitud_data)

        # PREDECIR CON EL MODELO ML
        X_scaled = preprocesar_solicitud(solicitud_obj) 
        proba = model.predict(X_scaled)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = le_target.inverse_transform([pred_idx])[0]

        #CALCULAR LÍNEA DE CRÉDITO
        class_probs = {
            clase: float(prob)
            for clase, prob in zip(le_target.classes_, proba)
        }

        label_lower = str(pred_label).lower()
        if "good" in label_lower:
            linea = 60000
        elif "standard" in label_lower:
            linea = 20000
        else:
            linea = 5000

        # RETORNAR RESULTADO A REACT
        return {
            "mensaje": "Análisis completado exitosamente",
            "monto": linea,             
            "credit_score_predicho": pred_label,
            "probabilidades": class_probs,
            "usuario": nombre
        }

    except Exception as e:
        print(f"\n[ERROR CRÍTICO] en /predict: {str(e)}")
        import traceback
        traceback.print_exc() 
        return {
            "error": str(e),
            "mensaje": "Error interno al procesar la solicitud"
        }
