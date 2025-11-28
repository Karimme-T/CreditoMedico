#%%
# Carga de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras import regularizers
from keras.constraints import MaxNorm
  
#%%
# Cargar el dataset

df = pd.read_csv('Data.csv')
print(f"\nDimensiones del dataset: {df.shape}")
print(f"Número de filas: {df.shape[0]}")
print(f"Número de columnas: {df.shape[1]}")

print("\nPrimeras 5 filas")
print(df.head())

print("\nInformación del dataset")
print(df.info())

print("\nEstadísticas descriptivas")
print(df.describe())

print("\nDistribución de la variable objetivo (Credit_Score)")
print(df['Credit_Score'].value_counts())
print("\nProporción:")
print(df['Credit_Score'].value_counts(normalize=True))

#%%
# Visualizar distribución de la variable objetivo
plt.figure(figsize=(10, 6))
df['Credit_Score'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribución de Credit Score', fontsize=16, fontweight='bold')
plt.xlabel('Credit Score', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Identificar tipos de datos
print("\nTipos de datos por columna")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nColumnas categóricas ({len(categorical_cols)}): {categorical_cols}")
print(f"\nColumnas numéricas ({len(numerical_cols)}): {numerical_cols}")
# %%
# Crear copia del dataframe
df_processed = df.copy()

if 'Age' in df_processed.columns:
    edad_min = 18
    edad_max = 85
    
    df_processed['Age'] = pd.to_numeric(df_processed['Age'], errors='coerce')
    df_processed = df_processed[(df_processed['Age'] >= edad_min) & (df_processed['Age'] <= edad_max)]

#%%
#Limpieza de columnas numéricas
valores_invalidos = [
    '_', '__', '___', '____', '_____', '______', '_______',
    '-', '--', '---', '----', '-----', '------',
    '!@9#%8', '#F%$D@*&8', '!@9#%8', 
    '__10000__', '__-333333333333333333333333333__'
]

# Reemplazar en todo el dataframe
df_clean = df_processed.replace(valores_invalidos, np.nan)

# Limpiar espacios extras y valores vacíos en columnas de texto
for col in df_clean.select_dtypes(include=['object']).columns:
    df_clean[col] = df_clean[col].astype(str).str.strip()
    df_clean[col] = df_clean[col].replace(['', ' ', 'nan', 'None', 'null', 'NaN'], np.nan)

print("Valores especiales reemplazados por NaN")

# Limpieza de credit history age

if 'Credit_History_Age' in df_clean.columns:
    import re
    
    def age_credito_a_meses(s: pd.Series):
        """
        Convierte "22 Years and 1 Months" a número de meses
        Convierte "NA" a NaN
        """
        def parse(x):
            if pd.isna(x):
                return np.nan
            t = str(x).strip().lower()
            if t in {'na', 'nan', '', 'none'}:
                return np.nan
            
            y = re.search(r'(\d+)\s*year', t)
            m = re.search(r'(\d+)\s*month', t)
            yy = int(y.group(1)) if y else 0
            mm = int(m.group(1)) if m else 0

            return yy * 12 + mm
        return s.apply(parse).astype('float')
    
    df_clean['Credit_History_Age'] = age_credito_a_meses(df_clean['Credit_History_Age'])

numeric_cols_expected = [
    'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
    'Num_Credit_Card', 'Num_of_Loan', 'Interest_Rate', 'Num_of_Delayed_Payment',
    'Changed_Credit_Limit', 'Outstanding_Debt', 'Amount_invested_monthly',
    'Monthly_Balance', 'Delay_from_due_date', 'Num_Credit_Inquiries',
    'Credit_Utilization_Ratio', 'Total_EMI_per_month', 'Credit_History_Age'
]

for col in numeric_cols_expected:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')


#dado que hay varios registros del mismo usuario, tomeré el último que se hizo basándome en la columna del mes
df_clean = df_clean.drop_duplicates(subset='Customer_ID', keep='first')
# Eliminación de columnas innecesarias
cols_eliminar = ['ID','Customer_ID', 'Month', 'Name', 'SSN', 'Age', 'Name', 'Occupation']
df_clean = df_clean.drop(columns=cols_eliminar)

# Se eliminan filas de usuarios que tienen demasiados valores nulos en su información
null_percentage_per_row = df_clean.isnull().sum(axis=1) / df_clean.shape[1] * 100
threshold = 50
rows_to_drop = null_percentage_per_row > threshold
rows_dropped = rows_to_drop.sum()

if rows_dropped > 0:
    df_clean = df_clean[~rows_to_drop]

#%%

#Separación de X y Y
X = df_clean.drop('Credit_Score', axis=1)
y = df_clean['Credit_Score']

print(f"X: {X.shape}")
print(f"y: {y.shape}")

#%%
# Identificar tipos de columnas
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nColumnas categóricas ({len(categorical_cols)}): {categorical_cols}")
print(f"Columnas numéricas ({len(numerical_cols)}): {len(numerical_cols)} columnas")

#%%
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

print(f"Clases: {le_target.classes_}")
print(f"Mapeo: {dict(zip(le_target.classes_, range(len(le_target.classes_))))}")

#%%

#SPLIT de datos

# Primera división: 80% temporal, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

# Segunda división: 75% train, 25% val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, 
    test_size=0.25, 
    random_state=42, 
    stratify=y_temp
)

print(f"Train: {X_train.shape[0]} muestras ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Val:   {X_val.shape[0]} muestras ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test:  {X_test.shape[0]} muestras ({X_test.shape[0]/len(X)*100:.1f}%)")

print("\nDistribución de clases:")
print(f"Train: {np.bincount(y_train)}")
print(f"Val:   {np.bincount(y_val)}")
print(f"Test:  {np.bincount(y_test)}")

#%%
#Manejo de datos muy extremos
numeric_ranges = {
    'Annual_Income': (1000, 1000000),
    'Monthly_Inhand_Salary': (0, 100000),
    'Num_Bank_Accounts': (0, 15),
    'Num_Credit_Card': (0, 20),
    'Num_of_Loan': (0, 15),
    'Interest_Rate': (0, 40),
    'Num_of_Delayed_Payment': (0, 100),
    'Changed_Credit_Limit': (-50000, 100000),
    'Outstanding_Debt': (0, 500000),
    'Amount_invested_monthly': (0, 100000),
    'Monthly_Balance': (-20000, 200000),
    'Delay_from_due_date': (0, 90),
    'Num_Credit_Inquiries': (0, 30),
    'Credit_Utilization_Ratio': (0, 100),
    'Total_EMI_per_month': (0, 100000),
    'Credit_History_Age': (0, 600)
}

# Aplicar rangos y marcar outliers como NaN
for col, (min_val, max_val) in numeric_ranges.items():
    if col in numerical_cols:
        # Train
        out_train = ((X_train[col] < min_val) | (X_train[col] > max_val)).sum()
        X_train.loc[(X_train[col] < min_val) | (X_train[col] > max_val), col] = np.nan
        
        # Val
        out_val = ((X_val[col] < min_val) | (X_val[col] > max_val)).sum()
        X_val.loc[(X_val[col] < min_val) | (X_val[col] > max_val), col] = np.nan
        
        # Test
        out_test = ((X_test[col] < min_val) | (X_test[col] > max_val)).sum()
        X_test.loc[(X_test[col] < min_val) | (X_test[col] > max_val), col] = np.nan
        
        if out_train + out_val + out_test > 0:
            print(f"  {col}: {out_train + out_val + out_test} outliers marcados como NaN")


#%%
# para los valores nulos se va a tomar la mediana y la moda 
print("\nValores nulos antes de tratar los datos:")
print(f"Train: {X_train.isnull().sum().sum()}")
print(f"Val:   {X_val.isnull().sum().sum()}")
print(f"Test:  {X_test.isnull().sum().sum()}")

# Imputación numérica - FIT en train
if len(numerical_cols) > 0:
    imputer_num = SimpleImputer(strategy='median')
    imputer_num.fit(X_train[numerical_cols])
    
    X_train[numerical_cols] = imputer_num.transform(X_train[numerical_cols])
    X_val[numerical_cols] = imputer_num.transform(X_val[numerical_cols])
    X_test[numerical_cols] = imputer_num.transform(X_test[numerical_cols])
    print("✓ Valores numéricos imputados con mediana (fit en train)")

# Imputación categórica - FIT en train
if len(categorical_cols) > 0:
    imputer_cat = SimpleImputer(strategy='most_frequent')
    imputer_cat.fit(X_train[categorical_cols])
    
    X_train[categorical_cols] = imputer_cat.transform(X_train[categorical_cols])
    X_val[categorical_cols] = imputer_cat.transform(X_val[categorical_cols])
    X_test[categorical_cols] = imputer_cat.transform(X_test[categorical_cols])
    print("✓ Valores categóricos imputados con moda (fit en train)")

print("\nValores nulos DESPUÉS de imputación:")
print(f"Train: {X_train.isnull().sum().sum()}")
print(f"Val:   {X_val.isnull().sum().sum()}")
print(f"Test:  {X_test.isnull().sum().sum()}")
# %%
# Mapeo de payment of min amount
# Payment_of_Min_Amount: convertir Yes/No a 1/0 - FIT en train
if 'Payment_of_Min_Amount' in categorical_cols:
    # Crear mapeo basado SOLO en train
    unique_train = X_train['Payment_of_Min_Amount'].unique()
    print(f"  Payment_of_Min_Amount - valores únicos en train: {unique_train}")
    
    payment_mapping = {}
    for val in unique_train:
        val_lower = str(val).lower().strip()
        if val_lower in ['yes', 'y', '1']:
            payment_mapping[val] = 1
        elif val_lower in ['no', 'n', '0']:
            payment_mapping[val] = 0
        else:
            payment_mapping[val] = -1  # Desconocido
    
    # Aplicar mapeo
    X_train['Payment_of_Min_Amount'] = X_train['Payment_of_Min_Amount'].map(payment_mapping).fillna(-1)
    X_val['Payment_of_Min_Amount'] = X_val['Payment_of_Min_Amount'].map(payment_mapping).fillna(-1)
    X_test['Payment_of_Min_Amount'] = X_test['Payment_of_Min_Amount'].map(payment_mapping).fillna(-1)
    
    # Actualizar lista de columnas (ya no es categórica)
    categorical_cols.remove('Payment_of_Min_Amount')
    numerical_cols.append('Payment_of_Min_Amount')
    
    print(f"Payment_of_Min_Amount mapeado (1=Yes, 0=No, -1=Desconocido)")

# %%
# Label encoding para variables categóricas
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    
    # FIT solo en train
    le.fit(X_train[col].astype(str))
    
    # TRANSFORM en todos
    X_train[col] = le.transform(X_train[col].astype(str))
    
    # Para val y test: manejar categorías no vistas
    X_val[col] = X_val[col].astype(str).apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )
    X_test[col] = X_test[col].astype(str).apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )
    
    label_encoders[col] = le
    print(f"✓ '{col}' codificado ({len(le.classes_)} categorías)")

#%%
#Normalización
X_train_array = X_train.values
X_val_array = X_val.values
X_test_array = X_test.values

# StandardScaler - FIT en train
scaler = StandardScaler()
scaler.fit(X_train_array)

X_train_scaled = scaler.transform(X_train_array)
X_val_scaled = scaler.transform(X_val_array)
X_test_scaled = scaler.transform(X_test_array)

print("Datos normalizados (fit en train)")
print(f"  Media X_train: {X_train_scaled.mean():.6f}")
print(f"  Std X_train: {X_train_scaled.std():.6f}")

#%%
#Construcción del modelo de redes
n_features = X_train_scaled.shape[1]
n_classes = len(np.unique(y_encoded))

print(f"Features: {n_features}")
print(f"Clases: {n_classes}")

model = Sequential([
    Dense(128, activation='relu', input_shape=(n_features,),
          kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='relu',
          kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(32, activation='relu',
          kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nArquitectura del Modelo")
model.summary()

#%%
#callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

#%%
# Entrenamiento del modelo
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

#%%
# Visualización

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.title('Pérdida', fontsize=14, fontweight='bold')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
plt.title('Precisión', fontsize=14, fontweight='bold')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

#%%
#Evaluación
y_pred_proba = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

print("\nReporte de Clasificación")
print(classification_report(y_test, y_pred, target_names=le_target.classes_, digits=4))

#%%
#Matriz de confusión
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_target.classes_,
            yticklabels=le_target.classes_)
plt.title('Matriz de Confusión', fontsize=16, fontweight='bold')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.tight_layout()
plt.show()
# %%
