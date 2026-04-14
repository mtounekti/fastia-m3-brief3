# Modèle 1 – Baseline (13 features, données brief 1)
# Ce script entraîne le modèle de référence avec les données du brief 1
# Il sera sauvegardé puis ses poids seront transférés dans le modèle 2
# Architecture :
# Entrée (13) → Dense(128, relu) → Dense(64, relu) → Dense(32, relu) → Dense(1)
# Stack : TensorFlow/Keras + MLflow pour le tracking

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config.database import SessionLocal
from app.models.client import Client, Profil, DonneeFinanciere

os.makedirs("models",     exist_ok=True)
os.makedirs("graphiques", exist_ok=True)

# Paramètres
EPOCHS        = 100
BATCH_SIZE    = 64
LEARNING_RATE = 0.0005
TEST_SIZE     = 0.2
RANDOM_STATE  = 42

# SECTION 1 – CHARGEMENT DEPUIS LA BASE
print("=" * 65)
print("  MODÈLE 1 – CHARGEMENT DES DONNÉES (13 features)")
print("=" * 65)

db = SessionLocal()

resultats = db.query(
    Client.age,
    Client.sexe,
    Client.taille,
    Client.poids,
    Client.sport_licence,
    Client.smoker,
    Client.nationalite_francaise,
    Profil.niveau_etude,
    Profil.region,
    Profil.situation_familiale,
    DonneeFinanciere.revenu_estime_mois,
    DonneeFinanciere.risque_personnel,
    DonneeFinanciere.loyer_mensuel,
    DonneeFinanciere.montant_pret,
).join(Profil, Client.id == Profil.client_id)\
 .join(DonneeFinanciere, Client.id == DonneeFinanciere.client_id)\
 .all()

db.close()

colonnes = [
    "age", "sexe", "taille", "poids", "sport_licence", "smoker",
    "nationalite_francaise", "niveau_etude", "region",
    "situation_familiale", "revenu_estime_mois", "risque_personnel",
    "loyer_mensuel", "montant_pret"
]
df = pd.DataFrame(resultats, columns=colonnes)
print(f"\n✔  {len(df)} enregistrements chargés depuis la base")

# SECTION 2 – PRÉPROCESSING
print("\n" + "=" * 65)
print("  PRÉPROCESSING")
print("=" * 65)

# Encodage
df["sport_licence"]         = df["sport_licence"].astype(int)
df["smoker"]                = df["smoker"].astype(int)
df["nationalite_francaise"] = df["nationalite_francaise"].astype(int)
df["sexe"]                  = (df["sexe"] == "H").astype(int)

ordre_etude = ["aucun", "bac", "bac+2", "master", "doctorat"]
df["niveau_etude"] = df["niveau_etude"].map(
    {v: i for i, v in enumerate(ordre_etude)}).fillna(0)

le_region    = LabelEncoder()
le_situation = LabelEncoder()
df["region"]              = le_region.fit_transform(df["region"].fillna("Inconnu"))
df["situation_familiale"] = le_situation.fit_transform(
    df["situation_familiale"].fillna("Inconnu"))

# Imputation
imputer    = SimpleImputer(strategy="median")
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# log1p sur la target
df_imputed["montant_pret"] = np.log1p(df_imputed["montant_pret"])

# Split
X = df_imputed.drop(columns=["montant_pret"]).values
y = df_imputed["montant_pret"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train  = scaler_X.fit_transform(X_train)
X_test   = scaler_X.transform(X_test)
y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_s  = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"  ✔  Features : {X_train.shape[1]} | Train : {len(X_train)} | Test : {len(X_test)}")

# SECTION 3 – MODÈLE 1
print("\n" + "=" * 65)
print("  ARCHITECTURE MODÈLE 1")
print("=" * 65)

model1 = Sequential([
    Input(shape=(X_train.shape[1],), name="input_13f"),
    Dense(128, activation="relu", name="dense_1"),
    Dense(64,  activation="relu", name="dense_2"),
    Dense(32,  activation="relu", name="dense_3"),
    Dense(1,                      name="output")
], name="model1_baseline")

model1.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE),
    loss="mse",
    metrics=["mae"]
)
model1.summary()

# SECTION 4 – ENTRAÎNEMENT AVEC MLFLOW
print("\n" + "=" * 65)
print("  ENTRAÎNEMENT MODÈLE 1 (avec MLflow)")
print("=" * 65)

mlflow.set_experiment("fastia_modele_pret")

with mlflow.start_run(run_name="model1_baseline_13features"):

    # Logging des paramètres
    mlflow.log_params({
        "nb_features":    X_train.shape[1],
        "epochs":         EPOCHS,
        "batch_size":     BATCH_SIZE,
        "learning_rate":  LEARNING_RATE,
        "architecture":   "13→128→64→32→1",
        "modele":         "model1_baseline"
    })

    early_stop = EarlyStopping(monitor="val_loss", patience=10,
                               restore_best_weights=True)

    history = model1.fit(
        X_train, y_train_s,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    # Évaluation
    loss, mae = model1.evaluate(X_test, y_test_s, verbose=0)
    r2 = 1 - np.sum((y_test_s - model1.predict(X_test).flatten())**2) / \
             np.sum((y_test_s - np.mean(y_test_s))**2)

    # Logging des métriques
    mlflow.log_metrics({
        "test_loss": round(loss, 4),
        "test_mae":  round(mae, 4),
        "test_r2":   round(r2, 4),
    })

    print(f"\n  📊 Résultats Modèle 1 :")
    print(f"     Loss (MSE) : {loss:.4f}")
    print(f"     MAE        : {mae:.4f}")
    print(f"     R²         : {r2:.4f}")

    # Graphique loss
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.history["loss"],     label="Train loss", color="#E07B54", linewidth=2)
    ax.plot(history.history["val_loss"], label="Val loss",   color="#5B8DB8",
            linewidth=2, linestyle="--")
    ax.set_title("Modèle 1 – Loss (13 features)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Époque")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("graphiques/loss_model1.png", dpi=150)
    plt.close()
    mlflow.log_artifact("graphiques/loss_model1.png")
    print("  ✔  Graphique sauvegardé : graphiques/loss_model1.png")

    # Sauvegarde du modèle
    model1.save("models/model1.keras")
    mlflow.log_artifact("models/model1.keras")
    print("  ✔  Modèle sauvegardé : models/model1.keras")

print("\n  Entraînement modèle 1 terminé ✅")
print(f"  👉 Lance : mlflow ui  pour voir les résultats")