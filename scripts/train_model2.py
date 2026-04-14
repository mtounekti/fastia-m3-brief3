# Modèle 2 – Extension couche d'entrée + transfert de poids (15 features)
# Ce script :
#   1. Charge le modèle 1 sauvegardé (13 features)
#   2. Crée un modèle 2 avec 15 features (+ nb_enfants + quotient_caf)
#   3. Transfère les poids compatibles (dense_1, dense_2, dense_3, output)
#   4. Réentraîne le modèle avec les nouvelles données
#   5. Compare les performances avec le modèle 1

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

#  params
EPOCHS        = 100
BATCH_SIZE    = 64
LEARNING_RATE = 0.0005
TEST_SIZE     = 0.2
RANDOM_STATE  = 42

# SECTION 1 – CHARGEMENT DEPUIS LA BASE (avec nouvelles colonnes)

print("=" * 65)
print("  MODÈLE 2 – CHARGEMENT DES DONNÉES (15 features)")
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
    DonneeFinanciere.nb_enfants,       # ← nouvelle colonne
    DonneeFinanciere.quotient_caf,     # ← nouvelle colonne
    DonneeFinanciere.montant_pret,
).join(Profil, Client.id == Profil.client_id)\
 .join(DonneeFinanciere, Client.id == DonneeFinanciere.client_id)\
 .all()

db.close()

colonnes = [
    "age", "sexe", "taille", "poids", "sport_licence", "smoker",
    "nationalite_francaise", "niveau_etude", "region",
    "situation_familiale", "revenu_estime_mois", "risque_personnel",
    "loyer_mensuel", "nb_enfants", "quotient_caf", "montant_pret"
]
df = pd.DataFrame(resultats, columns=colonnes)
print(f"\n✔  {len(df)} enregistrements chargés depuis la base")

# vérification des nouvelles colonnes
nb_enfants_null = df["nb_enfants"].isnull().sum()
caf_null        = df["quotient_caf"].isnull().sum()
print(f"  nb_enfants NULL  : {nb_enfants_null}")
print(f"  quotient_caf NULL: {caf_null}")

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

# SECTION 3 – CHARGEMENT MODÈLE 1 + TRANSFERT DE POIDS
print("\n" + "=" * 65)
print("  TRANSFERT DE POIDS – MODÈLE 1 → MODÈLE 2")
print("=" * 65)

if not os.path.exists("models/model1.keras"):
    print("❌ models/model1.keras introuvable ! Lance d'abord train_model1.py")
    sys.exit(1)

# chargement du modèle 1
model1_loaded = tf.keras.models.load_model(
    "models/model1.keras",
    compile=False
)
print(f"\n✔  Modèle 1 chargé : {model1_loaded.name}")

# création du modèle 2 avec 15 features
model2 = Sequential([
    Input(shape=(X_train.shape[1],), name="input_15f"),  # ← couche d'entrée étendue
    Dense(128, activation="relu", name="dense_1"),
    Dense(64,  activation="relu", name="dense_2"),
    Dense(32,  activation="relu", name="dense_3"),
    Dense(1,                      name="output")
], name="model2_extended")

# transfert des poids des couches compatibles
print("\n  Transfert des poids :")
couches_transferees = 0
for layer in model2.layers:
    try:
        old_layer = model1_loaded.get_layer(layer.name)
        layer.set_weights(old_layer.get_weights())
        print(f"  ✅ Poids transférés : {layer.name}")
        couches_transferees += 1
    except ValueError:
        print(f"  ⛔ Nouvelle couche (pas de transfert) : {layer.name}")

print(f"\n  {couches_transferees} couches transférées sur {len(model2.layers)}")
print(f"  → La couche d'entrée est nouvelle (13→15 features)")
print(f"  → Les couches cachées conservent leurs apprentissages")

model2.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE),
    loss="mse",
    metrics=["mae"]
)
model2.summary()

# SECTION 4 – ENTRAÎNEMENT AVEC MLFLOW
print("\n" + "=" * 65)
print("  ENTRAÎNEMENT MODÈLE 2 (avec MLflow)")
print("=" * 65)

mlflow.set_experiment("fastia_modele_pret")

with mlflow.start_run(run_name="model2_extended_15features"):

    mlflow.log_params({
        "nb_features":         X_train.shape[1],
        "epochs":              EPOCHS,
        "batch_size":          BATCH_SIZE,
        "learning_rate":       LEARNING_RATE,
        "architecture":        "15→128→64→32→1",
        "modele":              "model2_extended",
        "transfert_poids":     True,
        "couches_transferees": couches_transferees,
        "nouvelles_features":  "nb_enfants, quotient_caf",
    })

    early_stop = EarlyStopping(monitor="val_loss", patience=10,
                               restore_best_weights=True)

    history = model2.fit(
        X_train, y_train_s,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    # Évaluation
    loss, mae = model2.evaluate(X_test, y_test_s, verbose=0)
    r2 = 1 - np.sum((y_test_s - model2.predict(X_test).flatten())**2) / \
             np.sum((y_test_s - np.mean(y_test_s))**2)

    mlflow.log_metrics({
        "test_loss": round(loss, 4),
        "test_mae":  round(mae, 4),
        "test_r2":   round(r2, 4),
    })

    print(f"\n  📊 Résultats Modèle 2 :")
    print(f"     Loss (MSE) : {loss:.4f}")
    print(f"     MAE        : {mae:.4f}")
    print(f"     R²         : {r2:.4f}")

    #Graphique comparatif
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Comparaison Modèle 1 vs Modèle 2", fontsize=13, fontweight="bold")

    # Loss modèle 2
    ax1 = axes[0]
    ax1.plot(history.history["loss"],     label="Train loss", color="#E07B54", linewidth=2)
    ax1.plot(history.history["val_loss"], label="Val loss",   color="#5B8DB8",
             linewidth=2, linestyle="--")
    ax1.set_title("Modèle 2 – Loss (15 features + transfert)", fontsize=11)
    ax1.set_xlabel("Époque")
    ax1.set_ylabel("MSE Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    #comparaison métriques
    ax2 = axes[1]
    modeles  = ["Modèle 1\n(13 features)", "Modèle 2\n(15 features)"]
    r2_scores = [0.5842, round(r2, 4)]  # R² modèle 1 stocké
    colors   = ["#AACCE0", "#028090"]
    barres   = ax2.bar(modeles, r2_scores, color=colors, edgecolor="white", width=0.4)
    ax2.set_title("Comparaison R²", fontsize=11)
    ax2.set_ylabel("R²")
    ax2.set_ylim(0, 1)
    for b, v in zip(barres, r2_scores):
        ax2.text(b.get_x() + b.get_width()/2, v + 0.01,
                 f"{v:.4f}", ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig("graphiques/comparaison_modeles.png", dpi=150)
    plt.close()
    mlflow.log_artifact("graphiques/comparaison_modeles.png")
    print("  ✔  Graphique sauvegardé : graphiques/comparaison_modeles.png")

    # Sauvegarde
    model2.save("models/model2.keras")
    mlflow.log_artifact("models/model2.keras")
    print("  ✔  Modèle sauvegardé : models/model2.keras")

#Résumé comparatif
print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  COMPARAISON MODÈLE 1 vs MODÈLE 2                          │
  ├──────────────────┬──────────────┬──────────────────────────┤
  │                  │  Modèle 1    │  Modèle 2                │
  │                  │  13 features │  15 features + transfert │
  ├──────────────────┼──────────────┼──────────────────────────┤
  │  Loss (MSE)      │  0.4280      │  {loss:.4f}                    │
  │  MAE             │  0.5369      │  {mae:.4f}                    │
  │  R²              │  0.5842      │  {r2:.4f}                    │
  └──────────────────┴──────────────┴──────────────────────────┘
""")
print("  Entraînement modèle 2 terminé ✅")
print(f"  👉 Lance : mlflow ui  pour comparer les deux runs")