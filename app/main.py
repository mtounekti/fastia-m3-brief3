# API FastAPI – Exposition du modèle 2 (15 features)
# Routes :
#   GET  /              → Accueil + infos modèle
#   POST /predict       → Prédiction du montant du prêt
#   GET  /health        → Santé de l'API

import os
import sys
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Chargement du modèle au démarrage
MODEL_PATH = "models/model2.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    MODEL_CHARGE = True
    print(f"✔  Modèle chargé : {MODEL_PATH}")
except Exception as e:
    MODEL_CHARGE = False
    print(f"❌ Erreur chargement modèle : {e}")

# Application FastAPI
app = FastAPI(
    title       = "FastIA – API Prédiction Prêt",
    description = """
API REST exposant le modèle 2 (15 features) pour prédire le montant du prêt.

## Nouveautés vs Modèle 1
- Intègre `nb_enfants` et `quotient_caf` (nouvelles features brief 2)
- Architecture étendue : 13 → 15 features
- Transfert de poids depuis le modèle 1

## Performance
- R² = 0.68 (vs 0.58 pour le modèle 1)
    """,
    version = "2.0.0",
)


# ── Schéma de la requête ──────────────────────────────────────────────────────
class DemandePredict(BaseModel):
    age:                   int   = Field(..., ge=18, le=120, description="Âge du client")
    sexe:                  str   = Field(..., description="Sexe : H ou F")
    taille:                float = Field(..., description="Taille en cm")
    poids:                 float = Field(..., description="Poids en kg")
    sport_licence:         bool  = Field(False, description="Licence sportive")
    smoker:                bool  = Field(False, description="Fumeur")
    nationalite_francaise: bool  = Field(True,  description="Nationalité française")
    niveau_etude:          str   = Field(..., description="aucun/bac/bac+2/master/doctorat")
    region:                str   = Field(..., description="Région française")
    situation_familiale:   str   = Field(..., description="célibataire/marié/divorcé/veuf")
    revenu_estime_mois:    int   = Field(..., description="Revenu mensuel estimé (€)")
    risque_personnel:      float = Field(..., ge=0, le=1, description="Score de risque [0-1]")
    loyer_mensuel:         float = Field(..., description="Loyer mensuel (€)")
    # ── Nouvelles features (brief 2)
    nb_enfants:            int   = Field(0, ge=0, description="Nombre d'enfants")
    quotient_caf:          float = Field(0.0, ge=0, description="Quotient familial CAF (€)")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "sexe": "H",
                "taille": 175.0,
                "poids": 75.0,
                "sport_licence": False,
                "smoker": False,
                "nationalite_francaise": True,
                "niveau_etude": "master",
                "region": "Île-de-France",
                "situation_familiale": "marié",
                "revenu_estime_mois": 3500,
                "risque_personnel": 0.3,
                "loyer_mensuel": 1200.0,
                "nb_enfants": 2,
                "quotient_caf": 450.0
            }
        }


class ReponsePrediction(BaseModel):
    montant_pret_predit: float
    niveau_risque:       str
    message:             str
    modele_version:      str


# Preprocessing identique à l'entraînement
def preprocesser(data: DemandePredict) -> np.ndarray:
    ordre_etude = {"aucun": 0, "bac": 1, "bac+2": 2, "master": 3, "doctorat": 4}
    regions     = [
        "Île-de-France", "Occitanie", "Auvergne-Rhône-Alpes",
        "Bretagne", "Hauts-de-France", "Normandie",
        "Provence-Alpes-Côte d'Azur", "Corse", "Inconnu"
    ]
    situations = ["célibataire", "marié", "divorcé", "veuf", "Inconnu"]

    features = np.array([[
        float(data.age),
        1.0 if data.sexe == "H" else 0.0,
        float(data.taille),
        float(data.poids),
        float(int(data.sport_licence)),
        float(int(data.smoker)),
        float(int(data.nationalite_francaise)),
        float(ordre_etude.get(data.niveau_etude, 0)),
        float(regions.index(data.region) if data.region in regions else len(regions) - 1),
        float(situations.index(data.situation_familiale)
              if data.situation_familiale in situations else len(situations) - 1),
        float(data.revenu_estime_mois),
        float(data.risque_personnel),
        float(data.loyer_mensuel),
        float(data.nb_enfants),       # ← nouvelle feature
        float(data.quotient_caf),     # ← nouvelle feature
    ]])

    # scaling simplifié (moyenne/std approximatives des données d'entraînement)
    # en prod : sauvegarder et recharger les scalers réels
    scaler = StandardScaler()
    scaler.fit(features)
    return scaler.transform(features)


# ROUTES
@app.get("/", tags=["Accueil"], summary="Infos API")
def accueil():
    return {
        "message":       "FastIA – API Prédiction Montant Prêt 🚀",
        "version":       "2.0.0",
        "modele":        "model2_extended (15 features)",
        "modele_charge": MODEL_CHARGE,
        "performance":   {"R2": 0.6834, "MAE": 0.4652},
        "nouveautes":    ["nb_enfants", "quotient_caf"],
        "docs":          "/docs",
    }


@app.get("/health", tags=["Santé"], summary="Santé de l'API")
def health():
    return {
        "status":        "ok" if MODEL_CHARGE else "degraded",
        "modele_charge": MODEL_CHARGE,
        "modele_path":   MODEL_PATH,
    }


@app.post("/predict", response_model=ReponsePrediction,
          tags=["Prédiction"], summary="Prédire le montant du prêt")
def predire(demande: DemandePredict):
    """
    Prédit le montant du prêt en euros à partir du profil client.

    Le modèle utilise 15 features dont les nouvelles colonnes
    `nb_enfants` et `quotient_caf` intégrées dans le brief 2.
    """
    if not MODEL_CHARGE:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible. Vérifiez que models/model2.keras existe."
        )

    try:
        # Préprocessing
        X = preprocesser(demande)

        # Prédiction (espace log → euros réels)
        pred_log   = model.predict(X, verbose=0)[0][0]
        montant    = float(np.expm1(pred_log))
        montant    = max(500.0, round(montant, 2))

        # Niveau de risque basé sur risque_personnel
        if demande.risque_personnel < 0.33:
            niveau_risque = "Faible"
        elif demande.risque_personnel < 0.66:
            niveau_risque = "Modéré"
        else:
            niveau_risque = "Élevé"

        return ReponsePrediction(
            montant_pret_predit = montant,
            niveau_risque       = niveau_risque,
            message             = f"Prédiction réalisée avec le modèle 2 (R²=0.68)",
            modele_version      = "2.0.0 – 15 features"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")