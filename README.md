# M3 Brief 3 – Extension du Modèle IA
### Projet FastIA – Adaptation de la couche d'entrée sans perte des apprentissages

---

## Description

FastIA a enrichi sa base de données avec de nouvelles features (`nb_enfants`, `quotient_caf`).
Ce projet adapte le réseau de neurones existant pour intégrer ces nouvelles entrées **sans perdre
les poids appris** lors de l'entraînement précédent, via un transfert de poids.

---

## Structure du projet

```
fastia-m3-brief3-model-extension/
├── app/
│   ├── config/
│   │   └── database.py          # Connexion SQLAlchemy
│   ├── models/
│   │   └── client.py            # Modèles ORM (avec nb_enfants, quotient_caf)
│   └── main.py                  # API FastAPI – exposition modèle 2
├── scripts/
│   ├── train_model1.py          # Entraînement modèle baseline (13 features)
│   └── train_model2.py          # Adaptation + transfert poids (15 features)
├── models/
│   ├── model1.keras             # Modèle baseline sauvegardé
│   └── model2.keras             # Modèle étendu sauvegardé
├── graphiques/
│   ├── loss_model1.png          # Courbe loss modèle 1
│   └── comparaison_modeles.png  # Comparaison modèle 1 vs 2
├── requirements.txt
└── README.md
```

---

## Principe du transfert de poids

Le brief impose d'**adapter la couche d'entrée sans perdre les apprentissages**.

```
MODÈLE 1 (13 features)          MODÈLE 2 (15 features)
────────────────────             ────────────────────────
Input(13)                  →     Input(15)  ← NOUVELLE couche
Dense(128) relu   ══════════════ Dense(128) relu  ✅ poids transférés
Dense(64)  relu   ══════════════ Dense(64)  relu  ✅ poids transférés
Dense(32)  relu   ══════════════ Dense(32)  relu  ✅ poids transférés
Dense(1)          ══════════════ Dense(1)          ✅ poids transférés
```

**3 couches sur 4 conservent leurs poids.** Seule la couche d'entrée
est réinitialisée car sa forme change (13→15 features).

---

## Résultats et comparaison

| | Modèle 1 | Modèle 2 | Évolution |
|---|---|---|---|
| Features | 13 | **15** | +nb_enfants, +quotient_caf |
| Loss (MSE) | 0.4280 | **0.3259** | ✅ -24% |
| MAE | 0.5369 | **0.4652** | ✅ -13% |
| R² | 0.5842 | **0.6834** | ✅ +10 points |

Les nouvelles features `nb_enfants` et `quotient_caf` apportent un **gain significatif** de performance.

---

## Installation et utilisation

```bash
# Cloner le repo
git clone https://github.com/mtounekti/fastia-m3-brief3-model-extension.git
cd fastia-m3-brief3-model-extension

# Environnement virtuel
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copier la base de données (depuis brief 2)
cp ~/projects/fastia-m3-brief2-db-extension/fastia.db .

# 1. Entraîner le modèle 1 (baseline)
python3 scripts/train_model1.py

# 2. Entraîner le modèle 2 (avec transfert)
python3 scripts/train_model2.py

# 3. Lancer l'API
uvicorn app.main:app --reload

# 4. Voir les runs MLflow
mlflow ui
```

---

## API FastAPI

Documentation Swagger : **http://localhost:8000/docs**

| Route | Méthode | Description |
|---|---|---|
| `/` | GET | Infos API et performances |
| `/health` | GET | Santé de l'API |
| `/predict` | POST | Prédiction du montant du prêt |

### Exemple de requête

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "sexe": "H",
    "taille": 175.0,
    "poids": 75.0,
    "sport_licence": false,
    "smoker": false,
    "nationalite_francaise": true,
    "niveau_etude": "master",
    "region": "Île-de-France",
    "situation_familiale": "marié",
    "revenu_estime_mois": 3500,
    "risque_personnel": 0.3,
    "loyer_mensuel": 1200.0,
    "nb_enfants": 2,
    "quotient_caf": 450.0
  }'
```

### Exemple de réponse

```json
{
  "montant_pret_predit": 12500.50,
  "niveau_risque": "Faible",
  "message": "Prédiction réalisée avec le modèle 2 (R²=0.68)",
  "modele_version": "2.0.0 – 15 features"
}
```

---

## Suivi MLflow

```bash
mlflow ui
# → http://localhost:5000
```

Deux runs disponibles :
- `model1_baseline_13features` – R²=0.58
- `model2_extended_15features` – R²=0.68

---

## Stack technique

| Outil | Usage |
|---|---|
| TensorFlow/Keras | Architecture et entraînement |
| MLflow | Tracking des expériences |
| FastAPI | Exposition du modèle |
| SQLAlchemy | Chargement des données depuis la BDD |

---

*Brief M3 – Extension Modèle IA | FastIA 2025 By Maroua Tounekti*