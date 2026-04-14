# Modèles ORM SQLAlchemy – mis à jour pour le brief 2
# Modifications par rapport au brief 1 :
#   - DonneeFinanciere : ajout de nb_enfants et quotient_caf
#   - orientation_sexuelle NON ajoutée (supprimée – RGPD Art. 9)

from sqlalchemy import Column, Integer, Float, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from app.config.database import Base


class Client(Base):
    __tablename__ = "clients"

    id     = Column(Integer, primary_key=True, index=True, autoincrement=True)
    nom    = Column(String(100), nullable=False)
    prenom = Column(String(100), nullable=False)
    age    = Column(Integer, nullable=False)
    sexe   = Column(String(1), nullable=False)
    taille = Column(Float, nullable=True)
    poids  = Column(Float, nullable=True)

    sport_licence         = Column(Boolean, nullable=False, default=False)
    smoker                = Column(Boolean, nullable=False, default=False)
    nationalite_francaise = Column(Boolean, nullable=False, default=True)
    date_creation_compte  = Column(String(20), nullable=True)

    profil            = relationship("Profil", back_populates="client",
                                     uselist=False, cascade="all, delete-orphan")
    donnee_financiere = relationship("DonneeFinanciere", back_populates="client",
                                     uselist=False, cascade="all, delete-orphan")


class Profil(Base):
    __tablename__ = "profils"

    id        = Column(Integer, primary_key=True, index=True, autoincrement=True)
    client_id = Column(Integer, ForeignKey("clients.id", ondelete="CASCADE"),
                       unique=True, nullable=False)

    niveau_etude        = Column(String(20), nullable=True)
    region              = Column(String(50), nullable=True)
    situation_familiale = Column(String(20), nullable=True)

    client = relationship("Client", back_populates="profil")


class DonneeFinanciere(Base):
    """
    Table mise à jour en brief 2 :
    Ajout de nb_enfants et quotient_caf via migration Alembic.
    orientation_sexuelle NON intégrée (RGPD Art. 9).
    """
    __tablename__ = "donnees_financieres"

    id        = Column(Integer, primary_key=True, index=True, autoincrement=True)
    client_id = Column(Integer, ForeignKey("clients.id", ondelete="CASCADE"),
                       unique=True, nullable=False)

    # ── Colonnes existantes (brief 1) ─────────────────────────────────────
    revenu_estime_mois  = Column(Integer, nullable=True)
    historique_credits  = Column(Float,   nullable=True)
    risque_personnel    = Column(Float,   nullable=True)
    score_credit        = Column(Float,   nullable=True)
    loyer_mensuel       = Column(Float,   nullable=True)
    montant_pret        = Column(Float,   nullable=True)

    # ── Nouvelles colonnes (brief 2) ──────────────────────────────────────
    nb_enfants   = Column(Integer, nullable=True)   # Ajouté via migration Alembic
    quotient_caf = Column(Float,   nullable=True)   # Ajouté via migration Alembic

    client = relationship("Client", back_populates="donnee_financiere")