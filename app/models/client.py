# Architecture :
#   - Client : entité principale (données personnelles + financières)
#   - Profil : données de profil (éducation, région, situation familiale)
#   - DonneeFinanciere : données financières (loyer, prêt, score)
# Relations :
#   Client 1 ──── 1 Profil
#   Client 1 ──── 1 DonneeFinanciere

from sqlalchemy import (
    Column, Integer, Float, String, Boolean,
    Date, ForeignKey, Enum
)
from sqlalchemy.orm import relationship
from app.config.database import Base
import enum


# Énumérations pour les champs à valeurs fixes

class SexeEnum(str, enum.Enum):
    H = "H"
    F = "F"

class NiveauEtudeEnum(str, enum.Enum):
    aucun    = "aucun"
    bac      = "bac"
    bac2     = "bac+2"
    master   = "master"
    doctorat = "doctorat"

class SituationFamilialeEnum(str, enum.Enum):
    celibataire = "célibataire"
    marie       = "marié"
    divorce     = "divorcé"
    veuf        = "veuf"

class RegionEnum(str, enum.Enum):
    ile_de_france         = "Île-de-France"
    occitanie             = "Occitanie"
    auvergne_rhone_alpes  = "Auvergne-Rhône-Alpes"
    bretagne              = "Bretagne"
    hauts_de_france       = "Hauts-de-France"
    normandie             = "Normandie"
    paca                  = "Provence-Alpes-Côte d'Azur"
    corse                 = "Corse"


# ENTITÉ  CLIENT

class Client(Base):
    """
    entité principale représentant un client FastIA.
    contient les données d'identité et les données corporelles.
    """
    __tablename__ = "clients"

    # Identifiant
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Données personnelles
    nom    = Column(String(100), nullable=False)
    prenom = Column(String(100), nullable=False)
    age    = Column(Integer,     nullable=False)
    sexe   = Column(String(1),   nullable=False)

    # Données corporelles
    taille = Column(Float, nullable=True)
    poids  = Column(Float, nullable=True)

    # Données comportementales
    sport_licence         = Column(Boolean, nullable=False, default=False)
    smoker                = Column(Boolean, nullable=False, default=False)
    nationalite_francaise = Column(Boolean, nullable=False, default=True)

    # Date de création du compte
    date_creation_compte = Column(String(20), nullable=True)

    # Relations (1-to-1)
    profil           = relationship("Profil",           back_populates="client",
                                    uselist=False, cascade="all, delete-orphan")
    donnee_financiere = relationship("DonneeFinanciere", back_populates="client",
                                    uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Client(id={self.id}, nom={self.nom}, prenom={self.prenom})>"


# ENTITÉ PROFIL (données socio-démographiques)

class Profil(Base):
    """
    données de profil socio-démographique d'un client
    relation 1-to-1 avec Client
    """
    __tablename__ = "profils"

    id        = Column(Integer, primary_key=True, index=True, autoincrement=True)
    client_id = Column(Integer, ForeignKey("clients.id", ondelete="CASCADE"),
                       unique=True, nullable=False)

    niveau_etude        = Column(String(20), nullable=True)
    region              = Column(String(50), nullable=True)
    situation_familiale = Column(String(20), nullable=True)

    client = relationship("Client", back_populates="profil")

    def __repr__(self):
        return f"<Profil(client_id={self.client_id}, region={self.region})>"


# ENTITÉ DONNÉE FINANCIÈRE

class DonneeFinanciere(Base):
    """
    données financières associées à un client
    relation 1-to-1 avec Client
    """
    __tablename__ = "donnees_financieres"

    # identifiant
    id        = Column(Integer, primary_key=True, index=True, autoincrement=True)
    client_id = Column(Integer, ForeignKey("clients.id", ondelete="CASCADE"),
                       unique=True, nullable=False)

    # données financières
    revenu_estime_mois  = Column(Integer, nullable=True)
    historique_credits  = Column(Float,   nullable=True)
    risque_personnel    = Column(Float,   nullable=True)
    score_credit        = Column(Float,   nullable=True)
    loyer_mensuel       = Column(Float,   nullable=True)
    montant_pret        = Column(Float,   nullable=True)

    # relation inverse
    client = relationship("Client", back_populates="donnee_financiere")

    def __repr__(self):
        return f"<DonneeFinanciere(client_id={self.client_id}, montant_pret={self.montant_pret})>"