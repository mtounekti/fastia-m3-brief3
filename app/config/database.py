from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# URL de la base de données SQLite (fichier local)
# on peut facilement basculer vers PostgreSQL en changeant cette URL
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fastia.db")

# création du moteur SQLAlchemy
engine = create_engine(
    DATABASE_URL,
    # connect_args nécessaire uniquement pour SQLite (gestion du multithreading)
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# fabrique de sessions – utilisée pour interagir avec la base
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Classe de base pour tous les modèles ORM
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()