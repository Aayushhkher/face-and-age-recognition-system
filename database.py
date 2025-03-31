from sqlalchemy import create_engine, Column, Integer, String, Float, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import numpy as np
import os

# Create the database directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Create database engine
engine = create_engine('sqlite:///data/faces.db')
Base = declarative_base()

class FaceEmbedding(Base):
    __tablename__ = 'face_embeddings'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    embedding = Column(LargeBinary, nullable=False)
    image_path = Column(String, nullable=True)

    def get_embedding(self):
        return np.frombuffer(self.embedding, dtype=np.float64)

    @staticmethod
    def create_embedding(name, embedding, image_path=None):
        return FaceEmbedding(
            name=name,
            embedding=embedding.tobytes(),
            image_path=image_path
        )

# Create all tables
Base.metadata.create_all(engine)

# Create session factory
Session = sessionmaker(bind=engine)

def add_face(name, embedding, image_path=None):
    """Add a new face embedding to the database."""
    session = Session()
    try:
        face = FaceEmbedding.create_embedding(name, embedding, image_path)
        session.add(face)
        session.commit()
        return True
    except Exception as e:
        print(f"Error adding face: {e}")
        session.rollback()
        return False
    finally:
        session.close()

def get_all_faces():
    """Retrieve all face embeddings from the database."""
    session = Session()
    try:
        faces = session.query(FaceEmbedding).all()
        return [(face.name, face.get_embedding()) for face in faces]
    finally:
        session.close()

def delete_face(name):
    """Delete a face embedding from the database."""
    session = Session()
    try:
        face = session.query(FaceEmbedding).filter_by(name=name).first()
        if face:
            session.delete(face)
            session.commit()
            return True
        return False
    except Exception as e:
        print(f"Error deleting face: {e}")
        session.rollback()
        return False
    finally:
        session.close()

def get_face_by_name(name):
    """Retrieve a specific face embedding by name."""
    session = Session()
    try:
        face = session.query(FaceEmbedding).filter_by(name=name).first()
        if face:
            return face.get_embedding()
        return None
    finally:
        session.close() 