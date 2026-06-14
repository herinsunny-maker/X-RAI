import os

def get_project_root():
    """
    Returns the absolute path to the project root directory.
    Assumes this file is located in 'backend/'.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def get_backend_path():
    """Returns absolute path to the backend directory."""
    return os.path.join(get_project_root(), "backend")

def get_ml_model_path():
    """Returns absolute path to the ml_model directory."""
    return os.path.join(get_project_root(), "ml_model")

def get_ml_pipeline_path():
    """Returns absolute path to the ml_pipeline directory."""
    return os.path.join(get_project_root(), "ml_pipeline")

def get_upload_path():
    """Returns absolute path to the uploads directory."""
    return os.path.join(get_backend_path(), "static", "uploads")

def get_temp_path():
    """Returns absolute path to the temp uploads directory."""
    return os.path.join(get_backend_path(), "static", "temp_uploads")

def get_db_path():
    """Returns absolute path to the database file."""
    return os.path.join(get_backend_path(), "knee_oa.db")
