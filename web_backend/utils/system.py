from pathlib import Path

# Get root directory
def get_root_dir() -> Path:
    return Path(__file__).resolve().parent.parent

# Get data directory
def get_data():
    return get_root_dir() / 'data'

# Get data directory
def get_format_data():
    return get_data() / 'format'

# Get report directory
def get_reports():
    return get_root_dir() / 'reports'

# Get backend data directory
def get_backend_data():
    return get_root_dir() / 'src' / 'web_backend' / 'data'

# Get backend chromadb directory
def get_backend_chromadb():
    return get_root_dir() / 'src' / 'web_backend' / 'chromadb'

# Get backend qdrant directory
def get_backend_qdrant():
    return get_root_dir() / 'src' / 'web_backend' / 'qdrant'

# Get config directory
def get_config():
    return get_root_dir() / 'src' / 'config'