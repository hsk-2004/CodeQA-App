import hashlib

def hash_password(password: str) -> str:
    """Hashes a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username: str, password: str, users_db: dict) -> bool:
    """Checks if the given username/password matches a stored user record."""
    if username not in users_db:
        return False
    return users_db[username] == hash_password(password)
