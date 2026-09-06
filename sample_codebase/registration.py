from auth import hash_password

def register_user(username: str, password: str, users_db: dict) -> str:
    """Registers a new user by storing their hashed password."""
    if username in users_db:
        return "Username already exists."
    users_db[username] = hash_password(password)
    return f"User {username} registered successfully."
