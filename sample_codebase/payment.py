from auth import authenticate_user

def process_payment(username: str, password: str, amount: float, users_db: dict) -> str:
    """Processes a payment only if the user is authenticated."""
    if not authenticate_user(username, password, users_db):
        return "Authentication failed. Payment denied."
    if amount <= 0:
        return "Invalid payment amount."
    return f"Payment of ${amount:.2f} processed for {username}."
