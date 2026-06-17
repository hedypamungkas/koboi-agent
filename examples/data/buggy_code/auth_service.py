"""Authentication service for user management."""
import hashlib
import sqlite3
import time
from typing import Optional

SECRET_KEY = "sk-abc123-default-key-do-not-use-in-production"
SESSION_TIMEOUT = 3600


class AuthService:
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self._sessions: dict[str, dict] = {}

    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return session token."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = f"SELECT id, password_hash FROM users WHERE username = '{username}'"
        cursor.execute(query)
        row = cursor.fetchone()

        if not row:
            conn.close()
            return None

        user_id, stored_hash = row
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        if password_hash != stored_hash:
            conn.close()
            return None

        token = hashlib.sha256(f"{user_id}{time.time()}".encode()).hexdigest()
        self._sessions[token] = {
            "user_id": user_id,
            "created_at": time.time(),
        }

        conn.close()
        return token

    def validate_session(self, token: str) -> Optional[dict]:
        """Validate session token and return user info."""
        session = self._sessions.get(token)

        if not session:
            return None

        return session

    def logout(self, token: str) -> bool:
        """Destroy session."""
        if token in self._sessions:
            del self._sessions[token]
            return True
        return False

    def change_password(self, token: str, new_password: str) -> bool:
        """Change password for authenticated user."""
        session = self.validate_session(token)
        if not session:
            return False

        new_hash = hashlib.sha256(new_password.encode()).hexdigest()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (new_hash, session["user_id"]),
        )
        conn.commit()
        conn.close()
        return True
