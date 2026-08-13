"""Local account storage with stdlib SQLite, scrypt password hashing and signed expiring tokens."""
import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path


class AuthError(Exception):
    pass


class AuthService:
    def __init__(self, database_path: Path, secret: str, token_hours: int):
        self.database_path, self.secret, self.token_hours = database_path, secret.encode(), token_hours

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE, password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL)""")
            connection.execute("""CREATE TABLE IF NOT EXISTS student_profiles (
                user_id INTEGER PRIMARY KEY, profile_json TEXT NOT NULL, updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id))""")

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _hash_password(password: str, salt: bytes | None = None) -> str:
        salt = salt or secrets.token_bytes(16)
        digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
        return f"scrypt${base64.urlsafe_b64encode(salt).decode()}${base64.urlsafe_b64encode(digest).decode()}"

    @staticmethod
    def _verify_password(password: str, stored: str) -> bool:
        _, salt_encoded, digest_encoded = stored.split("$", 2)
        candidate = AuthService._hash_password(password, base64.urlsafe_b64decode(salt_encoded)).split("$", 2)[2]
        return hmac.compare_digest(candidate, digest_encoded)

    def _token(self, user: dict) -> str:
        payload = {"sub": user["id"], "email": user["email"], "exp": int((datetime.now(UTC) + timedelta(hours=self.token_hours)).timestamp())}
        encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
        signature = hmac.new(self.secret, encoded.encode(), hashlib.sha256).hexdigest()
        return f"{encoded}.{signature}"

    def user_from_token(self, authorization: str | None) -> dict:
        if not authorization or not authorization.startswith("Bearer "):
            raise AuthError("Please sign in to access your saved profile.")
        try:
            encoded, signature = authorization.removeprefix("Bearer ").split(".", 1)
            expected = hmac.new(self.secret, encoded.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError("signature")
            padded = encoded + "=" * (-len(encoded) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded))
            if payload["exp"] < int(datetime.now(UTC).timestamp()):
                raise ValueError("expired")
            return {"id": int(payload["sub"]), "email": payload["email"]}
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            raise AuthError("Your session has expired. Please sign in again.") from exc

    def save_profile(self, user_id: int, profile: dict) -> None:
        with self._connection() as connection:
            connection.execute("""INSERT INTO student_profiles (user_id, profile_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET profile_json=excluded.profile_json, updated_at=excluded.updated_at""", (user_id, json.dumps(profile), datetime.now(UTC).isoformat()))

    def load_profile(self, user_id: int) -> dict | None:
        with self._connection() as connection:
            row = connection.execute("SELECT profile_json FROM student_profiles WHERE user_id = ?", (user_id,)).fetchone()
        return json.loads(row["profile_json"]) if row else None

    def signup(self, name: str, email: str, password: str) -> dict:
        try:
            with self._connection() as connection:
                cursor = connection.execute("INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)", (name.strip(), email.lower(), self._hash_password(password), datetime.now(UTC).isoformat()))
                user = {"id": cursor.lastrowid, "name": name.strip(), "email": email.lower()}
        except sqlite3.IntegrityError as exc:
            raise AuthError("An account with this email already exists. Please sign in instead.") from exc
        return {"access_token": self._token(user), "user": user}

    def login(self, email: str, password: str) -> dict:
        with self._connection() as connection:
            row = connection.execute("SELECT id, name, email, password_hash FROM users WHERE email = ?", (email.lower(),)).fetchone()
        if row is None or not self._verify_password(password, row["password_hash"]):
            raise AuthError("Incorrect email or password.")
        user = {"id": row["id"], "name": row["name"], "email": row["email"]}
        return {"access_token": self._token(user), "user": user}
