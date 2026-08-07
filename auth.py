import hashlib
import hmac
import secrets
import sqlite3
from pathlib import Path
from typing import Optional

DATABASE_PATH = Path(__file__).resolve().parent / "users.db"
PBKDF2_ITERATIONS = 310_000


def get_connection() -> sqlite3.Connection:
    """Return a connection to the local authentication database."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_auth_database() -> None:
    """Create the users table when it does not already exist."""
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()


def normalize_email(email: str) -> str:
    """Normalize email addresses before storage or comparison."""
    return email.strip().lower()


def hash_password(password: str, salt_hex: Optional[str] = None) -> tuple[str, str]:
    """Securely hash a password using PBKDF2-HMAC-SHA256."""
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(32)

    derived_key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )

    return derived_key.hex(), salt.hex()


def verify_password(
    password: str,
    stored_hash: str,
    stored_salt: str,
) -> bool:
    """Safely compare a supplied password against a stored password hash."""
    supplied_hash, _ = hash_password(password, stored_salt)
    return hmac.compare_digest(supplied_hash, stored_hash)


def validate_signup(
    full_name: str,
    email: str,
    password: str,
    confirm_password: str,
) -> tuple[bool, str]:
    """Validate signup details before creating the account."""
    full_name = full_name.strip()
    email = normalize_email(email)

    if len(full_name) < 2:
        return False, "Please enter your full name."

    if "@" not in email or "." not in email.split("@")[-1]:
        return False, "Please enter a valid email address."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    if not any(character.isupper() for character in password):
        return False, "Password must contain at least one uppercase letter."

    if not any(character.islower() for character in password):
        return False, "Password must contain at least one lowercase letter."

    if not any(character.isdigit() for character in password):
        return False, "Password must contain at least one number."

    if password != confirm_password:
        return False, "Passwords do not match."

    return True, ""


def create_user(
    full_name: str,
    email: str,
    password: str,
) -> tuple[bool, str]:
    """Create a new LabMind user."""
    initialize_auth_database()

    full_name = full_name.strip()
    email = normalize_email(email)
    password_hash, password_salt = hash_password(password)

    try:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO users (
                    full_name,
                    email,
                    password_hash,
                    password_salt
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    full_name,
                    email,
                    password_hash,
                    password_salt,
                ),
            )
            connection.commit()

        return True, "Account created successfully."

    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."

    except sqlite3.Error:
        return False, "The account could not be created. Please try again."


def authenticate_user(
    email: str,
    password: str,
) -> tuple[bool, Optional[dict], str]:
    """Authenticate a user and return safe user information."""
    initialize_auth_database()

    email = normalize_email(email)

    with get_connection() as connection:
        user = connection.execute(
            """
            SELECT
                id,
                full_name,
                email,
                password_hash,
                password_salt
            FROM users
            WHERE email = ?
            """,
            (email,),
        ).fetchone()

    if user is None:
        return False, None, "Incorrect email or password."

    if not verify_password(
        password,
        user["password_hash"],
        user["password_salt"],
    ):
        return False, None, "Incorrect email or password."

    safe_user = {
        "id": user["id"],
        "full_name": user["full_name"],
        "email": user["email"],
    }

    return True, safe_user, "Login successful."