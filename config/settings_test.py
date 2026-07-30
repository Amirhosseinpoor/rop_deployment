# config/settings_test.py
"""Test settings: run the suite on an in-memory SQLite DB so it needs no
Postgres CREATEDB permission. Use with:
    python manage.py test doctors_marketplace.tests_agents --settings=config.settings_test
"""
from .settings import *  # noqa: F401,F403

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

# Keep tests fast/quiet and independent of external infra.
MIGRATION_MODULES = {}  # build schema straight from models
PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]
