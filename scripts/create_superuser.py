"""Create/update the deployment administrator from environment variables.

This script is intentionally idempotent: it can run on every container start
and will repair the configured account if the database already exists.
"""

import os
import sys
from pathlib import Path

import django


# When this file is executed directly (as it is in the production container),
# Python starts with ``scripts/`` on sys.path rather than the project root.
# Add the root explicitly so the Django settings package is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "breastvisionai.settings")
django.setup()

from django.contrib.auth import get_user_model  # noqa: E402


username = os.getenv("DJANGO_SUPERUSER_USERNAME", "").strip()
password = os.getenv("DJANGO_SUPERUSER_PASSWORD", "")
email = os.getenv("DJANGO_SUPERUSER_EMAIL", "").strip()

missing = [
    name for name, value in (
        ("DJANGO_SUPERUSER_USERNAME", username),
        ("DJANGO_SUPERUSER_PASSWORD", password),
    ) if not value
]
if missing:
    raise RuntimeError(
        "Cannot create the deployment administrator; missing environment "
        f"variable(s): {', '.join(missing)}"
    )

User = get_user_model()
user, created = User.objects.get_or_create(username=username)
user.email = email
user.is_staff = True
user.is_superuser = True
user.is_active = True
user.set_password(password)
user.save()

if not user.check_password(password) or not user.is_staff or not user.is_active:
    raise RuntimeError(f"Deployment administrator verification failed: {username}")

action = "created" if created else "updated"
print(f"Deployment administrator {action} and verified: {username}")
