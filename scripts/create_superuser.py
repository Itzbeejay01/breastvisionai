"""Create/update the deployment administrator from environment variables."""

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


username = os.getenv("DJANGO_SUPERUSER_USERNAME")
password = os.getenv("DJANGO_SUPERUSER_PASSWORD")
email = os.getenv("DJANGO_SUPERUSER_EMAIL", "")

if username and password:
    User = get_user_model()
    user, _ = User.objects.get_or_create(username=username)
    user.email = email
    user.is_staff = True
    user.is_superuser = True
    user.is_active = True
    user.set_password(password)
    user.save()
    print(f"Deployment administrator ready: {username}")
else:
    print("Deployment administrator variables are not set; skipping creation.")
