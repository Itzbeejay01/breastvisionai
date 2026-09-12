"""Create/update the deployment administrator from environment variables."""

import os

import django


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
