from django.urls import path
from .auth_views import csrf, current_user, sign_in, sign_out
from .views import health

urlpatterns = [
    path('health/', health, name='health'),
    path('auth/csrf/', csrf, name='auth-csrf'),
    path('auth/login/', sign_in, name='auth-login'),
    path('auth/logout/', sign_out, name='auth-logout'),
    path('auth/me/', current_user, name='auth-me'),
]
