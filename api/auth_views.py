import json

from django.contrib.auth import authenticate, login, logout
from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST


@ensure_csrf_cookie
@require_GET
def csrf(request):
    return JsonResponse({"csrfToken": get_token(request)})


@require_POST
def sign_in(request):
    try:
        payload = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        payload = request.POST

    username = payload.get("username", "").strip()
    password = payload.get("password", "")
    user = authenticate(request, username=username, password=password)
    if user is None or not user.is_active or not user.is_staff:
        return JsonResponse({"error": "Invalid username or password."}, status=400)

    login(request, user)
    return JsonResponse({
        "authenticated": True,
        "username": user.get_username(),
        "is_staff": user.is_staff,
    })


@require_POST
def sign_out(request):
    logout(request)
    return JsonResponse({"authenticated": False})


@require_GET
def current_user(request):
    if not request.user.is_authenticated:
        return JsonResponse({"authenticated": False}, status=401)
    return JsonResponse({
        "authenticated": True,
        "username": request.user.get_username(),
        "is_staff": request.user.is_staff,
    })
