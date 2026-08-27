from django.http import JsonResponse


def root(request):
    return JsonResponse({
        "project": "BreastVisionAI",
        "message": "Welcome to BreastVisionAI API",
        "endpoints": {
            "health": request.build_absolute_uri('/api/health/'),
            "admin": request.build_absolute_uri('/admin/')
        }
    })
