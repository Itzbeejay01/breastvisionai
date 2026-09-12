from pathlib import Path

from django.http import FileResponse, JsonResponse


UI_DIST = Path(__file__).resolve().parent.parent / "breastvisionai-ui" / "dist"


def frontend(request):
    """Serve the compiled React SPA from Django when using port 8000."""
    index = UI_DIST / "index.html"
    if index.exists():
        return FileResponse(index.open("rb"), content_type="text/html")

    return JsonResponse({
        "project": "BreastVisionAI",
        "message": "React UI has not been built yet. Run: cd breastvisionai-ui && npm run build",
    }, status=503)


def root(request):
    return frontend(request)
