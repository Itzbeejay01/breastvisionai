from django.apps import AppConfig


class PredictionConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "prediction"

    def ready(self):
        """Keep Django startup lightweight; models load lazily on first use."""
        return None
