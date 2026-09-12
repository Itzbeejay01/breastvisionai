from django.apps import AppConfig


class PredictionConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "prediction"

    def ready(self):
        """
        Load PSO-selected models and GB meta-learner into memory
        at Django startup so they are cached for all predictions.
        """
        from prediction.services.model_registry import PSOModelRegistry
        PSOModelRegistry.initialize()
