from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("prediction", "0003_prediction_explanation_data"),
    ]

    operations = [
        migrations.AddField(
            model_name="uploadedimage",
            name="owner",
            field=models.ForeignKey(
                blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL,
                related_name="breastvision_uploaded_images", to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AddField(
            model_name="prediction",
            name="owner",
            field=models.ForeignKey(
                blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL,
                related_name="breastvision_predictions", to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AddField(
            model_name="prediction",
            name="meta_feature_schema",
            field=models.JSONField(blank=True, default=list),
        ),
    ]
