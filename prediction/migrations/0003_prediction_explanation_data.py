from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("prediction", "0002_alter_prediction_heatmap_base64"),
    ]

    operations = [
        migrations.AddField(
            model_name="prediction",
            name="explanation_data",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
