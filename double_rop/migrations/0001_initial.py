# Generated by Django 5.1.7 on 2025-03-30 16:58

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='DualPredictionLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('left_eye_class', models.CharField(max_length=50)),
                ('left_eye_prob', models.FloatField()),
                ('right_eye_class', models.CharField(max_length=50)),
                ('right_eye_prob', models.FloatField()),
                ('z_class', models.CharField(max_length=50)),
                ('z_prob', models.FloatField()),
                ('file_left', models.CharField(max_length=255)),
                ('file_right', models.CharField(max_length=255)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
