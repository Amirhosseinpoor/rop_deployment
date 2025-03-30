from django.db import models

class PredictionLog(models.Model):
    file_name = models.CharField(max_length=255)
    predicted_class = models.CharField(max_length=100)
    probability = models.FloatField()
    execution_time = models.IntegerField(default=0, help_text="Execution time in milliseconds")
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.file_name} → {self.predicted_class} ({self.probability:.3f})"
