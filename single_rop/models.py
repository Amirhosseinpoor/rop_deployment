# webapp/app/models.py
from django.db import models
from django.contrib.auth.models import User


class PredictionLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file_name = models.CharField(max_length=255)
    predicted_class = models.CharField(max_length=100)
    probability = models.FloatField()
    execution_time = models.IntegerField(default=0, help_text="Execution time in milliseconds")
    timestamp = models.DateTimeField(auto_now_add=True)

    image = models.ImageField(upload_to='single_rop/', null=True, blank=True)
    image_url = models.URLField(max_length=500, blank=True, null=True)

    corrected_class = models.CharField(max_length=100, blank=True, null=True)
    review_comment = models.TextField(blank=True, null=True)

    stage_class = models.CharField(max_length=100, blank=True, null=True)
    stage_probability = models.FloatField(blank=True, null=True, default=0.0)
    stage_corrected_class = models.CharField(max_length=100, blank=True, null=True)

    zone_class = models.CharField(max_length=100, blank=True, null=True)
    zone_probability = models.FloatField(blank=True, null=True, default=0.0)
    zone_corrected_class = models.CharField(max_length=100, blank=True, null=True)

    final_decision = models.CharField(max_length=100, blank=True, null=True)
    classification_status = models.IntegerField(default=1)

    segmented_image = models.ImageField(upload_to='segmented/', null=True, blank=True)
    segmented_image_url = models.URLField(max_length=500, blank=True, null=True)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        if self.image and not self.image_url:
            self.image_url = self.image.url
            super().save(update_fields=['image_url'])
        if self.segmented_image and not self.segmented_image_url:
            self.segmented_image_url = self.segmented_image.url
            super().save(update_fields=['segmented_image_url'])

    def __str__(self):
        return f"{self.user.username} ({self.user.email})"
