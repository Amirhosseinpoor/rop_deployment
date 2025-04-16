from django.db import models
from django.contrib.auth.models import User


class PredictionResult(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    # Predicted labels
    left_label = models.CharField(max_length=50)
    left_probability = models.FloatField()

    right_label = models.CharField(max_length=50)
    right_probability = models.FloatField()

    z_class_label = models.CharField(max_length=50)
    z_class_probability = models.FloatField()
    left_image = models.ImageField(upload_to='double_rop/', null=True, blank=True)
    right_image = models.ImageField(upload_to='double_rop/', null=True, blank=True)
    left_image_url = models.URLField(max_length=500, blank=True, null=True)
    right_image_url = models.URLField(max_length=500, blank=True, null=True)
    corrected_left_label = models.CharField(max_length=100, blank=True, null=True)
    corrected_right_label = models.CharField(max_length=100, blank=True, null=True)
    corrected_z_label = models.CharField(max_length=100, blank=True, null=True)
    review_comment = models.TextField(blank=True, null=True)

    # Inference time
    inference_time = models.FloatField(help_text="Inference duration in seconds")


    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        updated = False
        if self.left_image and not self.left_image_url:
            self.left_image_url = self.left_image.url
            updated = True
        if self.right_image and not self.right_image_url:
            self.right_image_url = self.right_image.url
            updated = True
        if updated:
            super().save(update_fields=['left_image_url', 'right_image_url'])


    def __str__(self):
        return f"{self.user.username} ({self.user.email})"
    #1
