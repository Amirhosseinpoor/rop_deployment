from django.contrib import admin
from django.http import HttpResponse
import csv
from .models import PredictionResult

@admin.action(description="Export misclassified KC cases as CSV")
def export_misclassified_kc(modeladmin, request, queryset):
    misclassified = queryset.filter(classification_status=-1)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="misclassified_kc.csv"'

    writer = csv.writer(response)
    writer.writerow([
        'Username',
        'Email',
        'Predicted Labels',
        'Corrected Labels',
        'Probabilities',
        'Comment',
        'Image URLs',
        'Inference Time',
        'Timestamp',
    ])

    for obj in misclassified:
        image_urls = []
        if obj.left_image:
            image_urls.append(request.build_absolute_uri(obj.left_image.url))
        if obj.right_image:
            image_urls.append(request.build_absolute_uri(obj.right_image.url))

        writer.writerow([
            obj.user.username,
            obj.user.email,
            f"L: {obj.left_label}, R: {obj.right_label}, Z: {obj.z_class_label}",
            f"L: {obj.corrected_left_label}, R: {obj.corrected_right_label}, Z: {obj.corrected_z_label}",
            f"L: {obj.left_probability:.4f}, R: {obj.right_probability:.4f}, Z: {obj.z_class_probability:.4f}",
            obj.review_comment or "",
            " | ".join(image_urls),
            f"{obj.inference_time:.2f}s",
            obj.created_at
        ])

    return response

class PredictionResultAdmin(admin.ModelAdmin):
    list_display = ('user', 'left_label', 'right_label', 'z_class_label', 'classification_status', 'created_at')
    actions = [export_misclassified_kc]

admin.site.register(PredictionResult, PredictionResultAdmin)
