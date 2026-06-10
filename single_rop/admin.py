from django.contrib import admin
from django.http import HttpResponse
import csv
from .models import PredictionLog
@admin.action(description="Export misclassified ROP cases as CSV")
def export_misclassified_rop(modeladmin, request, queryset):
    misclassified = queryset.filter(classification_status=-1)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="misclassified_rop.csv"'

    writer = csv.writer(response)
    writer.writerow([
        'Username',
        'Email',
        'File Name',
        'Predicted Class',
        'Corrected Class',
        'Stage Class',
        'Corrected Stage',
        'Zone Class',
        'Corrected Zone',
        'Probability',
        'Stage Probability',
        'Zone Probability',
        'Final Decision',  # NEW
        'Comment',
        'Image URL',
        'Execution Time',
        'Timestamp',
    ])

    for obj in misclassified:
        writer.writerow([
            obj.user.username,
            obj.user.email,
            obj.file_name,
            obj.predicted_class,
            obj.corrected_class or "",
            obj.stage_class or "",
            obj.stage_corrected_class or "",
            obj.zone_class or "",
            obj.zone_corrected_class or "",
            f"{obj.probability:.4f}",
            f"{obj.stage_probability:.4f}" if obj.stage_probability is not None else "",
            f"{obj.zone_probability:.4f}" if obj.zone_probability is not None else "",
            obj.final_decision or "",  # NEW
            obj.review_comment or "",
            request.build_absolute_uri(obj.image_url) if obj.image_url else "",
            obj.execution_time,
            obj.timestamp
        ])

    return response

class PredictionLogAdmin(admin.ModelAdmin):
    list_display = (
        'user', 'file_name',
        'predicted_class', 'stage_class', 'zone_class',
        'final_decision',                       # NEW
        'classification_status', 'timestamp'
    )
    actions = [export_misclassified_rop]


admin.site.register(PredictionLog, PredictionLogAdmin)
