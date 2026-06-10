from django.contrib import admin
from single_rop.models import PredictionLog
from double_rop.models import PredictionResult

class CombinedAdmin(admin.ModelAdmin):
    list_display = ('user', 'file_name', 'predicted_class', 'timestamp')  # متناسب با مدل
#
# admin.site.register(PredictionLog)
# admin.site.register(PredictionResult)
