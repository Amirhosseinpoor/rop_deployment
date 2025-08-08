# usac/admin_custom.py
from django.contrib.admin import AdminSite
from django.urls import path
from django.utils.decorators import method_decorator
from django.contrib.admin.views.decorators import staff_member_required

from django.contrib.auth.models import User, Group
from django.contrib.auth.admin import UserAdmin, GroupAdmin

from single_rop.models import PredictionLog
from single_rop.admin import PredictionLogAdmin
from double_rop.models import PredictionResult
from double_rop.admin import PredictionResultAdmin
from test_analysis.models import HealthProfile
from test_analysis.admin import HealthProfileAdmin

from .views import export_misclassified_rop_csv, export_misclassified_kc_csv

class CustomAdminSite(AdminSite):
    site_header = "Mediverse AI "

    def get_urls(self):
        urls = super().get_urls()
        custom = [
            path(
                "misclassified/rop/",
                self.admin_view(export_misclassified_rop_csv),
                name="miscls_rop"
            ),
            path(
                "misclassified/kc/",
                self.admin_view(export_misclassified_kc_csv),
                name="miscls_kc"
            ),
        ]
        return custom + urls

custom_admin_site = CustomAdminSite(name="custom_admin")


# Auth models
custom_admin_site.register(User, UserAdmin)
custom_admin_site.register(Group, GroupAdmin)

# Your app models
custom_admin_site.register(PredictionLog, PredictionLogAdmin)
custom_admin_site.register(PredictionResult, PredictionResultAdmin)

custom_admin_site.register(HealthProfile, HealthProfileAdmin)
