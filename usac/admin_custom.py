# usac/admin_custom.py
from django.contrib import admin
from django.contrib.admin import AdminSite
from django.urls import path

from django.contrib.auth.models import User, Group
from django.contrib.auth.admin import UserAdmin, GroupAdmin

from single_rop.models import PredictionLog
from single_rop.admin import PredictionLogAdmin
from double_rop.models import PredictionResult
from double_rop.admin import PredictionResultAdmin
from test_analysis.models import HealthProfile
from test_analysis.admin import HealthProfileAdmin

from .models import Company, UserProfile, Invitation
from .views import export_misclassified_rop_csv, export_misclassified_kc_csv


class CustomAdminSite(AdminSite):
    site_header = "Mediverse AI Admin"
    site_title = "Mediverse AI Portal"
    index_title = "Dashboard"

    def get_urls(self):
        urls = super().get_urls()
        custom = [
            path("misclassified/rop/", self.admin_view(export_misclassified_rop_csv), name="miscls_rop"),
            path("misclassified/kc/", self.admin_view(export_misclassified_kc_csv), name="miscls_kc"),
        ]
        return custom + urls


# instantiate custom admin site
custom_admin_site = CustomAdminSite(name="custom_admin")


# --- Company / UserProfile / Invitation ---
@admin.register(Company, site=custom_admin_site)
class CompanyAdmin(admin.ModelAdmin):
    list_display = ("name", "email", "phone", "manager", "created_at")
    search_fields = ("name", "email", "manager__username")


@admin.register(UserProfile, site=custom_admin_site)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "role", "company", "national_code", "phone", "created_at")
    list_filter = ("role", "company")
    search_fields = ("user__username", "user__email", "national_code")


@admin.register(Invitation, site=custom_admin_site)
class InvitationAdmin(admin.ModelAdmin):
    list_display = ("company", "role", "national_code", "used_by", "created_at", "used_at")
    list_filter = ("role", "company")
    search_fields = ("national_code", "company__name", "used_by__username")


# --- Auth models ---
custom_admin_site.register(User, UserAdmin)
custom_admin_site.register(Group, GroupAdmin)

# --- Your app models ---
custom_admin_site.register(PredictionLog, PredictionLogAdmin)
custom_admin_site.register(PredictionResult, PredictionResultAdmin)
custom_admin_site.register(HealthProfile, HealthProfileAdmin)
