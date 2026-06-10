# doctors_marketplace/admin.py
from django.contrib import admin
from .models import Doctor, ChatSession, ChatMessage

@admin.register(Doctor)
class DoctorAdmin(admin.ModelAdmin):
    list_display = ("name_fa", "specialization_fa", "persona", "is_active")
    list_filter = ("specialization", "persona", "is_active")
    search_fields = ("name", "name_fa", "bio", "bio_fa", "headline_fa", "specialization_fa")
    prepopulated_fields = {"slug": ("name",)}  # نام انگلیسی برای اسلاگ پایدار

class ChatMessageInline(admin.TabularInline):
    model = ChatMessage
    extra = 0
    readonly_fields = ("role", "content", "tokens", "created_at")

@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "doctor", "created_at", "updated_at")
    list_filter = ("doctor",)
    search_fields = ("user__username", "doctor__name_fa", "doctor__name")
    inlines = [ChatMessageInline]
