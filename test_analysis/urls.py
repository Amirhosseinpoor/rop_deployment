from django.urls import path
from . import views

urlpatterns = [
    # آدرس برای ساخت یا ویرایش پروفایل
    path('profile/', views.create_or_update_health_profile, name='create_update_profile'),

    # آدرس برای نمایش پروفایل و توصیه‌ها
    path('profile/detail/', views.profile_detail_view, name='profile_detail'),
]
