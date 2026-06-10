# test_analysis/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('profile/', views.create_or_update_health_profile, name='create_update_profile'),
    path('profile/detail/', views.profile_detail_view, name='profile_detail'),  # own profile
    path('profile/detail/<int:user_id>/', views.profile_detail_view, name='profile_detail_user'),  # viewing others
    path('processing/', views.processing_page, name='health_processing'),
    path('processing/status/', views.report_status, name='health_processing_status'),
    path('play/', views.minigame_page, name='health_minigame'),
    path('doctor-dashboard/', views.doctor_dashboard, name='doctor_dashboard'),
    path('profile/<int:user_id>/edit/', views.manager_edit_profile, name='manager_edit_profile'),
    path('chat/health-chat/', views.health_chat_api, name='health_chat_api'),
]
