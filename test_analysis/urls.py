# test_analysis/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('profile/', views.create_or_update_health_profile, name='create_update_profile'),
    path('profile/detail/', views.profile_detail_view, name='profile_detail'),  # own profile
    path('profile/eye-scan/', views.upload_eye_scan, name='upload_eye_scan'),  # add & screen eye photos
    path('profile/detail/<int:user_id>/', views.profile_detail_view, name='profile_detail_user'),  # viewing others
    path('processing/', views.processing_page, name='health_processing'),
    path('processing/status/', views.report_status, name='health_processing_status'),
    path('play/', views.minigame_page, name='health_minigame'),
    path('doctor-dashboard/', views.doctor_dashboard, name='doctor_dashboard'),
    path('profile/<int:user_id>/edit/', views.manager_edit_profile, name='manager_edit_profile'),
    # Deep Health Research (in-process, no Celery) — own profile + others
    path('deep-research/start/', views.deep_research_start, name='deep_research_start'),
    path('deep-research/status/', views.deep_research_status, name='deep_research_status'),
    path('deep-research/result/', views.deep_research_result, name='deep_research_result'),
    path('deep-research/start/<int:user_id>/', views.deep_research_start, name='deep_research_start_user'),
    path('deep-research/status/<int:user_id>/', views.deep_research_status, name='deep_research_status_user'),
    path('deep-research/result/<int:user_id>/', views.deep_research_result, name='deep_research_result_user'),
    path('chat/health-chat/', views.health_chat_api, name='health_chat_api'),
    path('chat/doctor-assist/', views.doctor_assist_api, name='doctor_assist_api'),
    path('chat/doctor-research/', views.doctor_research_api, name='doctor_research_api'),
]
