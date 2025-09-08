# test_analysis/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('profile/', views.create_or_update_health_profile, name='create_update_profile'),
    path('profile/detail/', views.profile_detail_view, name='profile_detail'),
    path('processing/', views.processing_page, name='health_processing'),
    path('processing/status/', views.report_status, name='health_processing_status'),
    path('play/', views.minigame_page, name='health_minigame'),
]
