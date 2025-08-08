from django.urls import path
from . import views

urlpatterns = [

    path('profile/', views.create_or_update_health_profile, name='create_update_profile'),

    path('profile/detail/', views.profile_detail_view, name='profile_detail'),
]
