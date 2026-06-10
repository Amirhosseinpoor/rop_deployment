# doctors_marketplace/urls.py
from django.urls import path
from . import views

app_name = 'doctors_marketplace'

urlpatterns = [
    # PUBLIC
    path('', views.market_index, name='index'),
    path('doctor/<slug:slug>/', views.doctor_detail, name='doctor_detail'),
    path('chat/<uuid:session_id>/', views.chat_view, name='chat'),
    path('api/chat/<uuid:session_id>/send/', views.api_send_message, name='api_send'),

    # STUDIO (superusers)
    path('studio/', views.studio_index, name='studio_index'),
    path('studio/new/', views.studio_new, name='studio_new'),
    path('studio/<slug:slug>/edit/', views.studio_edit, name='studio_edit'),
    path('studio/<slug:slug>/delete/', views.studio_delete, name='studio_delete'),
    path('studio/<slug:slug>/kb/', views.studio_kb, name='studio_kb'),
    path('studio/<slug:slug>/kb/<int:pk>/delete/', views.studio_kb_delete, name='studio_kb_delete'),
]
