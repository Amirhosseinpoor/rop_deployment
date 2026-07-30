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
    path('api/chat/<uuid:session_id>/stream/', views.api_stream_message, name='api_stream'),
    path('api/chat/<uuid:session_id>/rename/', views.api_rename_session, name='api_rename'),
    path('api/chat/<uuid:session_id>/pin/', views.api_pin_session, name='api_pin'),
    path('api/chat/<uuid:session_id>/delete/', views.api_delete_session, name='api_delete'),
    path('api/chat/<uuid:session_id>/edit/', views.api_edit_message, name='api_edit'),
    path('api/chat/<uuid:session_id>/feedback/', views.api_feedback, name='api_feedback'),
    path('api/chat/<uuid:session_id>/search/', views.api_search_sessions, name='api_search'),
    path('api/tts/', views.api_tts, name='api_tts'),
    path('api/tts-timed/', views.api_tts_timed, name='api_tts_timed'),

    # STUDIO (superusers)
    path('studio/', views.studio_index, name='studio_index'),
    path('studio/copilot/', views.studio_copilot, name='studio_copilot'),
    path('studio/copilot/avatar/', views.studio_copilot_avatar, name='studio_copilot_avatar'),
    path('studio/new/', views.studio_new, name='studio_new'),
    path('studio/<slug:slug>/edit/', views.studio_edit, name='studio_edit'),
    path('studio/<slug:slug>/delete/', views.studio_delete, name='studio_delete'),
    path('studio/<slug:slug>/kb/', views.studio_kb, name='studio_kb'),
    path('studio/<slug:slug>/kb/reindex/', views.studio_kb_reindex, name='studio_kb_reindex'),
    path('studio/<slug:slug>/kb/<int:pk>/delete/', views.studio_kb_delete, name='studio_kb_delete'),
]
