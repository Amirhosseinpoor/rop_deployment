# config/urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from usac.admin_custom import custom_admin_site
from single_rop import views as single_rop_views

urlpatterns = [
    path('admin/', custom_admin_site.urls),
    path('accounts/', include('allauth.urls')),
    path('', include('usac.urls')),

    # RAG chat assistant — served in-process so it works with `manage.py runserver`
    # (both with and without trailing slash, to avoid a redirect on POST).
    path('chat', single_rop_views.chat, name='chat'),
    path('chat/', single_rop_views.chat),
    path('chat/stream', single_rop_views.chat_stream, name='chat_stream'),
    path('chat/stream/', single_rop_views.chat_stream),
    path('chat/upload', single_rop_views.chat_upload, name='chat_upload'),
    path('chat/transcribe', single_rop_views.chat_transcribe, name='chat_transcribe'),

    # فقط یکبار و تمیز:
    path('health/', include('test_analysis.urls')),  # prefix تمیز
    path('rop/', include('single_rop.urls')),
    path('kc/', include('double_rop.urls')),
    path('market/', include('doctors_marketplace.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
