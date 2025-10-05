# config/urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from usac.admin_custom import custom_admin_site

urlpatterns = [
    path('admin/', custom_admin_site.urls),
    path('accounts/', include('allauth.urls')),
    path('', include('usac.urls')),

    # فقط یکبار و تمیز:
    path('health/', include('test_analysis.urls')),  # prefix تمیز
    path('rop/', include('single_rop.urls')),
    path('kc/', include('double_rop.urls')),
    path('market/', include('doctors_marketplace.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
