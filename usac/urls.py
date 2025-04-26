from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

from .views import export_misclassified_rop_csv, export_misclassified_kc_csv

urlpatterns = [
    path('', views.custom_login, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    path('dilemma/', views.dilemma_view, name='dilemma'),
    path('history/', views.history_view, name='history'),
    path('history/export/', views.export_history_csv, name='export_history_csv'),

    path("export/rop/", export_misclassified_rop_csv, name="export_rop_csv"),
    path("export/kc/", export_misclassified_kc_csv, name="export_kc_csv"),
]

from .views import send_test_email

urlpatterns += [
    path('send-test-email/', send_test_email, name='send_test_email'),
]
