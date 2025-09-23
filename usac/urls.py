from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    # Auth
    path('', views.custom_login, name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),

    # Main
    path('dilemma/', views.dilemma_view, name='dilemma'),
    path('history/', views.history_view, name='history'),
    path('history/export/', views.export_history_csv, name='export_history_csv'),

    # CSV exports (admin)
    path("export/rop/", views.export_misclassified_rop_csv, name="export_rop_csv"),
    path("export/kc/", views.export_misclassified_kc_csv, name="export_kc_csv"),

    # Signup flow (role-based)
    path('signup/choose-role/', views.choose_role_view, name='signup_choose_role'),
    path('signup/manager/', views.signup_manager_view, name='signup_manager'),
    path('signup/doctor/', views.signup_doctor_view, name='signup_doctor'),
    path('signup/employee/', views.signup_employee_view, name='signup_employee'),

    # Manager area
    path('managing/', views.manager_dashboard, name='manager_dashboard'),
    path('managing/member/<int:user_id>/', views.member_detail_view, name='member_detail'),

    # Utils
    path('send-test-email/', views.send_test_email, name='send_test_email'),
]
