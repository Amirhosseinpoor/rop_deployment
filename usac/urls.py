from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.custom_login, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    path('dilemma/', views.dilemma_view, name='dilemma'),
    path('history/', views.history_view, name='history'),
    path('history/export/', views.export_history_csv, name='export_history_csv'),
]
