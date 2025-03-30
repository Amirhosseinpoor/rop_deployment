from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),  # Home page
    path("predict/", views.predict, name="predict"),  # API endpoint for prediction
]