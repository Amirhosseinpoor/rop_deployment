# webapp/app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("predict/", views.predict, name="predict"),
    path("feedback/", views.submit_feedback, name="feedback"),
    path("notes/", views.save_notes, name="rop_notes"),
]
