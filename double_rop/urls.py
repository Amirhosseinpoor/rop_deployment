# double_rop/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="double_home"),       # http://localhost:8000/double/
    path("predict/", views.predict, name="double_predict"),
]
