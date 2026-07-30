# double_rop/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="double_home"),       # http://localhost:8000/double/
    path("predict/", views.predict, name="double_predict"),
    path("notes/", views.save_notes, name="kc_notes"),

    # KC chat assistant (served under /kc/…)
    path("chat", views.chat, name="kc_chat"),
    path("chat/", views.chat),
    path("chat/stream", views.chat_stream, name="kc_chat_stream"),
    path("chat/stream/", views.chat_stream),
    path("chat/upload", views.chat_upload, name="kc_chat_upload"),
    path("chat/transcribe", views.chat_transcribe, name="kc_chat_transcribe"),
]
