# doctors_marketplace/signals.py
"""
Signals that keep each doctor's RAG index in sync with their knowledge files.

Indexing runs in a daemon thread instead of Celery so the whole feature works
with a plain `python manage.py runserver` — no Redis/worker required. If
`DM_USE_CELERY=1` is set and a broker is reachable, it offloads to Celery
instead.
"""
import os
import threading
import logging

from django.core.files.base import ContentFile
from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import Doctor, DoctorKnowledge

log = logging.getLogger(__name__)


@receiver(post_save, sender=Doctor)
def init_doctor_kb(sender, instance: Doctor, created, **kwargs):
    if not created:
        return
    instance.vector_dir()  # ensure the per-doctor vector dir exists

    # Seed a README so the storage folder exists and the studio isn't empty.
    if not DoctorKnowledge.objects.filter(doctor=instance, title="README").exists():
        dk = DoctorKnowledge(doctor=instance, title="README")
        dk.file.save("README.txt", ContentFile(
            "توضیحات: فایل‌های دانش این پزشک را اینجا آپلود کنید.\n"
            "هر بار آپلود، ایندکس RAG به‌صورت خودکار به‌روزرسانی می‌شود."
        ))
        dk.save()


def _index_in_thread(dk_id: int):
    """Index a single knowledge item in the background (own DB/file access)."""
    from .models import DoctorKnowledge
    from .services.rag import index_file_for_doctor
    try:
        dk = DoctorKnowledge.objects.select_related("doctor").get(pk=dk_id)
    except DoctorKnowledge.DoesNotExist:
        return
    try:
        ok, msg = index_file_for_doctor(dk.doctor, dk.file.path, dk.title)
        log.info("KB index for %s: ok=%s msg=%s", dk.doctor.slug, ok, msg)
    except Exception:  # noqa: BLE001
        log.exception("KB indexing failed for DoctorKnowledge #%s", dk_id)
    finally:
        from django.db import connection
        connection.close()


@receiver(post_save, sender=DoctorKnowledge)
def enqueue_embedding(sender, instance: DoctorKnowledge, created, **kwargs):
    # The placeholder README carries no real knowledge — skip indexing it.
    if instance.title == "README":
        return

    if os.getenv("DM_USE_CELERY") == "1":
        try:
            from .tasks import build_kb_item
            build_kb_item.delay(instance.id)
            return
        except Exception:  # noqa: BLE001 - broker down -> fall back to thread
            log.warning("Celery unavailable, indexing inline in a thread instead")

    threading.Thread(
        target=_index_in_thread, args=(instance.id,), daemon=True
    ).start()
