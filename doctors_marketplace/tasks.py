# doctors_marketplace/tasks.py
from celery import shared_task
from .models import DoctorKnowledge
from .services.rag import index_file_for_doctor
from django.utils import timezone
@shared_task(bind=True, max_retries=3, default_retry_delay=30)
def build_kb_item(self, dk_id: int):
    try:
        dk = DoctorKnowledge.objects.select_related("doctor").get(pk=dk_id)
    except DoctorKnowledge.DoesNotExist:
        return {"ok": False, "error": "not found"}

    ok, msg = index_file_for_doctor(dk.doctor, dk.file.path, dk.title)
    if ok:
        dk.title = dk.title  # dummy save if needed
        dk.save(update_fields=[])
        return {"ok": True, "msg": f"✅ Embedding done at {timezone.now():%H:%M}"}
    else:
        return {"ok": False, "msg": f"⚠️ Failed embedding: {msg}"}