# doctors_marketplace/signals.py
import os
from django.core.files.base import ContentFile
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Doctor, DoctorKnowledge

@receiver(post_save, sender=Doctor)
def init_doctor_kb(sender, instance: Doctor, created, **kwargs):
    if not created:
        return
    # Ensure vector dir and KB dir exist
    instance.vector_dir()  # creates vectors dir
    kb_dir = os.path.join("doctor_knowledge", instance.slug)
    os.makedirs(os.path.join(instance._meta.apps.get_app_config('doctors_marketplace').path, '..', '..'), exist_ok=True)  # safety

    # Create a default README KB item so the folder exists in storage
    if not DoctorKnowledge.objects.filter(doctor=instance, title="README").exists():
        dk = DoctorKnowledge(doctor=instance, title="README")
        dk.file.save("README.txt", ContentFile(
            "توضیحات: فایل‌های دانش این پزشک را اینجا آپلود کنید.\n"
            "هر بار آپلود، ایندکس RAG به‌صورت خودکار به‌روزرسانی می‌شود."
        ))
        dk.save()

# doctors_marketplace/signals.py (append)
from .tasks import build_kb_item
from django.db.models.signals import post_save

@receiver(post_save, sender=DoctorKnowledge)
def enqueue_embedding(sender, instance: DoctorKnowledge, created, **kwargs):
    # On every save (upload or change), rebuild/merge
    build_kb_item.delay(instance.id)
