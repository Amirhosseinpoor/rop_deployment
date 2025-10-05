# doctors_marketplace/models.py
import uuid
from django.conf import settings
from django.db import models
from django.utils.text import slugify
import os, re, unicodedata
from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver


def slugify_filename(name: str) -> str:
    # Keep Persian letters, normalize, remove weird whitespace
    name = unicodedata.normalize("NFKC", name).strip()
    # Collapse spaces and forbidden fs chars
    name = re.sub(r"[\\/:*?\"<>|]+", "-", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def kb_upload_to(instance, filename):
    # store under: doctor_knowledge/<doctor-slug>/<title>-<orig>
    doctor = instance.doctor
    folder = f"doctor_knowledge/{doctor.slug}/"
    base, ext = os.path.splitext(filename)
    title = slugify_filename(instance.title or "doc")
    orig  = slugify_filename(base)
    final = f"{title}-{orig}{ext}".strip("-")
    return os.path.join(folder, final)

class Doctor(models.Model):
    class Specialization(models.TextChoices):
        KIDNEY_STONE = 'kidney_stone', 'Kidney Stone (Urology)'
        HEART_DISEASE = 'heart_disease', 'Heart Disease (Cardiology)'
        DIABETES = 'diabetes', 'Diabetes (Endocrinology)'
        CKD = 'ckd', 'Chronic Kidney Disease'
        CARDIOVASCULAR = 'cardiovascular', 'Cardiovascular'
        ANEMIA = 'anemia', 'Anemia (Hematology)'
        HYPERTENSION = 'hypertension', 'Hypertension'
        FAMILY_COUNSELOR = 'family_counselor', 'Family Counselor'

    class Persona(models.TextChoices):
        KIND = 'kind', 'Kind & Patient'
        IN_HURRY = 'in_hurry', 'Efficient & To-the-point'
        CALM = 'calm', 'Calm & Reassuring'
        ANALYTICAL = 'analytical', 'Analytical'

    id = models.BigAutoField(primary_key=True)
    # English (primary)
    name = models.CharField(max_length=120)
    slug = models.SlugField(max_length=140, unique=True, blank=True)
    specialization = models.CharField(max_length=40, choices=Specialization.choices)
    persona = models.CharField(max_length=20, choices=Persona.choices, default=Persona.KIND)
    headline = models.CharField(max_length=200, blank=True)
    bio = models.TextField(blank=True)

    # Persian
    name_fa = models.CharField(max_length=140, blank=True, default='')
    specialization_fa = models.CharField(max_length=140, blank=True, default='')
    headline_fa = models.CharField(max_length=220, blank=True, default='')
    bio_fa = models.TextField(blank=True, default='')
    tags_fa = models.TextField(blank=True, default='')  # comma-separated

    # NEW: long system prompt (used to seed chat)
    system_prompt = models.TextField(blank=True, default='')

    avatar = models.ImageField(upload_to='doctor_avatars/', blank=True, null=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    def vector_dir(self):
        root = os.getenv("VECTOR_ROOT", os.path.join(settings.MEDIA_ROOT, "doctor_vectors"))
        path = os.path.join(root, self.slug)
        os.makedirs(path, exist_ok=True)
        return path
    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name or self.name_fa)
        super().save(*args, **kwargs)

    def __str__(self):
        title = self.name_fa or self.name
        spec = self.specialization_fa or self.get_specialization_display()
        return f"{title} — {spec}"

class ChatSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='dm_sessions')
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE, related_name='sessions')
    title = models.CharField(max_length=160, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user} ↔ {self.doctor} ({self.created_at:%Y-%m-%d})"

class ChatMessage(models.Model):
    class Role(models.TextChoices):
        SYSTEM = 'system', 'System'
        USER = 'user', 'User'
        ASSISTANT = 'assistant', 'Assistant'

    id = models.BigAutoField(primary_key=True)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=Role.choices)
    content = models.TextField()
    tokens = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

# NEW: per-doctor RAG file
class DoctorKnowledge(models.Model):
    id = models.BigAutoField(primary_key=True)
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE, related_name='knowledge_items')
    title = models.CharField(max_length=200)
    file = models.FileField(upload_to=kb_upload_to)  # pdf/txt/etc.
    created_at = models.DateTimeField(auto_now_add=True)
    def save(self, *args, **kwargs):
        # Force renaming pattern on re-save as well
        # (Django will place in upload_to; no extra rename needed)
        super().save(*args, **kwargs)
    def __str__(self):
        return f"{self.title} · {self.doctor}"
