# doctors_marketplace/management/commands/seed_doctors.py
from django.core.management.base import BaseCommand
from doctors_marketplace.models import Doctor
from doctors_marketplace.prompts import DOCTOR_DEFS

class Command(BaseCommand):
    help = "Seed default AI doctors (FA-first)."

    def handle(self, *args, **options):
        created, updated = 0, 0
        for d in DOCTOR_DEFS:
            obj, is_created = Doctor.objects.update_or_create(
                slug=d["slug"],
                defaults=dict(
                    name=d.get("name",""),
                    specialization=d["specialization"],
                    persona=d.get("persona","kind"),
                    headline=d.get("headline",""),
                    bio=d.get("bio",""),
                    name_fa=d.get("name_fa",""),
                    specialization_fa=d.get("specialization_fa",""),
                    headline_fa=d.get("headline_fa",""),
                    bio_fa=d.get("bio_fa",""),
                    tags_fa=d.get("tags_fa",""),
                    system_prompt=d.get("system",""),
                    is_active=True,
                ),
            )
            created += int(is_created)
            updated += int(not is_created)
        self.stdout.write(self.style.SUCCESS(f"پزشکان مقداردهی شدند. ایجاد: {created} | به‌روزرسانی: {updated}"))
