"""
Rebuild every doctor's FAISS vector index from their knowledge files using the
current embedding model.

Run this once after switching embedding models (e.g. local BGE -> GAPGPT
text-embedding-3-large) so old, dimension-incompatible indexes are replaced:

    python manage.py reindex_kb
    python manage.py reindex_kb --slug kidney-stone
"""
from django.core.management.base import BaseCommand

from doctors_marketplace.models import Doctor
from doctors_marketplace.services.rag import rebuild_doctor_index


class Command(BaseCommand):
    help = "Rebuild per-doctor RAG indexes from their knowledge files."

    def add_arguments(self, parser):
        parser.add_argument('--slug', help="Reindex only this doctor slug.")

    def handle(self, *args, **opts):
        qs = Doctor.objects.all()
        if opts.get('slug'):
            qs = qs.filter(slug=opts['slug'])

        if not qs.exists():
            self.stdout.write(self.style.WARNING("No matching doctors."))
            return

        for doctor in qs:
            self.stdout.write(f"→ {doctor.slug} … ", ending="")
            try:
                ok, msg = rebuild_doctor_index(doctor)
            except Exception as e:  # noqa: BLE001
                ok, msg = False, str(e)
            style = self.style.SUCCESS if ok else self.style.WARNING
            self.stdout.write(style(msg))
