# doctors_marketplace/management/commands/deliver_due_messages.py
"""Deliver due WhatsApp reminders / follow-ups via WAHA.

Run on a schedule (system cron), e.g. every minute:
    * * * * * cd /path && venv/bin/python manage.py deliver_due_messages
"""
from django.core.management.base import BaseCommand
from django.utils import timezone

from doctors_marketplace.models import ScheduledMessage
from doctors_marketplace.services.reminders import deliver_due


class Command(BaseCommand):
    help = "Send due WhatsApp reminders/follow-ups (ScheduledMessage) via WAHA."

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=100)
        parser.add_argument("--dry-run", action="store_true",
                            help="List due messages without sending.")

    def handle(self, *args, **opts):
        if opts["dry_run"]:
            now = timezone.now()
            due = ScheduledMessage.objects.filter(
                status=ScheduledMessage.Status.PENDING, send_at__lte=now).order_by("send_at")
            for m in due[:opts["limit"]]:
                self.stdout.write(f"[due] #{m.pk} {m.kind} -> {m.phone}: {m.text[:60]}")
            self.stdout.write(f"{due.count()} due.")
            return
        sent, failed, total = deliver_due(limit=opts["limit"])
        self.stdout.write(f"Delivered {sent}, failed {failed}, of {total} due.")
