# doctors_marketplace/services/reminders.py
"""Deliver due WhatsApp reminders/follow-ups.

`deliver_due()` is the single source of truth, shared by:
  * the `deliver_due_messages` management command (for system cron), and
  * a lightweight in-process background worker started at app ready, so reminders
    fire automatically under `runserver`/gunicorn without any extra cron setup.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time

from django.utils import timezone

log = logging.getLogger("doctors_marketplace.services.reminders")

MAX_ATTEMPTS = 3
POLL_SECONDS = 60


def deliver_due(limit: int = 100) -> tuple[int, int, int]:
    """Send all pending messages whose time has come. Returns (sent, failed, due)."""
    from ..models import ScheduledMessage
    from .waha import send_whatsapp, is_configured

    now = timezone.now()
    due = list(ScheduledMessage.objects.filter(
        status=ScheduledMessage.Status.PENDING, send_at__lte=now
    ).order_by("send_at")[:limit])
    if not due:
        return (0, 0, 0)
    if not is_configured():
        log.warning("REMINDERS | %d due but WAHA not configured — left pending.", len(due))
        return (0, 0, len(due))

    sent = failed = 0
    for m in due:
        ok, detail = send_whatsapp(m.phone, m.text)
        m.attempts += 1
        m.detail = detail[:250]
        if ok:
            m.status = ScheduledMessage.Status.SENT
            m.sent_at = timezone.now()
            sent += 1
        elif m.attempts >= MAX_ATTEMPTS:
            m.status = ScheduledMessage.Status.FAILED
            failed += 1
        m.save(update_fields=["attempts", "detail", "status", "sent_at"])
    if sent or failed:
        log.info("REMINDERS | delivered %d, failed %d, of %d due.", sent, failed, len(due))
    return (sent, failed, len(due))


_started = False


def start_reminder_worker():
    """Start a daemon thread that delivers due messages every POLL_SECONDS.

    Guards: never during one-off management commands; single instance under the
    runserver autoreloader (only the RUN_MAIN child)."""
    global _started
    if _started:
        return
    argv = " ".join(sys.argv)
    skip = ("makemigrations", "migrate", "collectstatic", "test", "shell",
            "createsuperuser", "deliver_due_messages", "dumpdata", "loaddata")
    if any(c in argv for c in skip):
        return
    # Under runserver's autoreloader, ready() runs in both the parent and the
    # child — only start in the child (RUN_MAIN) to avoid two workers. With
    # --noreload or under gunicorn there's a single process, so start normally.
    if "runserver" in argv and "--noreload" not in argv and os.environ.get("RUN_MAIN") != "true":
        return

    _started = True

    def _loop():
        log.info("REMINDERS | background delivery worker started (every %ss).", POLL_SECONDS)
        while True:
            try:
                deliver_due()
            except Exception:  # noqa: BLE001
                log.exception("REMINDERS | delivery loop error")
            time.sleep(POLL_SECONDS)

    threading.Thread(target=_loop, name="dm-reminders", daemon=True).start()
