from django.apps import AppConfig


class DoctorsMarketplaceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'doctors_marketplace'
    def ready(self):
        from . import signals
        # Auto-deliver due WhatsApp reminders/follow-ups without needing a cron.
        try:
            from .services.reminders import start_reminder_worker
            start_reminder_worker()
        except Exception:  # noqa: BLE001
            pass
