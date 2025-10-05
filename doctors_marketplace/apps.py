from django.apps import AppConfig


class DoctorsMarketplaceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'doctors_marketplace'
    def ready(self):
        from . import signals
