import os
import sys

from django.apps import AppConfig


class DoubleRopConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'double_rop'

    def ready(self):
        """Index the KC knowledge base once when the server starts."""
        argv = sys.argv
        is_manage_server = any(c in argv for c in ("runserver", "runserver_plus"))
        is_manage_cmd = (len(argv) > 1 and argv[0].endswith("manage.py"))

        if is_manage_server:
            if "--noreload" not in argv and os.environ.get("RUN_MAIN") != "true":
                return
        elif is_manage_cmd:
            return

        try:
            from . import chat_service
            chat_service.start_background_index()
        except Exception:
            import logging
            logging.getLogger("kc.chat").exception("KB | startup indexing hook failed")
