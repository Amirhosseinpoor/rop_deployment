import os
import sys

from django.apps import AppConfig


class SingleRopConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'single_rop'

    def ready(self):
        """
        Index the ROP knowledge base once, when the server starts.

        Guards:
        * Only for `runserver` (and gunicorn/uvicorn WSGI boots) — never during
          migrations, shell, tests, collectstatic, etc.
        * Under the autoreloader, only the worker process (RUN_MAIN=="true")
          indexes, so the KB is not built twice.
        """
        argv = sys.argv
        is_manage_server = any(c in argv for c in ("runserver", "runserver_plus"))
        is_manage_cmd = os.path.basename(argv[0] if argv else "") == "manage.py" \
            or (len(argv) > 1 and argv[0].endswith("manage.py"))

        if is_manage_server:
            # Skip the autoreload supervisor; only its worker sets RUN_MAIN=true.
            if "--noreload" not in argv and os.environ.get("RUN_MAIN") != "true":
                return
        elif is_manage_cmd:
            # Some other management command (migrate, shell, test…) — do nothing.
            return
        # else: WSGI/ASGI server boot (gunicorn/uvicorn) — index as well.

        try:
            from . import chat_service
            chat_service.start_background_index()
        except Exception:  # never let indexing break app startup
            import logging
            logging.getLogger("rop.chat").exception(
                "KB | startup indexing hook failed")
