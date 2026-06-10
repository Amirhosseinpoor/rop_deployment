# test_analysis/tasks.py
from celery import shared_task
from .models import HealthProfile
from .utils import format_profile_for_llm   # ✅ از utils ایمپورت کن
from . import ai_pipeline
# test_analysis/tasks.py

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded

@shared_task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 2, 'countdown': 15})
def generate_health_report(self, profile_id: int, selected_model: str):
    from .models import HealthProfile
    from .utils import format_profile_for_llm
    from . import ai_pipeline

    try:
        profile = HealthProfile.objects.get(id=profile_id)
        profile_text = format_profile_for_llm(profile)
        final_report = ai_pipeline.run_health_analysis_pipeline(profile_text, selected_model)

        profile.llm_advice = final_report
        profile.model_used_for_advice = selected_model
        profile.report_ready = True
        profile.report_error = None
        profile.save(update_fields=['llm_advice','model_used_for_advice','report_ready','report_error'])
        return {"ok": True}

    except SoftTimeLimitExceeded as e:
        p = HealthProfile.objects.filter(id=profile_id).first()
        if p:
            p.report_ready = False
            p.report_error = "زمان پردازش به پایان رسید."
            p.save(update_fields=['report_ready','report_error'])
        raise

    except Exception as e:
        p = HealthProfile.objects.filter(id=profile_id).first()
        if p:
            p.report_ready = False
            p.report_error = str(e)
            p.save(update_fields=['report_ready','report_error'])
        raise
