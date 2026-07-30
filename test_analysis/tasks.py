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


@shared_task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 1, 'countdown': 15})
def analyze_eye_image_task(self, eye_image_id: int):
    """Run the eye segmentation + anemia classification pipeline in the background."""
    from .models import EyeImage
    from .services.eye_pipeline import analyze_eye_image

    eye_image = EyeImage.objects.filter(id=eye_image_id).first()
    if not eye_image:
        return {"ok": False, "reason": "eye_image not found"}
    analyze_eye_image(eye_image)  # best-effort, never raises — persists its own status
    return {"ok": True}


@shared_task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 1, 'countdown': 15})
def extract_medical_test_task(self, medical_test_id: int):
    """Run the LLM medical-test extraction pipeline in the background."""
    from .models import MedicalTest
    from .services.medical_test_extraction import extract_medical_test

    mt = MedicalTest.objects.filter(id=medical_test_id).first()
    if not mt:
        return {"ok": False, "reason": "medical_test not found"}
    extract_medical_test(mt)  # best-effort, never raises — persists its own status
    return {"ok": True}
