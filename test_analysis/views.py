from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.urls import reverse  # ✅ اضافه شد
from .forms import HealthProfileForm, PreviousJobFormSet, ReferralFormSet
from .models import HealthProfile, PreviousJob, Referral
import markdown2
from . import ai_pipeline
from .tasks import generate_health_report
from django.http import JsonResponse
from django.views.decorators.http import require_GET  # ✅ فقط این را بگیر


def get_llm_advice(profile_text, selected_model):
    try:
        final_report = ai_pipeline.run_health_analysis_pipeline(profile_text, selected_model)
        return final_report
    except Exception as e:
        print(f"🔥 CRITICAL ERROR calling the AI pipeline: {e}")
        return "متاسفانه در حال حاضر به دلیل یک خطای داخلی، امکان تولید گزارش وجود ندارد. لطفا بعداً دوباره تلاش کنید."


@login_required
def create_or_update_health_profile(request):
    """
    Handles both creating/updating the profile and its related jobs and referrals.
    """
    try:
        instance = HealthProfile.objects.get(user=request.user)
    except HealthProfile.DoesNotExist:
        instance = None

    if request.method == 'POST':
        form = HealthProfileForm(request.POST, request.FILES, instance=instance)
        job_formset = PreviousJobFormSet(request.POST, prefix='jobs', queryset=PreviousJob.objects.filter(
            profile=instance) if instance else PreviousJob.objects.none())
        referral_formset = ReferralFormSet(request.POST, prefix='referrals', queryset=Referral.objects.filter(
            profile=instance) if instance else Referral.objects.none())

        if form.is_valid() and job_formset.is_valid() and referral_formset.is_valid():
            profile = form.save(commit=False)
            profile.user = request.user
            profile.save()

            jobs = job_formset.save(commit=False)
            for job in jobs:
                job.profile = profile
                job.save()
            job_formset.save_m2m()
            for obj in job_formset.deleted_objects:  # ✅
                obj.delete()

            referrals = referral_formset.save(commit=False)
            for referral in referrals:
                referral.profile = profile
                referral.save()
            referral_formset.save_m2m()
            for obj in referral_formset.deleted_objects:  # ✅
                obj.delete()

            # Generate the text summary from the saved profile
            # profile_text_for_llm = format_profile_for_llm(profile)
            # selected_model = request.POST.get('selected_model', 'cloud_gpt')
            # profile.model_used_for_advice = selected_model
            # # Call our NEW, powerful pipeline
            # advice = get_llm_advice(profile_text_for_llm, selected_model)
            #
            # # Save the final report to the profile
            # profile.llm_advice = advice
            # profile.save()

            # test_analysis/views.py (درون create_or_update_health_profile، بخش POST و valid)

            selected_model = request.POST.get('selected_model', 'cloud_gpt')

            profile.report_ready = False
            profile.report_error = None
            profile.model_used_for_advice = selected_model
            profile.save()

            try:
                task = generate_health_report.apply_async(
                    args=[profile.id, selected_model],
                    ignore_result=True  # تأکید بر عدم انتظار نتیجه
                )
                profile.report_task_id = task.id or ''
                profile.save(update_fields=['report_task_id'])
            except Exception as e:
                # لاگ و Fail-safe: باز هم می‌فرستیم صفحه پردازش تا پیام خطا را ببینند
                profile.report_error = f"Celery enqueue failed: {e}"
                profile.report_ready = False
                profile.save(update_fields=['report_error', 'report_ready'])

            return redirect('health_processing')

        else:
            # Your debugging code for form errors
            print("\n--- FORM VALIDATION FAILED ---")
            if form.errors: print("Main form errors:", form.errors)
            if job_formset.errors: print("Job Formset errors:", job_formset.errors)
            if referral_formset.errors: print("Referral Formset errors:", referral_formset.errors)
            print("------------------------------\n")
    else:
        form = HealthProfileForm(instance=instance)
        job_formset = PreviousJobFormSet(prefix='jobs', queryset=PreviousJob.objects.filter(
            profile=instance) if instance else PreviousJob.objects.none())
        referral_formset = ReferralFormSet(prefix='referrals', queryset=Referral.objects.filter(
            profile=instance) if instance else Referral.objects.none())

    context = {
        'form': form,
        'job_formset': job_formset,
        'referral_formset': referral_formset
    }
    return render(request, 'test_analysis/profile_form.html', context)


@login_required
def profile_detail_view(request):
    """
    Displays the complete profile, including the final report.
    """
    profile = get_object_or_404(HealthProfile, user=request.user)
    # The final report from the pipeline might already contain markdown
    html_advice = markdown2.markdown(profile.llm_advice or "No advice generated.")

    context = {
        'profile': profile,
        'html_advice': html_advice,
    }
    return render(request, 'test_analysis/profile_detail.html', context)


@login_required
def processing_page(request):
    # فقط صفحه‌ای که مودال را نشان می‌دهد
    profile = get_object_or_404(HealthProfile, user=request.user)
    return render(request, 'test_analysis/processing.html', {'profile': profile})


@login_required
@require_GET
def report_status(request):
    profile = get_object_or_404(HealthProfile, user=request.user)
    return JsonResponse({
        "ready": profile.report_ready,
        "error": bool(profile.report_error),
        "error_msg": profile.report_error or "",
        "detail_url": request.build_absolute_uri(
            # صفحه‌ی گزارش موجود خودت
            reverse('profile_detail')
        ),
    })


@login_required
def minigame_page(request):
    return render(request, 'test_analysis/minigame.html', {})
