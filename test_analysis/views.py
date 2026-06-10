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
from .forms import EmployeeProfileForm, DoctorNotesForm
from django.contrib.auth.models import User
from usac.models import UserProfile
from usac.views import is_manager
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.urls import reverse
from django.http import JsonResponse, HttpResponseForbidden
from django.views.decorators.http import require_GET
from django.contrib.auth.models import User

from .forms import (
    EmployeeProfileForm, DoctorNotesForm,
    PreviousJobFormSet, ReferralFormSet,
    HealthProfileForm,  # still needed if you use it elsewhere
)
from .models import HealthProfile, PreviousJob, Referral
from .tasks import generate_health_report
from . import ai_pipeline
import markdown2

# Import the is_manager check from usac.views
from usac.views import is_manager


def _can_employee_edit(profile):
    """Return True if the employee can still edit their profile (no doctor notes yet)."""
    if not profile:
        return True
    doctor_fields = [
        'general_exam_notes', 'eye_exam_notes', 'skin_hair_nails_exam_notes',
        'ent_mouth_exam_notes', 'head_neck_exam_notes', 'lung_exam_notes',
        'cardiovascular_exam_notes', 'abdomen_pelvis_exam_notes',
        'urinary_system_exam_notes', 'musculoskeletal_exam_notes',
        'nervous_system_exam_notes', 'mental_health_exam_notes',
        'opinion_fit', 'opinion_fit_with_conditions', 'opinion_unfit',
        'opinion_fit_conditions_details', 'opinion_unfit_reason',
        'medical_recommendations'
    ]
    for field in doctor_fields:
        value = getattr(profile, field, None)
        if field in ('opinion_fit', 'opinion_fit_with_conditions', 'opinion_unfit'):
            if value:  # bool True
                return False
        elif value:  # non-empty string etc.
            return False
    return True


def get_llm_advice(profile_text, selected_model):
    try:
        final_report = ai_pipeline.run_health_analysis_pipeline(profile_text, selected_model)
        return final_report
    except Exception as e:
        print(f"🔥 CRITICAL ERROR calling the AI pipeline: {e}")
        return "متاسفانه در حال حاضر به دلیل یک خطای داخلی، امکان تولید گزارش وجود ندارد. لطفا بعداً دوباره تلاش کنید."


@login_required
def create_or_update_health_profile(request):
    # Only employees (or doctors) can create their own profile
    if request.user.profile.role not in [UserProfile.ROLE_EMPLOYEE, UserProfile.ROLE_DOCTOR]:
        messages.error(request, "شما مجاز به ایجاد پرونده سلامت نیستید.")
        return redirect('dilemma')

    instance = HealthProfile.objects.filter(user=request.user).first()
    # If profile exists and no explicit edit request, redirect to detail
    if instance and not request.GET.get('edit'):
        return redirect('profile_detail')

    # If doctor has already started work, lock the form
    if instance and not _can_employee_edit(instance):
        messages.error(request, "پزشک یادداشت‌های خود را ثبت کرده است. شما نمی‌توانید پرونده را ویرایش کنید.")
        return redirect('profile_detail')

    if request.method == 'POST':
        form = EmployeeProfileForm(request.POST, request.FILES, instance=instance)
        job_formset = PreviousJobFormSet(request.POST, prefix='jobs',
                                         queryset=PreviousJob.objects.filter(
                                             profile=instance) if instance else PreviousJob.objects.none())
        if form.is_valid() and job_formset.is_valid():
            profile = form.save(commit=False)
            profile.user = request.user
            profile.save()

            jobs = job_formset.save(commit=False)
            for job in jobs:
                job.profile = profile
                job.save()
            job_formset.save_m2m()
            for obj in job_formset.deleted_objects:
                obj.delete()

            selected_model = request.POST.get('selected_model', 'cloud_gpt')
            profile.report_ready = False
            profile.report_error = None
            profile.model_used_for_advice = selected_model
            profile.save()

            try:
                task = generate_health_report.apply_async(
                    args=[profile.id, selected_model],
                    ignore_result=True
                )
                profile.report_task_id = task.id or ''
                profile.save(update_fields=['report_task_id'])
            except Exception as e:
                profile.report_error = f"Celery enqueue failed: {e}"
                profile.report_ready = False
                profile.save(update_fields=['report_error', 'report_ready'])

            return redirect('health_processing')
    else:
        form = EmployeeProfileForm(instance=instance)
        job_formset = PreviousJobFormSet(prefix='jobs',
                                         queryset=PreviousJob.objects.filter(
                                             profile=instance) if instance else PreviousJob.objects.none())

    context = {
        'form': form,
        'job_formset': job_formset,
        'referral_formset': None,
    }
    return render(request, 'test_analysis/profile_form2.html', context)


# test_analysis/views.py
from django.contrib.auth.models import User
from django.http import HttpResponseForbidden
from usac.models import UserProfile  # for role check


@login_required
def profile_detail_view(request, user_id=None):
    # Determine target user
    if user_id and request.user.id != user_id:
        # Manager can view any employee in company (keep existing check)
        # Doctor can view employees in their company
        if request.user.profile.role == UserProfile.ROLE_MANAGER:
            target_user = get_object_or_404(User, pk=user_id, profile__company__manager=request.user)
        elif request.user.profile.role == UserProfile.ROLE_DOCTOR:
            # Doctor must be in the same company as target
            doctor_company = request.user.profile.company
            target_user = get_object_or_404(User, pk=user_id, profile__company=doctor_company)
        else:
            return HttpResponseForbidden("دسترسی ندارید.")
    else:
        target_user = request.user
        user_id = request.user.id

    profile = HealthProfile.objects.filter(user=target_user).first()
    if not profile:
        # Employee hasn't filled their profile yet – show a simple page
        return render(request, 'test_analysis/profile_detail2.html', {
            'profile': None,
            'html_advice': None,
            'viewed_user': target_user,
            'doctor_form': None,
            'referral_formset': None,
            'can_edit_doctor_notes': False,
            'can_edit_employee': False,
        })

    # Determine if doctor notes can be edited by the current user
    can_edit_doctor_notes = False
    if request.user.profile.role == UserProfile.ROLE_DOCTOR and request.user.id != target_user.id:
        if request.user.profile.company == target_user.profile.company:
            can_edit_doctor_notes = True
    elif request.user.profile.role == UserProfile.ROLE_MANAGER:
        can_edit_doctor_notes = True  # manager can edit doctor notes of anyone in company
    # Can the employee themselves edit this profile?
    can_edit_employee = False
    if request.user == target_user and profile and _can_employee_edit(profile):
        can_edit_employee = True

    if request.method == 'POST' and can_edit_doctor_notes:
        doctor_form = DoctorNotesForm(request.POST, instance=profile)
        referral_formset = ReferralFormSet(request.POST, prefix='referrals',
                                           queryset=Referral.objects.filter(profile=profile))
        if doctor_form.is_valid() and referral_formset.is_valid():
            profile = doctor_form.save(commit=False)
            profile.examining_doctor = request.user  # track who examined
            profile.save()

            referrals = referral_formset.save(commit=False)
            for ref in referrals:
                ref.profile = profile
                ref.save()
            referral_formset.save_m2m()
            for obj in referral_formset.deleted_objects:
                obj.delete()
            if request.user.profile.role == UserProfile.ROLE_MANAGER:
                messages.success(request, "یادداشت‌های مدیر با موفقیت ذخیره شد.")
            else:
                messages.success(request, "یادداشت‌های پزشک با موفقیت ذخیره شد.")
            if request.user.profile.role == UserProfile.ROLE_DOCTOR:
                return redirect('doctor_dashboard')
            elif request.user.profile.role == UserProfile.ROLE_MANAGER:
                return redirect('manager_dashboard')
            return redirect('profile_detail_user', user_id=target_user.id)
        # if invalid, fall through to render with errors
    else:
        doctor_form = DoctorNotesForm(instance=profile) if can_edit_doctor_notes else None
        referral_formset = ReferralFormSet(prefix='referrals',
                                           queryset=Referral.objects.filter(
                                               profile=profile)) if can_edit_doctor_notes else None

    html_advice = markdown2.markdown(profile.llm_advice or "گزارشی تولید نشده است.")
    context = {
        'profile': profile,
        'html_advice': html_advice,
        'viewed_user': target_user,
        'doctor_form': doctor_form,
        'referral_formset': referral_formset,
        'can_edit_doctor_notes': can_edit_doctor_notes,
        'can_edit_employee': can_edit_employee,  # <-- new line
        'is_manager': request.user.profile.role == UserProfile.ROLE_MANAGER,
    }
    return render(request, 'test_analysis/profile_detail2.html', context)


@login_required
def processing_page(request):
    # فقط صفحه‌ای که مودال را نشان می‌دهد
    profile = get_object_or_404(HealthProfile, user=request.user)
    return render(request, 'test_analysis/processing2.html', {'profile': profile})


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


@login_required
def doctor_dashboard(request):
    # Only doctors can access
    if request.user.profile.role != UserProfile.ROLE_DOCTOR:
        return HttpResponseForbidden("دسترسی ندارید.")
    doctor_company = request.user.profile.company
    employees = User.objects.filter(
        profile__company=doctor_company,
        profile__role=UserProfile.ROLE_EMPLOYEE
    ).select_related('profile')

    # Stats: how many employees examined by this doctor?
    examined_count = HealthProfile.objects.filter(examining_doctor=request.user).count()
    total_employees = employees.count()
    completed_exams = HealthProfile.objects.filter(
        user__in=employees,
        opinion_fit__isnull=False  # at least one opinion filled
    ).exclude(opinion_fit=False, opinion_fit_with_conditions=False, opinion_unfit=False).count()

    context = {
        'employees': employees,
        'examined_count': examined_count,
        'total_employees': total_employees,
        'completed_exams': completed_exams,
    }
    return render(request, 'test_analysis/doctor_dashboard.html', context)


@login_required
@user_passes_test(is_manager)
def manager_edit_profile(request, user_id):
    target_user = get_object_or_404(User, pk=user_id, profile__company__manager=request.user)
    profile, created = HealthProfile.objects.get_or_create(user=target_user)

    if request.method == 'POST':
        # use the full HealthProfileForm (or EmployeeProfileForm + DoctorNotesForm combined)
        # For simplicity, we can use the same multi-step form but without restrictions.
        form = EmployeeProfileForm(request.POST, request.FILES, instance=profile)
        job_formset = PreviousJobFormSet(request.POST, prefix='jobs',
                                         queryset=PreviousJob.objects.filter(profile=profile))
        referral_formset = ReferralFormSet(request.POST, prefix='referrals',
                                           queryset=Referral.objects.filter(profile=profile))
        if form.is_valid() and job_formset.is_valid() and referral_formset.is_valid():
            profile = form.save(commit=False)
            profile.user = target_user
            profile.save()
            jobs = job_formset.save(commit=False)
            for job in jobs:
                job.profile = profile
                job.save()
            job_formset.save_m2m()
            for obj in job_formset.deleted_objects: obj.delete()
            referrals = referral_formset.save(commit=False)
            for ref in referrals:
                ref.profile = profile
                ref.save()
            referral_formset.save_m2m()
            for obj in referral_formset.deleted_objects: obj.delete()
            messages.success(request, "پرونده با موفقیت ویرایش شد.")
            return redirect('member_detail', user_id=target_user.id)
    else:
        form = EmployeeProfileForm(instance=profile)
        job_formset = PreviousJobFormSet(prefix='jobs', queryset=PreviousJob.objects.filter(profile=profile))
        referral_formset = ReferralFormSet(prefix='referrals', queryset=Referral.objects.filter(profile=profile))

    context = {
        'form': form,
        'job_formset': job_formset,
        'referral_formset': referral_formset,
        'target_user': target_user,
    }
    return render(request, 'test_analysis/manager_edit_profile.html', context)


import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from .health_chat_agent import chat_with_assistant


@csrf_exempt
@login_required
def health_chat_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    try:
        data = json.loads(request.body)
        message = data.get('message', '')
        history = data.get('history', [])

        # فراخوانی کانال چت با ساختار جدید خروجی
        reply, finder_results = chat_with_assistant(message, history)

        return JsonResponse({
            'reply': reply,
            'finder_results': finder_results,  # تغییر نام متغیر برای پوشش پزشک و داروخانه
            'tool_called': finder_results is not None
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
