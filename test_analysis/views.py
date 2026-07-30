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

# ------------------------------------------------------------------
# Logging setup — verbose terminal logging for test_analysis.views.
# A dedicated StreamHandler is attached so log records are always
# printed to the terminal/stdout regardless of Django's LOGGING config.
# ------------------------------------------------------------------
import logging

logger = logging.getLogger("test_analysis.views")
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    ))
    logger.addHandler(_console_handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

logger.info("📦 test_analysis.views module imported and logger initialized")


def _can_employee_edit(profile):
    """Return True if the employee can still edit their profile (no doctor notes yet)."""
    logger.debug("🔎 _can_employee_edit() called | profile=%s", getattr(profile, 'pk', None))
    if not profile:
        logger.debug("   ↳ no profile instance → editable=True")
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
                logger.debug("   ↳ doctor opinion field '%s' is set → editable=False", field)
                return False
        elif value:  # non-empty string etc.
            logger.debug("   ↳ doctor note field '%s' is filled → editable=False", field)
            return False
    logger.debug("   ↳ no doctor fields filled → editable=True")
    return True


def get_llm_advice(profile_text, selected_model):
    logger.info("🤖 get_llm_advice() called | model=%s | profile_text_len=%d",
                selected_model, len(profile_text or ""))
    try:
        logger.debug("   ↳ invoking ai_pipeline.run_health_analysis_pipeline ...")
        final_report = ai_pipeline.run_health_analysis_pipeline(profile_text, selected_model)
        logger.info("   ✅ AI pipeline returned a report | report_len=%d", len(final_report or ""))
        return final_report
    except Exception as e:
        logger.exception("🔥 CRITICAL ERROR calling the AI pipeline: %s", e)
        print(f"🔥 CRITICAL ERROR calling the AI pipeline: {e}")
        return "متاسفانه در حال حاضر به دلیل یک خطای داخلی، امکان تولید گزارش وجود ندارد. لطفا بعداً دوباره تلاش کنید."


@login_required
def create_or_update_health_profile(request):
    logger.info("➡️  create_or_update_health_profile() | user=%s (id=%s) | method=%s | GET=%s",
                request.user.username, request.user.id, request.method, dict(request.GET))
    # Only employees (or doctors) can create their own profile
    if request.user.profile.role not in [UserProfile.ROLE_EMPLOYEE, UserProfile.ROLE_DOCTOR]:
        logger.warning("⛔ user=%s role=%s not allowed to create health profile → redirect dilemma",
                       request.user.username, request.user.profile.role)
        messages.error(request, "شما مجاز به ایجاد پرونده سلامت نیستید.")
        return redirect('dilemma')

    instance = HealthProfile.objects.filter(user=request.user).first()
    logger.debug("   ↳ existing HealthProfile instance=%s", getattr(instance, 'pk', None))
    # If profile exists and this is plain GET navigation (no edit flag), send the
    # user to their detail page. A POST is always a form submission — process it
    # so edits (and file uploads) are saved even though the form action URL does
    # not carry the ?edit flag.
    if instance and not request.GET.get('edit') and request.method != 'POST':
        logger.info("   ↳ profile exists & no edit flag (GET) → redirect to profile_detail")
        return redirect('profile_detail')

    # If doctor has already started work, lock the form
    if instance and not _can_employee_edit(instance):
        logger.warning("🔒 profile locked (doctor notes present) for user=%s → redirect profile_detail",
                       request.user.username)
        messages.error(request, "پزشک یادداشت‌های خود را ثبت کرده است. شما نمی‌توانید پرونده را ویرایش کنید.")
        return redirect('profile_detail')

    if request.method == 'POST':
        logger.info("   📨 POST received | files=%s | post_keys=%d",
                    list(request.FILES.keys()), len(request.POST))
        form = EmployeeProfileForm(request.POST, request.FILES, instance=instance)
        job_formset = PreviousJobFormSet(request.POST, prefix='jobs',
                                         queryset=PreviousJob.objects.filter(
                                             profile=instance) if instance else PreviousJob.objects.none())
        if form.is_valid() and job_formset.is_valid():
            logger.info("   ✅ form & job_formset valid → saving profile")
            profile = form.save(commit=False)
            profile.user = request.user
            profile.save()
            logger.debug("   ↳ HealthProfile saved | id=%s", profile.id)

            jobs = job_formset.save(commit=False)
            logger.debug("   ↳ saving %d previous-job rows", len(jobs))
            for job in jobs:
                job.profile = profile
                job.save()
            job_formset.save_m2m()
            for obj in job_formset.deleted_objects:
                logger.debug("   🗑️  deleting previous-job row id=%s", getattr(obj, 'pk', None))
                obj.delete()

            # Save uploaded eye photos + medical tests (duplicates skipped).
            # enqueue=False: the Deep Research thread screens the eyes and
            # extracts the tests itself, and only then raises report_ready — so
            # the segmentation, anemia screen and lab tables are guaranteed ready
            # on the detail page when the user is redirected there.
            _save_eye_images(profile, request.FILES.getlist('eye_images'), enqueue=False)
            _save_medical_tests(profile, request.FILES.getlist('medical_test_files'), enqueue=False)

            selected_model = request.POST.get('selected_model', 'cloud_gpt')
            logger.info("   🧠 selected_model=%s | resetting report flags", selected_model)
            profile.report_ready = False
            profile.report_error = None
            profile.model_used_for_advice = selected_model
            profile.save()

            # Kick off the Deep Health Research v2 pipeline (agentic, multimodal,
            # grounded). It runs in a background daemon thread, waits for the
            # eye/medical Celery pipelines to finish, then researches + writes the
            # componentized report and flips report_ready so the processing page
            # redirects to the detail view. No Celery needed for DR itself.
            try:
                from .services.deep_research import runner as dr_runner
                dr_runner.start(profile.id)
                logger.info("   ✅ Deep Research v2 generation started for profile=%s", profile.id)
            except Exception as e:  # noqa: BLE001
                logger.exception("   🔥 Deep Research start failed for profile_id=%s: %s", profile.id, e)
                profile.report_error = f"Deep Research start failed: {e}"
                profile.report_ready = False
                profile.save(update_fields=['report_error', 'report_ready'])

            # First-time submission → processing page (report + minigame).
            # Editing an existing profile → straight back to the detail page.
            if instance is not None:
                logger.info("   ↪️  edit of existing profile → redirecting to profile_detail")
                messages.success(request, "پرونده شما با موفقیت به‌روزرسانی شد.")
                return redirect('profile_detail')
            logger.info("   ↪️  first-time submission → redirecting to health_processing")
            return redirect('health_processing')
        else:
            logger.warning("   ❌ form invalid | form_errors=%s | formset_errors=%s",
                           form.errors.as_json(), job_formset.errors)
    else:
        logger.debug("   ↳ GET → rendering empty/prefilled form")
        form = EmployeeProfileForm(instance=instance)
        job_formset = PreviousJobFormSet(prefix='jobs',
                                         queryset=PreviousJob.objects.filter(
                                             profile=instance) if instance else PreviousJob.objects.none())

    context = {
        'form': form,
        'job_formset': job_formset,
        'referral_formset': None,
    }
    logger.debug("   🖼️  rendering profile_form2.html")
    return render(request, 'test_analysis/profile_form2.html', context)


def _file_hash(f):
    """SHA-256 of an uploaded file's content, leaving the read pointer at 0."""
    import hashlib
    h = hashlib.sha256()
    for chunk in f.chunks():
        h.update(chunk)
    try:
        f.seek(0)
    except Exception:
        pass
    return h.hexdigest()


def _save_medical_tests(profile, test_files, enqueue=True):
    """Store uploaded medical-test documents (and optionally queue extraction).

    Duplicate uploads (the same file content already on this profile, whether
    re-selected or re-submitted across multiple edits) are skipped so each test
    is only extracted once. The LLM extraction is heavy, so it never runs in the
    web request:
      • ``enqueue=True``  → a Celery task extracts it in the background
        (used by the owner's add-photo / manager edit flows).
      • ``enqueue=False`` → only the rows are created; the Deep Research thread
        extracts them itself before it raises ``report_ready`` (the form-submit
        flow), so the detail page always has the tables ready on redirect.
    """
    if not test_files:
        return 0
    from .models import MedicalTest
    logger.info("   🧪 saving %d medical-test file(s) for profile=%s (enqueue=%s)",
                len(test_files), profile.id, enqueue)
    saved = 0
    for tf in test_files:
        h = _file_hash(tf)
        if MedicalTest.objects.filter(profile=profile, content_hash=h).exists():
            logger.info("   ⏭️  skipping duplicate medical test (hash=%s…) for profile=%s", h[:12], profile.id)
            continue
        mt = MedicalTest.objects.create(profile=profile, file=tf, content_hash=h)
        if enqueue:
            from .tasks import extract_medical_test_task
            extract_medical_test_task.apply_async(args=[mt.id], ignore_result=True)
        saved += 1
    return saved


def _save_eye_images(profile, image_files, enqueue=True):
    """Store uploaded eye photos (and optionally queue the screening pipeline).

    Duplicate uploads (same file content already on this profile) are skipped so
    the segmentation/anemia pipeline isn't re-run on an image we've already
    processed. Inference is heavy, so it never runs in the web request:
      • ``enqueue=True``  → a Celery task screens it in the background.
      • ``enqueue=False`` → only the rows are created; the Deep Research thread
        screens them itself before it raises ``report_ready`` (the form-submit
        flow), so the segmentation + anemia results are on the detail page when
        the user is redirected.
    """
    if not image_files:
        return 0
    from .models import EyeImage
    logger.info("   👁️  saving %d eye image(s) for profile=%s (enqueue=%s)",
                len(image_files), profile.id, enqueue)
    saved = 0
    for ef in image_files:
        h = _file_hash(ef)
        if EyeImage.objects.filter(profile=profile, content_hash=h).exists():
            logger.info("   ⏭️  skipping duplicate eye image (hash=%s…) for profile=%s", h[:12], profile.id)
            continue
        eye_img = EyeImage.objects.create(profile=profile, image=ef, content_hash=h)
        if enqueue:
            from .tasks import analyze_eye_image_task
            analyze_eye_image_task.apply_async(args=[eye_img.id], ignore_result=True)
        saved += 1
    return saved


@login_required
def upload_eye_scan(request):
    """Owner uploads eye photo(s) straight from their profile-detail page.

    Each photo is stored and run through the two-phase segmentation + anemia
    pipeline, then we redirect back to the detail page so the fresh results
    render inline — no full form / report round-trip needed.
    """
    logger.info("➡️  upload_eye_scan() | user=%s | method=%s | files=%s",
                request.user.username, request.method, list(request.FILES.keys()))
    if request.method != 'POST':
        return redirect('profile_detail')

    profile = HealthProfile.objects.filter(user=request.user).first()
    if not profile:
        messages.error(request, "First complete your health profile, then add eye photos.")
        return redirect('create_update_profile')

    eye_files = request.FILES.getlist('eye_images')
    if not eye_files:
        messages.error(request, "Please choose at least one image to screen.")
        return redirect('profile_detail')

    queued = _save_eye_images(profile, eye_files)
    messages.success(request, f"Eye screening started for {queued} of {len(eye_files)} image(s).")
    return redirect('profile_detail')


# test_analysis/views.py
from django.contrib.auth.models import User
from django.http import HttpResponseForbidden
from usac.models import UserProfile  # for role check


@login_required
def profile_detail_view(request, user_id=None):
    logger.info("➡️  profile_detail_view() | requester=%s (id=%s, role=%s) | target_user_id=%s | method=%s",
                request.user.username, request.user.id, request.user.profile.role, user_id, request.method)
    # Determine target user
    if user_id and request.user.id != user_id:
        # Manager can view any employee in company (keep existing check)
        # Doctor can view employees in their company
        if request.user.profile.role == UserProfile.ROLE_MANAGER:
            logger.debug("   ↳ manager viewing employee id=%s", user_id)
            target_user = get_object_or_404(User, pk=user_id, profile__company__manager=request.user)
        elif request.user.profile.role == UserProfile.ROLE_DOCTOR:
            # Doctor must be in the same company as target
            doctor_company = request.user.profile.company
            logger.debug("   ↳ doctor (company=%s) viewing employee id=%s", doctor_company, user_id)
            target_user = get_object_or_404(User, pk=user_id, profile__company=doctor_company)
        else:
            logger.warning("⛔ user=%s role=%s forbidden from viewing user_id=%s",
                           request.user.username, request.user.profile.role, user_id)
            return HttpResponseForbidden("دسترسی ندارید.")
    else:
        target_user = request.user
        user_id = request.user.id
        logger.debug("   ↳ self-view | target=%s", target_user.username)

    profile = HealthProfile.objects.filter(user=target_user).first()
    logger.debug("   ↳ target=%s | profile=%s", target_user.username, getattr(profile, 'pk', None))
    if not profile:
        # Employee hasn't filled their profile yet – show a simple page
        logger.info("   ℹ️  no profile for target=%s → rendering empty detail page", target_user.username)
        return render(request, 'test_analysis/profile_detail2.html', {
            'profile': None,
            'html_advice': None,
            'deep_report': None,
            'deep_report_json': "",
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
    logger.debug("   ↳ permissions | can_edit_doctor_notes=%s | can_edit_employee=%s",
                 can_edit_doctor_notes, can_edit_employee)

    if request.method == 'POST' and can_edit_doctor_notes:
        logger.info("   📨 POST doctor/manager notes for target=%s", target_user.username)
        doctor_form = DoctorNotesForm(request.POST, instance=profile)
        referral_formset = ReferralFormSet(request.POST, prefix='referrals',
                                           queryset=Referral.objects.filter(profile=profile))
        if doctor_form.is_valid() and referral_formset.is_valid():
            logger.info("   ✅ doctor_form & referral_formset valid → saving notes")
            profile = doctor_form.save(commit=False)
            profile.examining_doctor = request.user  # track who examined
            profile.save()
            logger.debug("   ↳ examining_doctor=%s saved on profile id=%s",
                         request.user.username, profile.id)

            referrals = referral_formset.save(commit=False)
            logger.debug("   ↳ saving %d referral rows", len(referrals))
            for ref in referrals:
                ref.profile = profile
                ref.save()
            referral_formset.save_m2m()
            for obj in referral_formset.deleted_objects:
                logger.debug("   🗑️  deleting referral row id=%s", getattr(obj, 'pk', None))
                obj.delete()
            if request.user.profile.role == UserProfile.ROLE_MANAGER:
                messages.success(request, "یادداشت‌های مدیر با موفقیت ذخیره شد.")
            else:
                messages.success(request, "یادداشت‌های پزشک با موفقیت ذخیره شد.")
            if request.user.profile.role == UserProfile.ROLE_DOCTOR:
                logger.info("   ↪️  doctor → redirect doctor_dashboard")
                return redirect('doctor_dashboard')
            elif request.user.profile.role == UserProfile.ROLE_MANAGER:
                logger.info("   ↪️  manager → redirect manager_dashboard")
                return redirect('manager_dashboard')
            logger.info("   ↪️  redirect profile_detail_user id=%s", target_user.id)
            return redirect('profile_detail_user', user_id=target_user.id)
        else:
            logger.warning("   ❌ doctor_form/referral_formset invalid | doctor_errors=%s | referral_errors=%s",
                           doctor_form.errors.as_json(), referral_formset.errors)
        # if invalid, fall through to render with errors
    else:
        doctor_form = DoctorNotesForm(instance=profile) if can_edit_doctor_notes else None
        referral_formset = ReferralFormSet(prefix='referrals',
                                           queryset=Referral.objects.filter(
                                               profile=profile)) if can_edit_doctor_notes else None

    # Deep Health Research report (structured JSON stored in llm_advice behind a
    # sentinel). When present the template renders the componentized report; the
    # old markdown advice remains a graceful fallback for legacy profiles.
    from .services.deep_research import runner as dr_runner
    deep_report = dr_runner.load_report(profile)
    logger.debug("   ↳ rendering advice | deep_report=%s | llm_advice_present=%s",
                 bool(deep_report), bool(profile.llm_advice))
    html_advice = "" if deep_report else markdown2.markdown(profile.llm_advice or "گزارشی تولید نشده است.")
    context = {
        'profile': profile,
        'html_advice': html_advice,
        'deep_report': deep_report,
        'deep_report_json': json.dumps(deep_report, ensure_ascii=False) if deep_report else "",
        'viewed_user': target_user,
        'doctor_form': doctor_form,
        'referral_formset': referral_formset,
        'can_edit_doctor_notes': can_edit_doctor_notes,
        'can_edit_employee': can_edit_employee,  # <-- new line
        'can_upload_eye': request.user.id == target_user.id,  # owner can add eye scans
        'is_manager': request.user.profile.role == UserProfile.ROLE_MANAGER,
    }
    logger.info("   🖼️  rendering profile_detail2.html for target=%s", target_user.username)
    return render(request, 'test_analysis/profile_detail2.html', context)


@login_required
def processing_page(request):
    logger.info("➡️  processing_page() | user=%s (id=%s)", request.user.username, request.user.id)
    # فقط صفحه‌ای که مودال را نشان می‌دهد
    profile = get_object_or_404(HealthProfile, user=request.user)
    logger.debug("   ↳ profile id=%s | report_ready=%s | report_error=%s",
                 profile.id, profile.report_ready, bool(profile.report_error))
    return render(request, 'test_analysis/processing2.html', {'profile': profile})


@login_required
@require_GET
def report_status(request):
    logger.debug("➡️  report_status() polled | user=%s (id=%s)", request.user.username, request.user.id)
    profile = get_object_or_404(HealthProfile, user=request.user)
    # Eye screening is independent of the (Celery) LLM report — it runs
    # synchronously on submit. Report this separately so the processing page can
    # let the user reach their results without waiting on the report.
    eye_ready = profile.eye_images.filter(analysis__status='done').exists()
    logger.info("   📊 report_status | profile_id=%s | ready=%s | error=%s | eye_ready=%s",
                profile.id, profile.report_ready, bool(profile.report_error), eye_ready)
    return JsonResponse({
        "ready": profile.report_ready,
        "eye_ready": eye_ready,
        "error": bool(profile.report_error),
        "error_msg": profile.report_error or "",
        "detail_url": request.build_absolute_uri(
            # صفحه‌ی گزارش موجود خودت
            reverse('profile_detail')
        ),
    })


@login_required
def minigame_page(request):
    logger.info("➡️  minigame_page() | user=%s", request.user.username)
    return render(request, 'test_analysis/minigame.html', {})


@login_required
def doctor_dashboard(request):
    logger.info("➡️  doctor_dashboard() | user=%s (id=%s, role=%s)",
                request.user.username, request.user.id, request.user.profile.role)
    # Only doctors can access
    if request.user.profile.role != UserProfile.ROLE_DOCTOR:
        logger.warning("⛔ non-doctor user=%s attempted doctor_dashboard", request.user.username)
        return HttpResponseForbidden("You don't have access to this page.")
    doctor_company = request.user.profile.company
    logger.debug("   ↳ doctor_company=%s", doctor_company)
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
    logger.info("   📊 dashboard stats | total_employees=%d | examined=%d | completed=%d",
                total_employees, examined_count, completed_exams)

    context = {
        'employees': employees,
        'examined_count': examined_count,
        'total_employees': total_employees,
        'completed_exams': completed_exams,
        'pending_exams': max(total_employees - completed_exams, 0),
    }
    return render(request, 'test_analysis/doctor_dashboard.html', context)


@login_required
@user_passes_test(is_manager)
def manager_edit_profile(request, user_id):
    logger.info("➡️  manager_edit_profile() | manager=%s | target_user_id=%s | method=%s",
                request.user.username, user_id, request.method)
    target_user = get_object_or_404(User, pk=user_id, profile__company__manager=request.user)
    profile, created = HealthProfile.objects.get_or_create(user=target_user)
    logger.debug("   ↳ target=%s | profile_id=%s | created=%s", target_user.username, profile.id, created)

    if request.method == 'POST':
        logger.info("   📨 POST manager edit | post_keys=%d | files=%s",
                    len(request.POST), list(request.FILES.keys()))
        # use the full HealthProfileForm (or EmployeeProfileForm + DoctorNotesForm combined)
        # For simplicity, we can use the same multi-step form but without restrictions.
        form = EmployeeProfileForm(request.POST, request.FILES, instance=profile)
        job_formset = PreviousJobFormSet(request.POST, prefix='jobs',
                                         queryset=PreviousJob.objects.filter(profile=profile))
        referral_formset = ReferralFormSet(request.POST, prefix='referrals',
                                           queryset=Referral.objects.filter(profile=profile))
        if form.is_valid() and job_formset.is_valid() and referral_formset.is_valid():
            logger.info("   ✅ all forms valid → saving manager edits for target=%s", target_user.username)
            profile = form.save(commit=False)
            profile.user = target_user
            profile.save()
            jobs = job_formset.save(commit=False)
            logger.debug("   ↳ saving %d job rows + referrals", len(jobs))
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

            # Save eye photos and medical-test documents uploaded during this
            # edit (duplicates are skipped so nothing is re-processed).
            _save_eye_images(profile, request.FILES.getlist('eye_images'))
            _save_medical_tests(profile, request.FILES.getlist('medical_test_files'))

            logger.info("   ↪️  saved → redirect member_detail id=%s", target_user.id)
            messages.success(request, "پرونده با موفقیت ویرایش شد.")
            return redirect('member_detail', user_id=target_user.id)
        else:
            logger.warning("   ❌ manager edit invalid | form=%s | jobs=%s | referrals=%s",
                           form.errors.as_json(), job_formset.errors, referral_formset.errors)
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
    logger.debug("   🖼️  rendering manager_edit_profile.html for target=%s", target_user.username)
    return render(request, 'test_analysis/manager_edit_profile.html', context)


import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from .health_chat_agent import chat_with_assistant, doctor_assist_assistant, doctor_research_chat


@csrf_exempt
@login_required
def health_chat_api(request):
    logger.info("➡️  health_chat_api() | user=%s | method=%s", request.user.username, request.method)
    if request.method != 'POST':
        logger.warning("   ⛔ non-POST request to health_chat_api → 405")
        return JsonResponse({'error': 'POST required'}, status=405)
    try:
        data = json.loads(request.body)
        message = data.get('message', '')
        history = data.get('history', [])
        logger.info("   💬 incoming message (len=%d) | history_turns=%d", len(message or ""), len(history))
        logger.debug("   ↳ user message: %r", message)

        # فراخوانی کانال چت با ساختار جدید خروجی
        logger.debug("   ↳ calling chat_with_assistant() ...")
        reply, finder_results = chat_with_assistant(message, history)
        logger.info("   ✅ assistant replied (len=%d) | tool_called=%s",
                    len(reply or ""), finder_results is not None)

        return JsonResponse({
            'reply': reply,
            'finder_results': finder_results,  # تغییر نام متغیر برای پوشش پزشک و داروخانه
            'tool_called': finder_results is not None
        })
    except Exception as e:
        logger.exception("   🔥 health_chat_api failed: %s", e)
        return JsonResponse({'error': str(e)}, status=500)


def _can_user_edit_doctor_notes(request_user, target_user):
    """Mirror of the permission logic in profile_detail_view (doctor same-company / manager)."""
    role = request_user.profile.role
    logger.debug("🔎 _can_user_edit_doctor_notes() | requester=%s role=%s | target=%s",
                 request_user.username, role, target_user.username)
    if role == UserProfile.ROLE_DOCTOR and request_user.id != target_user.id:
        allowed = request_user.profile.company == target_user.profile.company
        logger.debug("   ↳ doctor same-company check → %s", allowed)
        return allowed
    if role == UserProfile.ROLE_MANAGER:
        logger.debug("   ↳ manager → allowed=True")
        return True
    logger.debug("   ↳ default → allowed=False")
    return False


def _build_profile_context(profile, target_user):
    """Render a comprehensive, grounded clinical summary of the employee's profile for the LLM assistant."""
    logger.debug("🧱 _build_profile_context() | target=%s | profile_id=%s",
                 target_user.username, getattr(profile, 'pk', None))
    lines = []

    def add(label, value):
        if value not in (None, '', 0, False):
            lines.append(f"- {label}: {value}")

    full_name = target_user.get_full_name() or target_user.username
    lines.append(f"## ۱. مشخصات فردی بیمار/شاغل: {full_name}")
    add("نام کاربری", target_user.username)
    add("کد ملی", profile.national_id)
    add("نام پدر", profile.father_name)
    add("تاریخ تولد", profile.date_of_birth)
    add("سن", profile.age)
    add("جنسیت", profile.get_gender_display() if profile.gender else profile.gender)
    add("استان محل سکونت", profile.living_province or profile.province_of_residence)
    add("محله", profile.neighborhood)
    add("بیمه", profile.insurance)
    add("وضعیت تاهل", profile.marital_status)
    add("تعداد فرزند", profile.children_count)
    add("وضعیت نظام وظیفه", profile.military_service_status)
    add("رسته خدمت", profile.military_service_rank)
    if profile.medical_exemption_reason:
        add("علت معافیت پزشکی", profile.medical_exemption_reason)
    add("آدرس محل کار", profile.work_address)
    add("تلفن محل کار", profile.work_phone)

    lines.append("\n## ۲. مشخصات شغلی و مواجهات زیان‌آور شغلی")
    add("شغل فعلی", profile.current_job_title)
    add("وظایف محوله فعلی", profile.current_job_duties)
    add("تاریخ شروع کار فعلی", profile.current_job_start_date)

    hazards = []
    for f in profile._meta.get_fields():
        name = getattr(f, 'name', '')
        if name.startswith('hazard_') and name.endswith('_other'):
            val = getattr(profile, name, None)
            if val:
                hazards.append(f"{getattr(f, 'verbose_name', name)}: {val}")
        elif name.startswith('hazard_'):
            if getattr(profile, name, False) is True:
                hazards.append(str(getattr(f, 'verbose_name', name)))
    if hazards:
        lines.append("- عوامل زیان‌آور شغلی شناسایی‌شده: " + "، ".join(hazards))
    else:
        lines.append("- عوامل زیان‌آور شغلی: موردی ثبت نشده است.")

    try:
        prev_jobs = profile.previous_jobs.all()
        if prev_jobs.exists():
            lines.append("\n### سوابق شغلی قبلی:")
            for pj in prev_jobs:
                lines.append(f"  * {pj.title} (وظایف: {pj.duties or '-'} | از {pj.start_date or '-'} تا {pj.end_date or '-'} | علت ترک: {pj.reason_for_leaving or '-'})")
    except Exception:
        pass

    lines.append("\n## ۳. سوابق پزشکی، شخصی و خانوادگی")
    if profile.has_disease_history:
        add("سابقه بیماری قبلی", profile.disease_history_details or "دارد")
    if profile.does_symptoms_change_at_work:
        lines.append("- تغییر علائم در محیط کار: دارد")
    if profile.do_colleagues_have_similar_symptoms:
        lines.append("- علائم مشابه در همکاران: دارد")
    if profile.does_symptoms_change_on_holidays:
        lines.append("- بهبود علائم در مرخصی/تعطیلات: دارد")
    if profile.has_allergies:
        add("حساسیت و آلرژی", profile.allergy_details or "دارد")
    if profile.has_hospitalization_history:
        add("سابقه بستری بیمارستانی", profile.hospitalization_reason or "دارد")
    if profile.has_surgery_history:
        add("سابقه عمل جراحی", profile.surgery_details or "دارد")
    if profile.has_family_cancer_or_chronic_disease:
        add("سابقه بیماری مزمن/سرطان در خانواده", profile.family_disease_details or "دارد")
    if profile.is_on_medication:
        add("داروهای مصرفی فعلی", profile.medication_details or "دارد")
    if profile.is_currently_smoking:
        add("مصرف دخانیات فعلی", f"مصرف می‌کند ({profile.smoking_details or ''} - روزانه {profile.cigs_per_day or 0} نخ)")
    elif profile.has_past_smoking_history:
        add("سوابق مصرف دخانیات", "سابقه مصرف در گذشته دارد")
    if profile.on_bp_meds:
        lines.append("- مصرف داروی فشار خون: دارد")
    if profile.has_diabetes:
        lines.append("- سابقه دیابت: دارد")
    if profile.hobbies:
        add("سرگرمی‌ها", profile.hobbies)
    if profile.has_occupational_accident_history:
        add("سابقه حادثه ناشی از کار", profile.accident_details or "دارد")
    if profile.has_absence_over_3_days:
        lines.append("- سابقه غیبت از کار بیش از ۳ روز: دارد")
    if profile.lives_near_industrial_center:
        lines.append("- سکونت در مجاورت مراکزی صنعتی: دارد")
    if profile.has_medical_commission_referral:
        lines.append("- سابقه معرفی به کمیسیون پزشکی: دارد")

    lines.append("\n## ۴. علائم حیاتی و یافته‌های معاینه فیزیکی اولیه")
    add("تاریخ معاینه", profile.exam_date)
    add("وزن (kg)", profile.exam_weight)
    add("قد (cm)", profile.exam_height)
    add("BMI", profile.bmi)
    if profile.exam_systolic_bp or profile.exam_diastolic_bp or profile.exam_blood_pressure:
        bp_str = profile.exam_blood_pressure or f"{profile.exam_systolic_bp or '-'}/{profile.exam_diastolic_bp or '-'}"
        lines.append(f"- فشار خون: {bp_str} mmHg")
    add("تعداد نبض", f"{profile.exam_pulse_rate} در دقیقه" if profile.exam_pulse_rate else None)

    existing_notes = []
    note_fields = [
        ("معاینه عمومی", profile.general_exam_notes),
        ("سر و گردن", profile.head_neck_exam_notes),
        ("چشم", profile.eye_exam_notes),
        ("گوش، حلق و بینی", profile.ent_mouth_exam_notes),
        ("ریه", profile.lung_exam_notes),
        ("قلب و عروق", profile.cardiovascular_exam_notes),
        ("شکم و لگن", profile.abdomen_pelvis_exam_notes),
        ("دستگاه ادراری", profile.urinary_system_exam_notes),
        ("اسکلتی-عضلانی", profile.musculoskeletal_exam_notes),
        ("سیستم عصبی", profile.nervous_system_exam_notes),
        ("سلامت روان", profile.mental_health_exam_notes),
        ("پوست و مو", profile.skin_hair_nails_exam_notes),
    ]
    for label, val in note_fields:
        if val and val.strip():
            existing_notes.append(f"  * {label}: {val.strip()}")
    if existing_notes:
        lines.append("### یادداشت‌های معاینه قبلی ثبت‌شده توسط پزشک:")
        lines.extend(existing_notes)

    lines.append("\n## ۵. نتایج پاراکلینیک و آزمایشگاهی پرونده")
    if profile.spirometry_fvc or profile.spirometry_fev1 or profile.spirometry_fev1_fvc_ratio or profile.spirometry_interpretation:
        lines.append("### اسپیرومتری (تست تنفسی):")
        add("FVC", profile.spirometry_fvc)
        add("FEV1", profile.spirometry_fev1)
        add("FEV1/FVC Ratio", profile.spirometry_fev1_fvc_ratio)
        add("FEF 25-75%", profile.spirometry_fef_25_75)
        add("PEF", profile.spirometry_pef)
        add("تفسیر اسپیرومتری", profile.spirometry_interpretation)

    if profile.ecg_findings:
        add("یافته نوار قلب (ECG)", profile.ecg_findings)
    if profile.chest_xray_findings:
        add("یافته رادیوگرافی قفسه صدری (CXR)", profile.chest_xray_findings)
    if profile.other_paraclinical_notes:
        add("سایر پاراکلینیک (سونوگرافی/رادیولوژی)", profile.other_paraclinical_notes)

    direct_labs = []
    if profile.lab_glucose: direct_labs.append(f"قند خون (Glucose): {profile.lab_glucose} mg/dL")
    if profile.lab_fbs: direct_labs.append(f"قند خون ناشتا (FBS): {profile.lab_fbs}")
    if profile.lab_total_cholesterol: direct_labs.append(f"کلسترول کل: {profile.lab_total_cholesterol} mg/dL")
    if profile.lab_cbc_wbc: direct_labs.append(f"WBC: {profile.lab_cbc_wbc}")
    if profile.lab_cbc_rbc: direct_labs.append(f"RBC: {profile.lab_cbc_rbc}")
    if profile.lab_ua_prot: direct_labs.append(f"پروتئین ادرار: {profile.lab_ua_prot}")
    if direct_labs:
        lines.append("### آزمایش‌های ثبت‌شده روی پرونده:")
        for dl in direct_labs:
            lines.append(f"  * {dl}")

    try:
        med_tests = profile.medical_tests.all()
        if med_tests.exists():
            lines.append("\n## ۶. مدارک و آزمایش‌های پزشکی بارگذاری‌شده کاربر (Medical Tests)")
            for mt in med_tests:
                lines.append(f"\n### سند آزمایش: {mt.report_type or mt.filename} (وضعیت استخراج: {mt.get_status_display()})")
                if mt.lab_name: lines.append(f"- آزمایشگاه: {mt.lab_name}")
                if mt.collected_on: lines.append(f"- تاریخ نمونه‌گیری: {mt.collected_on}")
                if mt.summary: lines.append(f"- خلاصه نتایج استخراج‌شده: {mt.summary}")
                if mt.abnormal_count > 0: lines.append(f"- تعداد موارد خارج از محدوده نرمال: {mt.abnormal_count}")
                if mt.panels and isinstance(mt.panels, list):
                    lines.append("- پنل‌ها و آنالیت‌های آزمایشگاهی استخراج‌شده:")
                    for p in mt.panels:
                        p_name = p.get('name') or 'پنل'
                        analytes = p.get('analytes') or []
                        lines.append(f"  * **{p_name}**:")
                        for a in analytes:
                            a_name = a.get('name') or ''
                            res = a.get('result') or ''
                            unit = a.get('unit') or ''
                            flag = a.get('flag') or ''
                            ref = a.get('reference') or ''
                            flag_str = f" [{flag.upper()}]" if flag else ""
                            ref_str = f" (محدوده نرمال: {ref})" if ref else ""
                            lines.append(f"    - {a_name}: {res} {unit}{flag_str}{ref_str}")
    except Exception as e:
        logger.warning("   ⚠️ error formatting medical_tests: %s", e)

    try:
        eye_images = profile.eye_images.all()
        for ei in eye_images:
            if hasattr(ei, 'analysis') and ei.analysis:
                ea = ei.analysis
                if ea.status == 'done':
                    lines.append("\n## ۷. غربالگری هوشمند آنمی (تصویر چشم)")
                    lines.append(f"- نتیجه غربالگری آنمی: {ea.anemia_label} (درصد اطمینان: {int((ea.anemia_confidence or 0)*100)}%)")
    except Exception:
        pass

    try:
        refs = profile.referrals.all()
        if refs.exists():
            lines.append("\n## ۸. سوابق ارجاعات تخصصی قبلی")
            for r in refs:
                lines.append(f"- ارجاع به {r.specialty} در تاریخ {r.date}: علت '{r.reason}' | نتیجه: '{r.result}'")
    except Exception:
        pass

    if profile.llm_advice:
        snippet = profile.llm_advice.strip()
        if len(snippet) > 6000:
            snippet = snippet[:6000] + "\n... [ادامه گزارش تحلیل هوش مصنوعی]"
        lines.append("\n## ۹. گزارش و تحلیل ریسک هوش مصنوعی (AI Health Audit Report)\n" + snippet)

    context_text = "\n".join(lines)
    logger.debug("   ↳ built rich profile context | lines=%d | chars=%d", len(lines), len(context_text))
    return context_text


@csrf_exempt
@login_required
def doctor_assist_api(request):
    """Drafting assistant for doctors/managers: fills the examination record from a prompt."""
    logger.info("➡️  doctor_assist_api() | user=%s | method=%s", request.user.username, request.method)
    if request.method != 'POST':
        logger.warning("   ⛔ non-POST request to doctor_assist_api → 405")
        return JsonResponse({'error': 'POST required'}, status=405)
    try:
        data = json.loads(request.body)
        prompt = (data.get('prompt') or data.get('message') or '').strip()
        user_id = data.get('user_id')
        history = data.get('history', [])
        logger.info("   🩺 doctor-assist request | target_user_id=%s | prompt_len=%d | history_turns=%d",
                    user_id, len(prompt), len(history))
        logger.debug("   ↳ prompt: %r", prompt)

        if not prompt:
            logger.warning("   ❌ empty prompt → 400")
            return JsonResponse({'error': 'متن درخواست خالی است.'}, status=400)

        target_user = get_object_or_404(User, pk=user_id)
        if not _can_user_edit_doctor_notes(request.user, target_user):
            logger.warning("   ⛔ user=%s not permitted to assist target=%s → 403",
                           request.user.username, target_user.username)
            return JsonResponse({'error': 'شما مجاز به تکمیل پرونده این کاربر نیستید.'}, status=403)

        profile = HealthProfile.objects.filter(user=target_user).first()
        if not profile:
            logger.warning("   ❌ target=%s has no profile → 404", target_user.username)
            return JsonResponse({'error': 'این کاربر هنوز پرونده‌ای ثبت نکرده است.'}, status=404)

        context_text = _build_profile_context(profile, target_user)
        logger.debug("   ↳ calling doctor_assist_assistant() ...")
        suggestions, error = doctor_assist_assistant(prompt, context_text, history)
        if error:
            logger.error("   🔥 doctor_assist_assistant returned error → 502 | error=%s", error)
            return JsonResponse({'error': error}, status=502)

        logger.info("   ✅ doctor-assist suggestions generated | keys=%s",
                    list(suggestions.keys()) if isinstance(suggestions, dict) else type(suggestions))
        return JsonResponse({'suggestions': suggestions})
    except Exception as e:
        logger.exception("   🔥 doctor_assist_api failed: %s", e)
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@login_required
def doctor_research_api(request):
    """Research assistant for doctors/managers reviewing an employee's ROP/KC case.

    POST JSON: {message, context, history, employee_id}
    Only users with role DOCTOR or MANAGER may use it (else 403).
    Returns {'answer': <markdown>, 'sources': [{id,title,url,domain}, ...]}.
    """
    logger.info("➡️  doctor_research_api() | user=%s | method=%s", request.user.username, request.method)
    if request.method != 'POST':
        logger.warning("   ⛔ non-POST request to doctor_research_api → 405")
        return JsonResponse({'error': 'POST required'}, status=405)

    role = getattr(getattr(request.user, 'profile', None), 'role', None)
    if role not in (UserProfile.ROLE_DOCTOR, UserProfile.ROLE_MANAGER):
        logger.warning("   ⛔ user=%s role=%s not permitted → 403", request.user.username, role)
        return JsonResponse({'error': 'شما مجاز به استفاده از این دستیار نیستید.'}, status=403)

    try:
        data = json.loads(request.body)
        message = (data.get('message') or '').strip()
        context_text = data.get('context') or ''
        history = data.get('history', []) or []
        employee_id = data.get('employee_id')
        logger.info("   🔬 research request | employee_id=%s | message_len=%d | history_turns=%d",
                    employee_id, len(message), len(history))

        if not message:
            logger.warning("   ❌ empty message → 400")
            return JsonResponse({'error': 'پیام خالی است.'}, status=400)

        answer, sources = doctor_research_chat(message, history, context_text)
        logger.info("   ✅ research answer ready | answer_len=%d | sources=%d",
                    len(answer or ""), len(sources or []))
        return JsonResponse({'answer': answer, 'sources': sources})
    except Exception as e:
        logger.exception("   🔥 doctor_research_api failed: %s", e)
        return JsonResponse({'error': str(e)}, status=500)


# ============================================================================ #
#  Deep Health Research — grounded, componentized report (runs in-process,
#  no Celery). Start kicks off a background thread; the page polls status and
#  then fetches the finished JSON report.
# ============================================================================ #
def _resolve_deep_target(request, user_id):
    """Return the HealthProfile the requester is allowed to generate/view a deep
    report for, or None if forbidden. Mirrors profile_detail_view's rules."""
    if user_id and request.user.id != user_id:
        role = request.user.profile.role
        if role == UserProfile.ROLE_MANAGER:
            target = User.objects.filter(pk=user_id, profile__company__manager=request.user).first()
        elif role == UserProfile.ROLE_DOCTOR:
            target = User.objects.filter(pk=user_id, profile__company=request.user.profile.company).first()
        else:
            return None
    else:
        target = request.user
    if not target:
        return None
    return HealthProfile.objects.filter(user=target).first()


@login_required
@csrf_exempt
def deep_research_start(request, user_id=None):
    logger.info("➡️  deep_research_start() | requester=%s | target_user=%s", request.user.username, user_id)
    profile = _resolve_deep_target(request, user_id)
    if not profile:
        return JsonResponse({'error': 'forbidden or no profile'}, status=403)
    from .services.deep_research import runner as dr_runner
    prog = dr_runner.start(profile.id)
    return JsonResponse({'ok': True, 'progress': prog})


@login_required
@require_GET
def deep_research_status(request, user_id=None):
    profile = _resolve_deep_target(request, user_id)
    if not profile:
        return JsonResponse({'error': 'forbidden'}, status=403)
    from .services.deep_research import runner as dr_runner
    prog = dr_runner.get_progress(profile.id) or {'state': 'idle'}
    has_report = bool(dr_runner.load_report(profile))
    return JsonResponse({'progress': prog, 'has_report': has_report})


@login_required
@require_GET
def deep_research_result(request, user_id=None):
    profile = _resolve_deep_target(request, user_id)
    if not profile:
        return JsonResponse({'error': 'forbidden'}, status=403)
    from .services.deep_research import runner as dr_runner
    report = dr_runner.load_report(profile)
    if not report:
        return JsonResponse({'error': 'no report'}, status=404)
    return JsonResponse({'report': report})
