from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.db import connection
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
# NEW imports (add near the top of views.py)
from django.db.models import Q
from django.core.serializers.json import DjangoJSONEncoder
import json

import csv

# App models / forms
from .forms import RoleChoiceForm, ManagerSignupForm, StaffSignupForm
from .models import Company, UserProfile, Invitation

# Other app models
from django.contrib.auth.models import User
from single_rop.models import PredictionLog
from double_rop.models import PredictionResult
from test_analysis.models import HealthProfile

# -----------------------
# Helpers
# -----------------------

from django.contrib.auth.decorators import login_required

@login_required
def role_based_redirect(request):
    """Redirect user after login based on role."""
    role = getattr(getattr(request.user, "profile", None), "role", None)
    if role == UserProfile.ROLE_MANAGER:
        return redirect("manager_dashboard")
    elif role == UserProfile.ROLE_DOCTOR:
        return redirect("doctor_dashboard")
    return redirect("dilemma")


def _table_exists(table_name: str) -> bool:
    """Avoid touching tables that aren't migrated yet."""
    try:
        return table_name in connection.introspection.table_names()
    except Exception:
        return False


def is_manager(user):
    """Safe check for manager role (no direct OneToOne access)."""
    if not _table_exists('usac_userprofile'):
        return False
    try:
        return UserProfile.objects.filter(user=user, role=UserProfile.ROLE_MANAGER).exists()
    except Exception:
        return False


def _get_user_company(user):
    """
    Safely determine the user's company.
    1) Try via UserProfile.company
    2) Fallback to Company where manager=user
    3) If fallback found, attach it to profile (best-effort)
    """
    if not _table_exists('usac_userprofile'):
        return None

    company = None
    try:
        company_id = (
            UserProfile.objects
            .filter(user=user)
            .values_list('company', flat=True)
            .first()
        )
        if company_id:
            try:
                company = Company.objects.get(pk=company_id)
            except Company.DoesNotExist:
                company = None

        if not company:
            company = Company.objects.filter(manager=user).first()
            if company:
                UserProfile.objects.filter(user=user).update(company=company)
    except Exception:
        company = None

    return company


# -----------------------
# Auth / Login
# -----------------------
def custom_login(request):
    if request.user.is_authenticated:
        return redirect('dilemma')

    if request.method == 'POST':
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            # redirect by role
            try:
                if hasattr(user, "profile"):
                    if user.profile.role == UserProfile.ROLE_MANAGER:
                        return redirect("manager_dashboard")
                    elif user.profile.role == UserProfile.ROLE_DOCTOR:
                        return redirect("doctor_dashboard")
            except Exception:
                pass
            return redirect("dilemma")

        return render(request, 'usac/login2.html', {'error': 'Invalid credentials'})

    return render(request, 'usac/login2.html')


# -----------------------
# Dilemma (role-aware, table-safe)
# -----------------------
@login_required(login_url='')
def dilemma_view(request):
    # Determine role safely
    role = None
    if _table_exists('usac_userprofile'):
        try:
            prof = UserProfile.objects.filter(user=request.user).only('role').first()
            if prof:
                role = prof.role
        except Exception:
            role = None

    # Doctors and managers go directly to their dashboards
    if role == UserProfile.ROLE_MANAGER:
        return redirect('manager_dashboard')
    elif role == UserProfile.ROLE_DOCTOR:
        return redirect('doctor_dashboard')

    # Employees (and others) get the dilemma page
    return render(
        request,
        'usac/dilemma2.html',
        {
            'role': role,
            'is_manager': False,
            'is_doctor': False,
            'is_employee': True,
            'is_superuser': request.user.is_superuser,
        }
    )


# -----------------------
# History + CSV exports
# -----------------------
@login_required(login_url='')
def history_view(request):
    single = PredictionLog.objects.filter(user=request.user)
    double = PredictionResult.objects.filter(user=request.user)
    return render(request, 'usac/history.html', {'single_results': single, 'double_results': double})


@login_required(login_url='')
def export_history_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="prediction_history.csv"'
    writer = csv.writer(response)

    # ---- Single ROP ----
    writer.writerow(['==== ROP RECORDS ===='])
    writer.writerow([
        'Type',
        'Username',
        'Email',
        'File Name',
        'Predicted Class',
        'Probability',
        'Corrected Class',
        'Review Comment',
        'Stage Class',
        'Stage Probability',
        'Corrected Stage Class',
        'Execution Time',
        'Timestamp',
        'Image URL'
    ])

    single_logs = PredictionLog.objects.filter(user=request.user)
    for s in single_logs:
        writer.writerow([
            'ROP',
            s.user.username,
            s.user.email,
            getattr(s, 'file_name', ''),
            getattr(s, 'predicted_class', ''),
            f"{getattr(s, 'probability', 0.0):.4f}",
            getattr(s, 'corrected_class', '') or "",
            getattr(s, 'review_comment', '') or "",
            getattr(s, 'stage_class', ''),
            f"{getattr(s, 'stage_probability', 0.0):.3f}",
            getattr(s, 'stage_corrected_class', ''),
            getattr(s, 'execution_time', ''),
            getattr(s, 'timestamp', ''),
            request.build_absolute_uri(getattr(s, 'image_url', '')) if getattr(s, 'image_url', '') else ""
        ])

    writer.writerow([])

    # ---- Double ROP / KC ----
    writer.writerow(['==== KC RECORDS ===='])
    writer.writerow([
        'Type',
        'Username',
        'Email',
        'Predicted Labels',
        'Probabilities',
        'Corrected Labels',
        'Review Comment',
        'Inference Time',
        'Timestamp',
        'Image URLs'
    ])

    double_logs = PredictionResult.objects.filter(user=request.user)
    for d in double_logs:
        image_urls = []
        if getattr(d, 'left_image', None):
            image_urls.append(request.build_absolute_uri(d.left_image.url))
        if getattr(d, 'right_image', None):
            image_urls.append(request.build_absolute_uri(d.right_image.url))

        writer.writerow([
            'KC',
            d.user.username,
            d.user.email,
            f"L: {getattr(d, 'left_label', '')}, R: {getattr(d, 'right_label', '')}, Z: {getattr(d, 'z_class_label', '')}",
            f"L: {getattr(d, 'left_probability', 0.0):.4f}, R: {getattr(d, 'right_probability', 0.0):.4f}, Z: {getattr(d, 'z_class_probability', 0.0):.4f}",
            f"Corrected → L: {getattr(d, 'corrected_left_label', '')}, R: {getattr(d, 'corrected_right_label', '')}, Z: {getattr(d, 'corrected_z_label', '')}",
            getattr(d, 'review_comment', '') or "",
            f"{getattr(d, 'inference_time', 0.0):.4f}",
            getattr(d, 'created_at', ''),
            " | ".join(image_urls)
        ])

    return response


from django.contrib.admin.views.decorators import staff_member_required


@staff_member_required
def export_misclassified_rop_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="misclassified_rop.csv"'
    writer = csv.writer(response)

    writer.writerow([
        'Username',
        'Email',
        'File Name',
        'Predicted Class',
        'Probability',
        'Corrected Class',
        'Review Comment',
        'Stage Class',
        'Stage Probability',
        'Corrected Stage Class',
        'Execution Time',
        'Timestamp',
        'Image URL'
    ])

    for s in PredictionLog.objects.filter(classification_status=-1):
        writer.writerow([
            s.user.username,
            s.user.email,
            getattr(s, 'file_name', ''),
            getattr(s, 'predicted_class', ''),
            f"{getattr(s, 'probability', 0.0):.4f}",
            getattr(s, 'corrected_class', '') or "",
            getattr(s, 'review_comment', '') or "",
            getattr(s, 'stage_class', ''),
            f"{getattr(s, 'stage_probability', 0.0):.3f}",
            getattr(s, 'stage_corrected_class', ''),
            getattr(s, 'execution_time', ''),
            getattr(s, 'timestamp', ''),
            request.build_absolute_uri(getattr(s, 'image_url', '')) if getattr(s, 'image_url', '') else ""
        ])

    return response


@staff_member_required
def export_misclassified_kc_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="misclassified_kc.csv"'
    writer = csv.writer(response)

    writer.writerow([
        'Username',
        'Email',
        'Predicted Labels',
        'Probabilities',
        'Corrected Labels',
        'Review Comment',
        'Inference Time',
        'Timestamp',
        'Image URLs'
    ])

    for d in PredictionResult.objects.filter(classification_status=-1):
        image_urls = []
        if getattr(d, 'left_image', None):
            image_urls.append(request.build_absolute_uri(d.left_image.url))
        if getattr(d, 'right_image', None):
            image_urls.append(request.build_absolute_uri(d.right_image.url))

        writer.writerow([
            d.user.username,
            d.user.email,
            f"L: {getattr(d, 'left_label', '')}, R: {getattr(d, 'right_label', '')}, Z: {getattr(d, 'z_class_label', '')}",
            f"L: {getattr(d, 'left_probability', 0.0):.4f}, R: {getattr(d, 'right_probability', 0.0):.4f}, Z: {getattr(d, 'z_class_probability', 0.0):.4f}",
            f"L: {getattr(d, 'corrected_left_label', '')}, R: {getattr(d, 'corrected_right_label', '')}, Z: {getattr(d, 'corrected_z_label', '')}",
            getattr(d, 'review_comment', '') or "",
            f"{getattr(d, 'inference_time', 0.0):.2f}s",
            getattr(d, 'created_at', ''),
            " | ".join(image_urls),
        ])

    return response


# -----------------------
# Signup flow (role based)
# -----------------------
def choose_role_view(request):
    """First step of signup: choose manager / doctor / employee."""
    if request.method == 'POST':
        form = RoleChoiceForm(request.POST)
        if form.is_valid():
            role = form.cleaned_data['role']
            if role == UserProfile.ROLE_MANAGER:
                return redirect('signup_manager')
            elif role == UserProfile.ROLE_DOCTOR:
                return redirect('signup_doctor')
            else:
                return redirect('signup_employee')
    else:
        form = RoleChoiceForm()
    return render(request, 'usac/signup_choose_role2.html', {'form': form})


def signup_manager_view(request):
    if request.method == 'POST':
        form = ManagerSignupForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "ثبت نام مدیر و شرکت با موفقیت انجام شد. حالا وارد شوید.")
            return redirect('login')
    else:
        form = ManagerSignupForm()
    return render(request, 'usac/signup_manager2.html', {'form': form})


def signup_doctor_view(request):
    return _staff_signup(request, role=UserProfile.ROLE_DOCTOR)


def signup_employee_view(request):
    return _staff_signup(request, role=UserProfile.ROLE_EMPLOYEE)


def _staff_signup(request, role):
    if request.method == 'POST':
        form = StaffSignupForm(request.POST, role=role)
        if form.is_valid():
            form.save()
            messages.success(request, "ثبت نام با موفقیت انجام شد. حالا وارد شوید.")
            return redirect('login')
    else:
        form = StaffSignupForm(role=role)
    return render(request, 'usac/signup_staff2.html', {'form': form, 'role': role})

# -----------------------
# Manager dashboard
# -----------------------
@login_required(login_url='')
@user_passes_test(is_manager, login_url='')
def manager_dashboard(request):
    company = _get_user_company(request.user)

    # If no company is attached, render page with a clear message and block POSTs
    if request.method == 'POST' and not company:
        messages.error(request, "حساب مدیر شما هنوز به یک شرکت متصل نیست. لطفاً ثبت‌نام مدیر را کامل کنید یا با ادمین تماس بگیرید.")
        return redirect('manager_dashboard')

    if not company:
        context = {
            'company': None,
            'doctors': User.objects.none(),
            'employees': User.objects.none(),
            'invitations': Invitation.objects.none(),
            # Charts: empty payloads
            'bmi_bins_json': json.dumps({}, ensure_ascii=False),
            'risks_json': json.dumps({}, ensure_ascii=False),
            'opinions_json': json.dumps({}, ensure_ascii=False),
        }
        messages.warning(request, "شرکت شما یافت نشد. اگر همین الان ثبت‌نام کرده‌اید، یکبار خارج و وارد شوید یا ثبت‌نام مدیر را دوباره انجام دهید.")
        return render(request, 'usac/manager_dashboard2.html', context)

    # Normal flow when company exists
    doctors = User.objects.filter(
        profile__company=company,
        profile__role=UserProfile.ROLE_DOCTOR
    ).select_related('profile')

    employees = User.objects.filter(
        profile__company=company,
        profile__role=UserProfile.ROLE_EMPLOYEE
    ).select_related('profile').prefetch_related('health_profile')

    invitations = Invitation.objects.filter(company=company).order_by('-created_at')

    # Handle invite POST
    if request.method == 'POST':
        national_code = (request.POST.get('national_code') or '').strip()
        add_role = request.POST.get('invite_role', UserProfile.ROLE_EMPLOYEE)

        valid_roles = dict(UserProfile.ROLE_CHOICES).keys()
        if not (len(national_code) == 10 and national_code.isdigit() and add_role in valid_roles):
            messages.error(request, "کد ملی یا نقش نامعتبر است.")
            return redirect('manager_dashboard')

        # Create/update invitation scoped to this company
        inv, created = Invitation.objects.get_or_create(
            company=company,
            national_code=national_code,
            defaults={'role': add_role}
        )
        if not created:
            inv.role = add_role
            inv.save(update_fields=['role'])

        messages.success(request, "کد ملی ذخیره شد. کاربر می‌تواند با این کد ملی ثبت‌نام کند.")
        return redirect('manager_dashboard')

    # =========================
    #   ANALYTICS / CHART DATA
    # =========================
    # Health profiles for *employees* of this company
    profiles = HealthProfile.objects.filter(user__in=employees)

    # BMI distribution (WHO)
    bmi_bins = {
        "کم‌وزن (<18.5)": profiles.filter(bmi__lt=18.5).count(),
        "نرمال (18.5–24.9)": profiles.filter(bmi__gte=18.5, bmi__lt=25).count(),
        "اضافه‌وزن (25–29.9)": profiles.filter(bmi__gte=25, bmi__lt=30).count(),
        "چاق (≥30)": profiles.filter(bmi__gte=30).count(),
        "نامشخص": profiles.filter(Q(bmi__isnull=True) | Q(bmi__lte=0)).count(),
    }

    # Health risks prevalence
    risks = {
        "دیابت": profiles.filter(has_diabetes=True).count(),
        "سیگار": profiles.filter(is_currently_smoking=True).count(),
        "داروی فشار خون": profiles.filter(on_bp_meds=True).count(),
    }

    # Final medical opinion distribution
    fit = profiles.filter(opinion_fit=True).count()
    conditional = profiles.filter(opinion_fit_with_conditions=True).count()
    unfit = profiles.filter(opinion_unfit=True).count()
    unspecified = profiles.filter(
        Q(opinion_fit=False) &
        Q(opinion_fit_with_conditions=False) &
        Q(opinion_unfit=False)
    ).count()

    opinions = {
        "بلامانع": fit,
        "مشروط": conditional,
        "عدم صلاحیت": unfit,
        "نامشخص": unspecified,
    }

    context = {
        'company': company,
        'doctors': doctors,
        'employees': employees,
        'invitations': invitations,

        # Chart payloads (JSON)
        'bmi_bins_json': json.dumps(bmi_bins, cls=DjangoJSONEncoder, ensure_ascii=False),
        'risks_json': json.dumps(risks, cls=DjangoJSONEncoder, ensure_ascii=False),
        'opinions_json': json.dumps(opinions, cls=DjangoJSONEncoder, ensure_ascii=False),
    }
    return render(request, 'usac/manager_dashboard2.html', context)

@login_required(login_url='')
@user_passes_test(is_manager, login_url='')
def member_detail_view(request, user_id):
    """
    Manager views full info of a doctor/employee (basic + health/test history).
    """
    member = get_object_or_404(User, pk=user_id, profile__company__manager=request.user)

    single = PredictionLog.objects.filter(user=member).order_by('-timestamp')
    double = PredictionResult.objects.filter(user=member).order_by('-created_at')
    profile = HealthProfile.objects.filter(user=member).first()

    examined_employees = []
    if member.profile.role == UserProfile.ROLE_DOCTOR:
        examined_employees = User.objects.filter(
            health_profile__examining_doctor=member
        ).distinct().select_related('profile', 'health_profile')
    # Doctor summary stats
    doctor_stats = {}
    if member.profile.role == UserProfile.ROLE_DOCTOR:
        # examined_employees already fetched
        total_examined = examined_employees.count()
        # completed: those with a final opinion
        completed = sum(1 for emp in examined_employees
                        if hasattr(emp, 'health_profile') and emp.health_profile and
                        (emp.health_profile.opinion_fit or emp.health_profile.opinion_fit_with_conditions or emp.health_profile.opinion_unfit))
        pending = total_examined - completed
        doctor_stats = {
            'total_examined': total_examined,
            'completed': completed,
            'pending': pending,
        }
    return render(request, 'usac/member_detail2.html', {
        'member': member,
        'single_results': single,
        'double_results': double,
        'health_profile': profile,
        'examined_employees': examined_employees,
        'doctor_stats': doctor_stats,
    })


# -----------------------
# Utilities
# -----------------------
from django.core.mail import send_mail
from django.conf import settings

def send_test_email(request):
    send_mail(
        subject="Test Email from ROP System",
        message="This is a test email sent from your Django app!",
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=["yazdanbayat2004@gmail.com"],
        fail_silently=False,
    )
    return HttpResponse("Test email sent!")
