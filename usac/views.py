from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.db import connection
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse

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
            return redirect('dilemma')
        return render(request, 'usac/login.html', {'error': 'Invalid credentials'})

    return render(request, 'usac/login.html')


# -----------------------
# Dilemma (role-aware, table-safe)
# -----------------------
@login_required(login_url='')
def dilemma_view(request):
    """
    Render dilemma without crashing if usac_userprofile isn't migrated yet.
    Never access request.user.profile directly here.
    """
    role = None
    if _table_exists('usac_userprofile'):
        try:
            prof = UserProfile.objects.filter(user=request.user).only('role').first()
            if prof:
                role = prof.role
        except Exception:
            role = None

    is_manager_flag = (role == UserProfile.ROLE_MANAGER)
    is_doctor_flag = (role == UserProfile.ROLE_DOCTOR)
    is_employee_flag = (role == UserProfile.ROLE_EMPLOYEE)

    return render(
        request,
        'usac/dilemma.html',
        {
            'role': role,
            'is_manager': is_manager_flag,
            'is_doctor': is_doctor_flag,
            'is_employee': is_employee_flag,
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
    return render(request, 'usac/signup_choose_role.html', {'form': form})


def signup_manager_view(request):
    if request.method == 'POST':
        form = ManagerSignupForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "ثبت نام مدیر و شرکت با موفقیت انجام شد. حالا وارد شوید.")
            return redirect('login')
    else:
        form = ManagerSignupForm()
    return render(request, 'usac/signup_manager.html', {'form': form})


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
    return render(request, 'usac/signup_staff.html', {'form': form, 'role': role})


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
        }
        messages.warning(request, "شرکت شما یافت نشد. اگر همین الان ثبت‌نام کرده‌اید، یکبار خارج و وارد شوید یا ثبت‌نام مدیر را دوباره انجام دهید.")
        return render(request, 'usac/manager_dashboard.html', context)

    # Normal flow when company exists
    doctors = User.objects.filter(
        profile__company=company,
        profile__role=UserProfile.ROLE_DOCTOR
    ).select_related('profile')

    employees = User.objects.filter(
        profile__company=company,
        profile__role=UserProfile.ROLE_EMPLOYEE
    ).select_related('profile')

    invitations = Invitation.objects.filter(company=company).order_by('-created_at')

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

    context = {
        'company': company,
        'doctors': doctors,
        'employees': employees,
        'invitations': invitations,
    }
    return render(request, 'usac/manager_dashboard.html', context)


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

    return render(request, 'usac/member_detail.html', {
        'member': member,
        'single_results': single,
        'double_results': double,
        'health_profile': profile,
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
