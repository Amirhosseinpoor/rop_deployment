from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.db import connection
from .models import Company, UserProfile, Invitation

def _has_up_table():
    try:
        return 'usac_userprofile' in connection.introspection.table_names()
    except Exception:
        return False


class RoleChoiceForm(forms.Form):
    role = forms.ChoiceField(
        choices=UserProfile.ROLE_CHOICES,
        widget=forms.RadioSelect,
        label="انتخاب موقعیت شغلی"
    )


class ManagerSignupForm(UserCreationForm):
    # Extra non-User fields
    email = forms.EmailField(label="ایمیل مدیر")
    phone = forms.CharField(label="شماره مدیر", required=False)

    company_name = forms.CharField(label="نام شرکت")
    company_address = forms.CharField(label="آدرس دقیق شرکت", widget=forms.Textarea(attrs={"rows": 3}))
    company_email = forms.EmailField(label="ایمیل شرکت")
    company_phone = forms.CharField(label="شماره شرکت")

    class Meta:
        model = User
        # Only User model fields in Meta
        fields = ['username', 'email', 'password1', 'password2']

    def clean_company_name(self):
        name = self.cleaned_data['company_name'].strip()
        # Enforce unique company name (case-insensitive) BEFORE hitting DB unique constraint
        if Company.objects.filter(name__iexact=name).exists():
            raise ValidationError("نام شرکت قبلاً ثبت شده است. لطفاً نام دیگری انتخاب کنید.")
        return name

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()

        # Create profile + company once tables exist
        if _has_up_table():
            profile, _ = UserProfile.objects.get_or_create(user=user)
            profile.role = UserProfile.ROLE_MANAGER
            profile.phone = self.cleaned_data.get('phone') or ''
            profile.save()

            # Company (name uniqueness already validated)
            company = Company.objects.create(
                name=self.cleaned_data['company_name'],
                address=self.cleaned_data['company_address'],
                email=self.cleaned_data['company_email'],
                phone=self.cleaned_data['company_phone'],
                manager=user
            )

            profile.company = company
            profile.save(update_fields=['company'])

        return user


class StaffSignupForm(UserCreationForm):
    # Extra non-User fields
    email = forms.EmailField(label="ایمیل")
    phone = forms.CharField(label="شماره", required=False)
    national_code = forms.CharField(label="کد ملی (۱۰ رقمی)", max_length=10)

    # role is injected by the view (doctor/employee)
    def __init__(self, *args, **kwargs):
        self.role = kwargs.pop('role')
        super().__init__(*args, **kwargs)

    class Meta:
        model = User
        # Only User fields in Meta
        fields = ['username', 'email', 'password1', 'password2']

    def clean_national_code(self):
        nc = self.cleaned_data['national_code']
        if len(nc) != 10 or not nc.isdigit():
            raise ValidationError("کد ملی باید ۱۰ رقم باشد.")
        return nc

    def clean(self):
        cleaned = super().clean()
        if not _has_up_table():
            raise ValidationError("زیرساخت سیستم در حال آماده‌سازی است. کمی بعد تلاش کنید.")

        nc = cleaned.get('national_code')
        try:
            inv = Invitation.objects.get(national_code=nc, role=self.role, used_by__isnull=True)
        except Invitation.DoesNotExist:
            raise ValidationError("برای این کد ملی دعوت فعالی یافت نشد. لطفاً با مدیر شرکت خود هماهنگ کنید.")
        self._invitation = inv
        return cleaned

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()

        profile, _ = UserProfile.objects.get_or_create(user=user)
        profile.role = self.role
        profile.national_code = self.cleaned_data['national_code']
        profile.phone = self.cleaned_data.get('phone') or ''
        profile.company = self._invitation.company
        profile.save()

        self._invitation.mark_used(user)
        return user
