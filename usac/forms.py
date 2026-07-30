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
    email = forms.EmailField(label="Manager email")
    phone = forms.CharField(label="Manager phone", required=False)

    company_name = forms.CharField(label="Company name")
    company_address = forms.CharField(label="Company address", widget=forms.Textarea(attrs={"rows": 3}))
    company_email = forms.EmailField(label="Company email")
    company_phone = forms.CharField(label="Company phone")

    # Faint example text shown inside each empty field.
    _PLACEHOLDERS = {
        "username": "e.g. dr.ahmadi",
        "email": "e.g. manager@clinic.com",
        "phone": "e.g. +98 912 345 6789",
        "password1": "At least 8 characters",
        "password2": "Re-enter your password",
        "company_name": "e.g. Mediverse Clinic",
        "company_address": "e.g. No. 12, Valiasr St., Tehran",
        "company_email": "e.g. info@clinic.com",
        "company_phone": "e.g. +98 21 1234 5678",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            if name in self._PLACEHOLDERS:
                field.widget.attrs.setdefault("placeholder", self._PLACEHOLDERS[name])

    class Meta:
        model = User
        # Only User model fields in Meta
        fields = ['username', 'email', 'password1', 'password2']

    def clean_company_name(self):
        name = self.cleaned_data['company_name'].strip()
        # Enforce unique company name (case-insensitive) BEFORE hitting DB unique constraint
        if Company.objects.filter(name__iexact=name).exists():
            raise ValidationError("This company name is already registered. Please choose another one.")
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
    email = forms.EmailField(label="Email")
    phone = forms.CharField(label="Phone", required=False)
    national_code = forms.CharField(label="National code (10 digits)", max_length=10)

    # Faint example text shown inside each empty field.
    _PLACEHOLDERS = {
        "username": "e.g. dr.ahmadi",
        "email": "e.g. you@example.com",
        "phone": "e.g. +98 912 345 6789",
        "national_code": "e.g. 1234567890",
        "password1": "At least 8 characters",
        "password2": "Re-enter your password",
    }

    # role is injected by the view (doctor/employee)
    def __init__(self, *args, **kwargs):
        self.role = kwargs.pop('role')
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            if name in self._PLACEHOLDERS:
                field.widget.attrs.setdefault("placeholder", self._PLACEHOLDERS[name])

    class Meta:
        model = User
        # Only User fields in Meta
        fields = ['username', 'email', 'password1', 'password2']

    def clean_national_code(self):
        nc = self.cleaned_data['national_code']
        if len(nc) != 10 or not nc.isdigit():
            raise ValidationError("National code must be exactly 10 digits.")
        return nc

    def clean(self):
        cleaned = super().clean()
        if not _has_up_table():
            raise ValidationError("The system is still initializing. Please try again shortly.")

        nc = cleaned.get('national_code')
        try:
            inv = Invitation.objects.get(national_code=nc, role=self.role, used_by__isnull=True)
        except Invitation.DoesNotExist:
            raise ValidationError("No active invitation found for this national code. Please contact your company manager.")
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
