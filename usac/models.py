from django.conf import settings
from django.db import models
from django.core.validators import RegexValidator
from django.utils import timezone

User = settings.AUTH_USER_MODEL

class Company(models.Model):
    name = models.CharField(max_length=255, unique=True)
    address = models.TextField()
    email = models.EmailField()
    phone = models.CharField(max_length=32, validators=[RegexValidator(r'^[0-9+\-()\s]+$')])
    created_at = models.DateTimeField(auto_now_add=True)
    manager = models.OneToOneField('auth.User', on_delete=models.PROTECT, related_name='managed_company')
    def __str__(self): return self.name

class UserProfile(models.Model):
    ROLE_MANAGER = 'manager'
    ROLE_DOCTOR = 'doctor'
    ROLE_EMPLOYEE = 'employee'
    ROLE_CHOICES = [
        (ROLE_MANAGER, 'Manager'),
        (ROLE_DOCTOR, 'Doctor'),
        (ROLE_EMPLOYEE, 'Employee'),
    ]
    EXAM_PRE_EMPLOYMENT = 'pre_employment'
    EXAM_PERIODIC = 'periodic'
    EXAM_RETURN_TO_WORK = 'return_to_work'
    EXAM_TYPE_CHOICES = [
        (EXAM_PRE_EMPLOYMENT, 'Pre-Employment Medical Examination'),
        (EXAM_PERIODIC, 'Periodic Medical Examination'),
        (EXAM_RETURN_TO_WORK, 'Return-to-Work Examination'),
    ]
    user = models.OneToOneField('auth.User', on_delete=models.CASCADE, related_name='profile')
    role = models.CharField(max_length=16, choices=ROLE_CHOICES)
    national_code = models.CharField(max_length=10, blank=True, null=True,
                                     validators=[RegexValidator(r'^\d{10}$', message='کد ملی باید ۱۰ رقم باشد.')])
    company = models.ForeignKey(Company, on_delete=models.SET_NULL, null=True, blank=True, related_name='members')
    phone = models.CharField(max_length=32, blank=True, null=True)
    examination_type = models.CharField(max_length=32, choices=EXAM_TYPE_CHOICES, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    def __str__(self): return f"{self.user.username} ({self.role})"

class Invitation(models.Model):
    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name='invitations')
    role = models.CharField(max_length=16, choices=UserProfile.ROLE_CHOICES)
    national_code = models.CharField(max_length=10, validators=[RegexValidator(r'^\d{10}$')])
    examination_type = models.CharField(max_length=32, choices=UserProfile.EXAM_TYPE_CHOICES, blank=True, null=True)
    note = models.CharField(max_length=255, blank=True)
    used_by = models.OneToOneField('auth.User', on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    used_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = ('company', 'national_code')

    def mark_used(self, user):
        self.used_by = user
        self.used_at = timezone.now()
        self.save()

    def __str__(self): return f"{self.national_code} → {self.company.name} ({self.role})"
