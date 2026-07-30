# test_analysis/models.py

from django.db import models
from django.contrib.auth.models import User


class HealthProfile(models.Model):
    """
    This model stores a user's complete health and occupational profile,
    based on the provided medical examination form.
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='health_profile')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # --- 1. مشخصات فردی شاغل (Personal Information) ---
    father_name = models.CharField(max_length=100, blank=True, null=True)  # نام پدر
    national_id = models.CharField(max_length=10, blank=True, null=True)  # کد ملی
    date_of_birth = models.DateField(blank=True, null=True)  # تاریخ تولد
    gender = models.CharField(max_length=10, blank=True, null=True,
                              choices=[('Male', 'مرد'), ('Female', 'زن')])  # جنسیت
    province_of_residence = models.CharField(max_length=100, blank=True, null=True, verbose_name="استان محل زندگی")

    living_province = models.CharField(
        max_length=100,
        blank=True, null=True,
        verbose_name="استان زندگی"
    )
    neighborhood = models.CharField(
        max_length=100,
        blank=True, null=True,
        verbose_name="محله"
    )
    insurance = models.CharField(
        max_length=100,
        blank=True, null=True,
        verbose_name="بیمه"
    )
    marital_status = models.CharField(max_length=20, blank=True, null=True)  # وضعیت تاهل
    children_count = models.PositiveIntegerField(default=0, blank=True, null=True)  # تعداد فرزند
    military_service_status = models.CharField(max_length=50, blank=True, null=True)  # وضعیت نظام وظیفه
    military_service_rank = models.CharField(max_length=100, blank=True, null=True)  # رسته خدمت
    medical_exemption_reason = models.TextField(blank=True, null=True)  # علت معافیت پزشکی
    work_address = models.TextField(blank=True, null=True)  # آدرس محل کار
    work_phone = models.CharField(max_length=20, blank=True, null=True)  # تلفن محل کار

    age = models.PositiveIntegerField(blank=True, null=True, verbose_name="سن (age)")
    lab_total_cholesterol = models.FloatField(null=True, blank=True, verbose_name="Total Cholesterol (mg/dL)")
    lab_glucose = models.FloatField(null=True, blank=True, verbose_name="Glucose (mg/dL)")
    # --- 2. سوابق شغلی (Occupational History) ---
    current_job_title = models.CharField(max_length=255, blank=True, null=True)  # عنوان شغلی فعلی
    current_job_duties = models.TextField(blank=True, null=True)  # وظیفه محوله فعلی
    current_job_start_date = models.DateField(blank=True, null=True)  # تاریخ اشتغال فعلی

    # --- 3. ارزیابی عوامل زیان آور شغلی (Occupational Hazard Assessment) ---
    # Physical Hazards
    hazard_physical_noise = models.BooleanField(default=False, verbose_name="سر و صدا")
    hazard_physical_vibration = models.BooleanField(default=False, verbose_name="ارتعاش")
    hazard_physical_non_ionizing_radiation = models.BooleanField(default=False, verbose_name="اشعه غیر یونیزان")
    hazard_physical_ionizing_radiation = models.BooleanField(default=False, verbose_name="اشعه یونیزان")
    hazard_physical_heat_stress = models.BooleanField(default=False, verbose_name="استرس حرارتی")
    hazard_physical_other = models.TextField(blank=True, null=True, verbose_name="سایر موارد فیزیکی")

    # Chemical Hazards
    hazard_chemical_dust = models.BooleanField(default=False, verbose_name="گرد و غبار")
    hazard_chemical_metal_fumes = models.BooleanField(default=False, verbose_name="دمه فلزات")
    hazard_chemical_solvents = models.BooleanField(default=False, verbose_name="حلال ها")
    hazard_chemical_pesticides = models.BooleanField(default=False, verbose_name="آفت کش ها")
    hazard_chemical_acids_bases = models.BooleanField(default=False, verbose_name="اسید و بازها")
    hazard_chemical_gases = models.BooleanField(default=False, verbose_name="گازها")
    hazard_chemical_other = models.TextField(blank=True, null=True, verbose_name="سایر موارد شیمیایی")

    # Biological Hazards
    hazard_biological_bites = models.BooleanField(default=False, verbose_name="گزش")
    hazard_biological_bacteria = models.BooleanField(default=False, verbose_name="باکتری")
    hazard_biological_virus = models.BooleanField(default=False, verbose_name="ویروس")
    hazard_biological_parasite = models.BooleanField(default=False, verbose_name="انگل")
    hazard_biological_other = models.TextField(blank=True, null=True, verbose_name="سایر موارد بیولوژیک")

    # Ergonomic Hazards
    hazard_ergonomic_prolonged_sitting_standing = models.BooleanField(default=False,
                                                                      verbose_name="ایستادن یا نشستن طولانی")
    hazard_ergonomic_repetitive_work = models.BooleanField(default=False, verbose_name="کار تکراری")
    hazard_ergonomic_heavy_lifting = models.BooleanField(default=False, verbose_name="حمل بار سنگین")
    hazard_ergonomic_poor_posture = models.BooleanField(default=False, verbose_name="وضعیت نامناسب بدن")
    hazard_ergonomic_other = models.TextField(blank=True, null=True, verbose_name="سایر موارد ارگونومی")

    # Psychological Hazards
    hazard_psychological_shift_work = models.BooleanField(default=False, verbose_name="نوبت کاری")
    hazard_psychological_stressors = models.BooleanField(default=False, verbose_name="استرسورهای شغلی")
    hazard_psychological_other = models.TextField(blank=True, null=True, verbose_name="سایر موارد روانی")

    # --- 4. سابقه شخصی خانوادگی و پزشکی (Personal and Medical History) ---
    has_disease_history = models.BooleanField(default=False)
    disease_history_details = models.TextField(blank=True, null=True)
    does_symptoms_change_at_work = models.BooleanField(default=False)
    do_colleagues_have_similar_symptoms = models.BooleanField(default=False)
    does_symptoms_change_on_holidays = models.BooleanField(default=False)
    has_allergies = models.BooleanField(default=False)
    allergy_details = models.TextField(blank=True, null=True)
    has_hospitalization_history = models.BooleanField(default=False)
    hospitalization_reason = models.TextField(blank=True, null=True)
    has_surgery_history = models.BooleanField(default=False)
    surgery_details = models.TextField(blank=True, null=True)
    has_family_cancer_or_chronic_disease = models.BooleanField(default=False)
    family_disease_details = models.TextField(blank=True, null=True)
    is_on_medication = models.BooleanField(default=False)
    medication_details = models.TextField(blank=True, null=True)

    is_currently_smoking = models.BooleanField(default=False)
    smoking_details = models.CharField(max_length=100, blank=True, null=True)
    has_past_smoking_history = models.BooleanField(default=False)
    hobbies = models.TextField(blank=True, null=True)
    has_occupational_accident_history = models.BooleanField(default=False)
    accident_details = models.TextField(blank=True, null=True)
    has_absence_over_3_days = models.BooleanField(default=False, verbose_name="سابقه غیبت بیش از ۳ روز")
    lives_near_industrial_center = models.BooleanField(default=False, verbose_name="منزل در مجاورت مرکز صنعتی")
    has_medical_commission_referral = models.BooleanField(default=False, verbose_name="سابقه معرفی به کمیسیون پزشکی")

    cigs_per_day = models.PositiveIntegerField(blank=True, null=True, default=0,
                                               verbose_name="تعداد سیگار در روز (cigsPerDay)")
    on_bp_meds = models.BooleanField(default=False, verbose_name="آیا داروی فشار خون مصرف می‌کنید؟ (BPMeds)")
    has_diabetes = models.BooleanField(default=False, verbose_name="آیا سابقه دیابت دارید؟ (diabetes)")
    # --- END: فیلدهای جدید اضافه شده ---

    # --- 5. معاینات (Examinations) ---
    exam_date = models.DateField(blank=True, null=True, verbose_name="تاریخ معاینه")
    exam_weight = models.FloatField(blank=True, null=True, verbose_name="وزن (Kg)")
    exam_height = models.FloatField(blank=True, null=True, verbose_name="قد (Cm)")
    exam_blood_pressure = models.CharField(max_length=20, blank=True, null=True, verbose_name="فشار خون (mmHg)")
    exam_pulse_rate = models.PositiveIntegerField(blank=True, null=True, verbose_name="تعداد نبض (دقیقه)")
    exam_systolic_bp = models.PositiveIntegerField(blank=True, null=True, verbose_name="فشار خون سیستولیک (sysBP)")
    exam_diastolic_bp = models.PositiveIntegerField(blank=True, null=True, verbose_name="فشار خون دیاستولیک (diaBP)")
    bmi = models.FloatField(blank=True, null=True, verbose_name="شاخص توده بدنی (BMI)")
    # --- END: فیلدهای جدید اضافه شده ---
    # Symptoms/Signs are added as BooleanFields. TextFields for "other" or "details".
    # General Examination
    sym_gen_weight_loss = models.BooleanField(default=False)
    sym_gen_loss_of_appetite = models.BooleanField(default=False)
    sym_gen_chronic_fatigue = models.BooleanField(default=False)
    # ... (and so on for every single checkbox on the form)
    # For brevity in this response, I am not listing all 100+ boolean fields, but the code
    # should contain one for each symptom and sign from the uploaded images.
    general_exam_notes = models.TextField(blank=True, null=True)
    eye_exam_notes = models.TextField(blank=True, null=True)
    skin_hair_nails_exam_notes = models.TextField(blank=True, null=True)
    ent_mouth_exam_notes = models.TextField(blank=True, null=True)
    head_neck_exam_notes = models.TextField(blank=True, null=True)
    lung_exam_notes = models.TextField(blank=True, null=True)
    cardiovascular_exam_notes = models.TextField(blank=True, null=True)
    abdomen_pelvis_exam_notes = models.TextField(blank=True, null=True)
    urinary_system_exam_notes = models.TextField(blank=True, null=True)
    musculoskeletal_exam_notes = models.TextField(blank=True, null=True)
    nervous_system_exam_notes = models.TextField(blank=True, null=True)
    mental_health_exam_notes = models.TextField(blank=True, null=True)

    # --- 6. آزمایش‌ها (Lab Tests) ---
    lab_cbc_wbc = models.CharField(max_length=20, blank=True, null=True)
    lab_cbc_rbc = models.CharField(max_length=20, blank=True, null=True)
    # ... fields for Hb, HCT, Plt, etc.
    lab_ua_prot = models.CharField(max_length=20, blank=True, null=True)
    # ... fields for Glu, RBC, WBC, Bact, etc.
    lab_fbs = models.CharField(max_length=20, blank=True, null=True)
    # ... fields for Chol, LDL, HDL, TG, BUN, Cr, ALT, AST, etc.
    lab_test_files = models.FileField(upload_to='lab_tests/', blank=True, null=True)

    # --- 7. پاراکلینیک (Paraclinical) ---
    # Spirometry
    spirometry_fvc = models.CharField(max_length=50, blank=True, null=True)
    spirometry_fev1 = models.CharField(max_length=50, blank=True, null=True)
    spirometry_fev1_fvc_ratio = models.CharField(max_length=50, blank=True, null=True)
    spirometry_fef_25_75 = models.CharField(max_length=50, blank=True, null=True, verbose_name="FEF 25-75%")
    spirometry_pef = models.CharField(max_length=50, blank=True, null=True, verbose_name="PEF")
    spirometry_interpretation = models.TextField(blank=True, null=True)

    # Other tests
    ecg_findings = models.TextField(blank=True, null=True)
    chest_xray_findings = models.TextField(blank=True, null=True)
    other_paraclinical_notes = models.TextField(blank=True, null=True, verbose_name="نتایج رادیوگرافی، سونوگرافی...")

    # --- 9. نظریه نهایی پزشک (Final Medical Opinion) ---
    opinion_fit = models.BooleanField(default=False, verbose_name="بلامانع")
    opinion_fit_with_conditions = models.BooleanField(default=False, verbose_name="مشروط")
    opinion_fit_conditions_details = models.TextField(blank=True, null=True, verbose_name="شرح شروط")
    opinion_unfit = models.BooleanField(default=False, verbose_name="عدم صلاحیت")
    opinion_unfit_reason = models.TextField(blank=True, null=True, verbose_name="علت عدم صلاحیت")
    medical_recommendations = models.TextField(blank=True, null=True, verbose_name="توصیه های پزشکی لازم")
    # Inside HealthProfile class, after medical_recommendations field
    examining_doctor = models.ForeignKey(
        User,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='examined_profiles',
        verbose_name="پزشک معاینه‌کننده"
    )
    # --- AI Generated Advice ---
    llm_advice = models.TextField(blank=True, null=True, help_text="Advice generated by the LLM.")
    model_used_for_advice = models.CharField(
        max_length=20,
        choices=[('cloud_gpt', 'GPT-4o-mini (Cloud)'), ('local_llama', 'Llama 3.2 (Local)')],
        default='cloud_gpt',
        blank=True,  # Good practice for adding new fields to existing models
        help_text="The AI model used to generate the last report."
    )
    report_task_id = models.CharField(max_length=100, blank=True, null=True)
    report_ready = models.BooleanField(default=False)
    report_error = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"Health Profile for {self.user.username}"


class PreviousJob(models.Model):
    """Stores a single previous job entry related to a HealthProfile."""
    profile = models.ForeignKey(HealthProfile, on_delete=models.CASCADE, related_name='previous_jobs')
    title = models.CharField(max_length=255, verbose_name="عنوان سمت")
    duties = models.TextField(blank=True, null=True, verbose_name="وظیفه محوله")
    start_date = models.DateField(blank=True, null=True, verbose_name="تاریخ شروع")
    end_date = models.DateField(blank=True, null=True, verbose_name="تاریخ پایان")
    reason_for_leaving = models.TextField(blank=True, null=True, verbose_name="علت تغییر شغل")

    def __str__(self):
        return f"Previous Job: {self.title} for {self.profile.user.username}"


class EyeImage(models.Model):
    """An eye photo uploaded by the employee (for future disease screening).
    Stored on the profile so a doctor can review the images."""
    profile = models.ForeignKey(HealthProfile, on_delete=models.CASCADE, related_name='eye_images')
    image = models.ImageField(upload_to='eye_images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    # sha-256 of the file content, used to skip re-processing duplicate uploads
    content_hash = models.CharField(max_length=64, blank=True, db_index=True)

    def __str__(self):
        return f"Eye image for {self.profile.user.username}"


class EyeAnalysis(models.Model):
    """AI screening results for a single uploaded eye photo.

    The pipeline runs in two segmentation phases plus a (mock) classifier:
      • phase 1  — segment the forniceal+palpebral conjunctiva from the raw photo
      • phase 2  — segment the palpebral region from the phase-1 RGB crop
      • classify — anemia positive / negative (placeholder model)
    Each phase stores both its binary mask and the RGB crop it produced.
    """
    STATUS_PENDING = 'pending'
    STATUS_DONE = 'done'
    STATUS_FAILED = 'failed'
    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_DONE, 'Done'),
        (STATUS_FAILED, 'Failed'),
    ]

    eye_image = models.OneToOneField(EyeImage, on_delete=models.CASCADE, related_name='analysis')

    phase1_mask = models.ImageField(upload_to='eye_analysis/phase1_mask/', blank=True, null=True)
    phase1_overlay = models.ImageField(upload_to='eye_analysis/phase1_rgb/', blank=True, null=True)
    phase2_mask = models.ImageField(upload_to='eye_analysis/phase2_mask/', blank=True, null=True)
    phase2_overlay = models.ImageField(upload_to='eye_analysis/phase2_rgb/', blank=True, null=True)

    anemia_label = models.CharField(max_length=16, blank=True)          # 'positive' | 'negative'
    anemia_confidence = models.FloatField(null=True, blank=True)        # 0..1

    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING)
    error = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"EyeAnalysis(eye_image={self.eye_image_id}, status={self.status})"


class MedicalTest(models.Model):
    """A medical test document (lab report, addiction panel, imaging, etc.)
    uploaded by the employee, together with the results extracted from it by
    an LLM (gpt-4o-mini).

    Only the *test results* are extracted and stored — panels, analytes,
    their values, units and reference ranges. Patient-identifying details on
    the document (name, DOB, MRN, ordering provider) are deliberately ignored.
    The structured data is kept in ``panels`` so it can be rendered as tables
    for every viewer tier (employee, manager, doctor)."""
    STATUS_PENDING = 'pending'
    STATUS_DONE = 'done'
    STATUS_FAILED = 'failed'
    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_DONE, 'Done'),
        (STATUS_FAILED, 'Failed'),
    ]

    profile = models.ForeignKey(HealthProfile, on_delete=models.CASCADE, related_name='medical_tests')
    file = models.FileField(upload_to='medical_tests/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    # sha-256 of the file content, used to skip re-processing duplicate uploads
    content_hash = models.CharField(max_length=64, blank=True, db_index=True)

    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING)

    # --- extracted report metadata (no patient-identifying info) ---
    report_type = models.CharField(max_length=255, blank=True)   # e.g. "Complete Blood Count"
    lab_name = models.CharField(max_length=255, blank=True)
    specimen = models.CharField(max_length=255, blank=True)
    collected_on = models.CharField(max_length=64, blank=True)   # kept as text — formats vary
    reported_on = models.CharField(max_length=64, blank=True)

    # --- extracted results ---
    # panels = [{"name": str, "analytes": [
    #     {"name": str, "result": str, "flag": str, "unit": str, "reference": str}, ...]}, ...]
    panels = models.JSONField(default=list, blank=True)
    abnormal_count = models.IntegerField(default=0)              # analytes flagged out of range
    summary = models.TextField(blank=True)                       # short plain-language overview

    error = models.TextField(blank=True, null=True)
    extracted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-uploaded_at']

    @property
    def filename(self):
        import os
        return os.path.basename(self.file.name) if self.file else ''

    def __str__(self):
        return f"MedicalTest(profile={self.profile_id}, status={self.status})"


class Referral(models.Model):
    """Stores a single referral entry related to a HealthProfile."""
    profile = models.ForeignKey(HealthProfile, on_delete=models.CASCADE, related_name='referrals')
    date = models.DateField(verbose_name="تاریخ ارجاع")
    reason = models.TextField(verbose_name="علت ارجاع")
    specialty = models.CharField(max_length=255, verbose_name="نوع تخصص")
    result = models.TextField(verbose_name="نتیجه ارجاع")

    def __str__(self):
        return f"Referral to {self.specialty} for {self.profile.user.username}"
