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
    marital_status = models.CharField(max_length=20, blank=True, null=True)  # وضعیت تاهل
    children_count = models.PositiveIntegerField(default=0, blank=True, null=True)  # تعداد فرزند
    military_service_status = models.CharField(max_length=50, blank=True, null=True)  # وضعیت نظام وظیفه
    medical_exemption_reason = models.TextField(blank=True, null=True)  # علت معافیت پزشکی

    # --- 2. سوابق شغلی (Occupational History) ---
    # مشاغل فعلی
    current_job_title = models.CharField(max_length=255, blank=True, null=True)  # عنوان شغلی فعلی
    current_job_duties = models.TextField(blank=True, null=True)  # وظیفه محوله فعلی

    # مشاغل قبلی
    previous_job_title = models.CharField(max_length=255, blank=True, null=True)  # عنوان شغلی قبلی
    previous_job_duties = models.TextField(blank=True, null=True)  # وظیفه محوله قبلی
    job_change_reason = models.TextField(blank=True, null=True)  # علت تغییر شغل

    # ارزیابی عوامل زیان آور شغلی (Occupational Hazard Assessment)
    # برای سادگی، این فیلدها را به صورت متنی در نظر می‌گیریم تا کاربر موارد را لیست کند
    physical_hazards = models.TextField(blank=True, null=True, help_text="e.g., Noise, Vibration, Radiation")  # فیزیکی
    chemical_hazards = models.TextField(blank=True, null=True, help_text="e.g., Dust, Solvents, Acids")  # شیمیایی
    biological_hazards = models.TextField(blank=True, null=True, help_text="e.g., Viruses, Bacteria, Bites")  # بیولوژیک
    ergonomic_hazards = models.TextField(blank=True, null=True,
                                         help_text="e.g., Prolonged standing, Repetitive tasks")  # ارگونومی
    psychological_hazards = models.TextField(blank=True, null=True, help_text="e.g., Shift work, Stress")  # روانی

    # --- 4. سابقه شخصی خانوادگی و پزشکی (Personal and Medical History) ---
    has_disease_history = models.BooleanField(default=False)  # آیا سابقه بیماری دارید؟
    disease_history_details = models.TextField(blank=True, null=True)  # توضیحات بیماری

    does_symptoms_change_at_work = models.BooleanField(default=False)  # آیا علایم در محیط کار تغییر می‌کند؟
    do_colleagues_have_similar_symptoms = models.BooleanField(default=False)  # آیا همکاران علایم مشابه دارند؟
    does_symptoms_change_on_holidays = models.BooleanField(default=False)  # آیا علایم در تعطیلات تغییر می‌کند؟
    has_allergies = models.BooleanField(default=False)  # آیا حساسیت دارید؟
    allergy_details = models.TextField(blank=True, null=True)  # توضیحات حساسیت

    has_hospitalization_history = models.BooleanField(default=False)  # سابقه بستری
    hospitalization_reason = models.TextField(blank=True, null=True)  # علت بستری

    has_surgery_history = models.BooleanField(default=False)  # سابقه جراحی
    surgery_details = models.TextField(blank=True, null=True)  # توضیحات جراحی

    has_family_cancer_or_chronic_disease = models.BooleanField(default=False)  # سابقه سرطان یا بیماری مزمن در فامیل
    family_disease_details = models.TextField(blank=True, null=True)  # توضیحات بیماری فامیل

    is_on_medication = models.BooleanField(default=False)  # آیا داروی خاصی مصرف می‌کنید؟
    medication_details = models.TextField(blank=True, null=True)  # نام داروها

    is_currently_smoking = models.BooleanField(default=False)  # آیا اکنون سیگار می‌کشید؟
    smoking_details = models.CharField(max_length=100, blank=True, null=True)  # تعداد و مدت استعمال

    has_past_smoking_history = models.BooleanField(default=False)  # آیا سابقه قبلی مصرف سیگار دارید؟

    hobbies = models.TextField(blank=True, null=True)  # ورزش یا سرگرمی

    has_occupational_accident_history = models.BooleanField(default=False)  # سابقه حادثه شغلی
    accident_details = models.TextField(blank=True, null=True)  # نوع و علت آسیب

    # --- 5. معاینات (Examinations) ---
    # به جای تک تک چک باکس‌ها، یک فیلد متنی برای توضیحات هر ارگان قرار می‌دهیم
    general_exam_notes = models.TextField(blank=True, null=True)  # عمومی
    eye_exam_notes = models.TextField(blank=True, null=True)  # چشم
    skin_hair_nails_exam_notes = models.TextField(blank=True, null=True)  # پوست، مو و ناخن
    ent_mouth_exam_notes = models.TextField(blank=True, null=True)  # گوش، حلق، بینی و دهان
    head_neck_exam_notes = models.TextField(blank=True, null=True)  # سر و گردن
    lung_exam_notes = models.TextField(blank=True, null=True)  # ریه
    cardiovascular_exam_notes = models.TextField(blank=True, null=True)  # قلب و عروق
    abdomen_pelvis_exam_notes = models.TextField(blank=True, null=True)  # شکم و لگن
    urinary_system_exam_notes = models.TextField(blank=True, null=True)  # کلیه و مجاری ادراری
    musculoskeletal_exam_notes = models.TextField(blank=True, null=True)  # اسکلتی و عضلانی
    nervous_system_exam_notes = models.TextField(blank=True, null=True)  # سیستم عصبی
    mental_health_exam_notes = models.TextField(blank=True, null=True)  # اعصاب و روان

    # --- 6 & 7. آزمایش‌ها و پاراکلینیک (Lab Tests & Paraclinical) ---
    # فیلدهای آپلود فایل
    lab_test_file_1 = models.FileField(upload_to='lab_tests/', blank=True, null=True)
    lab_test_file_2 = models.FileField(upload_to='lab_tests/', blank=True, null=True)
    lab_test_file_3 = models.FileField(upload_to='lab_tests/', blank=True, null=True)

    # اسپیرومتری
    spirometry_fvc = models.CharField(max_length=50, blank=True, null=True)
    spirometry_fev1 = models.CharField(max_length=50, blank=True, null=True)
    spirometry_fev1_fvc_ratio = models.CharField(max_length=50, blank=True, null=True)
    spirometry_interpretation = models.TextField(blank=True, null=True)  # تفسیر

    # سایر موارد
    ecg_findings = models.TextField(blank=True, null=True)  # یافته‌های ECG
    chest_xray_findings = models.TextField(blank=True, null=True)  # یافته‌های CXR
    referral_notes = models.TextField(blank=True, null=True)  # نتایج ارجاع‌ها

    # --- فیلد نهایی برای ذخیره توصیه LLM ---
    llm_advice = models.TextField(blank=True, null=True, help_text="Advice generated by the LLM.")

    def __str__(self):
        return f"Health Profile for {self.user.username}"