# test_analysis/admin.py

from django.contrib import admin
from .models import HealthProfile, PreviousJob, Referral


class PreviousJobInline(admin.TabularInline):
    """Allows editing PreviousJob records from the HealthProfile admin page."""
    model = PreviousJob
    extra = 1
    classes = ('collapse',)
    verbose_name = "Previous Job"
    verbose_name_plural = "2a. Previous Occupational History"


class ReferralInline(admin.TabularInline):
    """Allows editing Referral records from the HealthProfile admin page."""
    model = Referral
    extra = 1
    classes = ('collapse',)
    verbose_name = "Referral"
    verbose_name_plural = "8. Specialist Referrals"


class HealthProfileAdmin(admin.ModelAdmin):
    """
    Custom admin view for the HealthProfile model.
    All fieldsets are visible at once – no collapsed sections.
    """
    list_display = (
        'user', 'current_job_title', 'opinion_fit',
        'opinion_fit_with_conditions', 'opinion_unfit', 'updated_at'
    )
    search_fields = ('user__username', 'user__email', 'current_job_title', 'national_id')
    list_filter = ('opinion_fit', 'opinion_fit_with_conditions', 'is_currently_smoking', 'created_at')
    readonly_fields = ('created_at', 'updated_at')
    inlines = [PreviousJobInline, ReferralInline]

    fieldsets = (
        ("User & AI Advice", {
            'fields': ('user', 'llm_advice')
        }),
        ("1. Personal Information", {
            'fields': (
                ('father_name', 'national_id'),
                ('date_of_birth', 'gender'),
                ('marital_status', 'children_count'),
                'living_province',
                'neighborhood',
                'insurance',
                ('military_service_status', 'military_service_rank'),
                'medical_exemption_reason',
                'work_address',
                'work_phone',
            )
        }),
        ("2. Current Occupational Info", {
            'fields': ('current_job_title', 'current_job_start_date', 'current_job_duties')
        }),
        ("3. Occupational Hazard Assessment", {
            'description': "Check all applicable hazards for the user's roles.",
            'fields': (
                ('hazard_physical_noise', 'hazard_physical_vibration',
                 'hazard_physical_ionizing_radiation', 'hazard_physical_non_ionizing_radiation'),
                'hazard_physical_heat_stress', 'hazard_physical_other',
                ('hazard_chemical_dust', 'hazard_chemical_metal_fumes', 'hazard_chemical_solvents'),
                ('hazard_chemical_pesticides', 'hazard_chemical_acids_bases', 'hazard_chemical_gases'),
                'hazard_chemical_other',
                ('hazard_biological_bites', 'hazard_biological_bacteria',
                 'hazard_biological_virus', 'hazard_biological_parasite'),
                'hazard_biological_other',
                ('hazard_ergonomic_prolonged_sitting_standing', 'hazard_ergonomic_repetitive_work'),
                ('hazard_ergonomic_heavy_lifting', 'hazard_ergonomic_poor_posture'),
                'hazard_ergonomic_other',
                ('hazard_psychological_shift_work', 'hazard_psychological_stressors'),
                'hazard_psychological_other'
            )
        }),
        ("4. Personal & Medical History", {
            'fields': (
                'has_disease_history', 'disease_history_details',
                ('does_symptoms_change_at_work', 'do_colleagues_have_similar_symptoms',
                 'does_symptoms_change_on_holidays'),
                'has_allergies', 'allergy_details',
                'has_hospitalization_history', 'hospitalization_reason',
                'has_surgery_history', 'surgery_details',
                'has_family_cancer_or_chronic_disease', 'family_disease_details',
                'is_on_medication', 'medication_details',
                ('is_currently_smoking', 'has_past_smoking_history'), 'smoking_details',
                'hobbies',
                'has_occupational_accident_history', 'accident_details',
                ('has_absence_over_3_days', 'lives_near_industrial_center',
                 'has_medical_commission_referral'),
            )
        }),
        ("5. Examinations", {
            'fields': (
                ('exam_date', 'exam_weight', 'exam_height'),
                ('exam_blood_pressure', 'exam_pulse_rate'),
                'general_exam_notes', 'eye_exam_notes', 'skin_hair_nails_exam_notes',
                'ent_mouth_exam_notes', 'head_neck_exam_notes', 'lung_exam_notes',
                'cardiovascular_exam_notes', 'abdomen_pelvis_exam_notes',
                'urinary_system_exam_notes', 'musculoskeletal_exam_notes',
                'nervous_system_exam_notes', 'mental_health_exam_notes',
            )
        }),
        ("6 & 7. Lab Tests & Paraclinical", {
            'fields': (
                'lab_test_files',
                ('spirometry_fvc', 'spirometry_fev1', 'spirometry_fev1_fvc_ratio'),
                ('spirometry_fef_25_75', 'spirometry_pef'),
                'spirometry_interpretation',
                'ecg_findings',
                'chest_xray_findings',
                'other_paraclinical_notes',
            )
        }),
        ("9. Final Medical Opinion (Physician's Assessment)", {
            'fields': (
                ('opinion_fit', 'opinion_fit_with_conditions', 'opinion_unfit'),
                'opinion_fit_conditions_details',
                'opinion_unfit_reason',
                'medical_recommendations',
            )
        }),
        ("Timestamps", {
            'fields': (('created_at', 'updated_at'),)
        }),
    )


# Register the model with the custom admin on the default admin site
admin.site.register(HealthProfile, HealthProfileAdmin)