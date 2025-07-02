from django.contrib import admin
from .models import HealthProfile


@admin.register(HealthProfile)
class HealthProfileAdmin(admin.ModelAdmin):
    """
    Custom admin view for the HealthProfile model.
    Uses fieldsets to organize the vast number of fields into logical groups.
    """
    # Fields to display in the main list view of profiles
    list_display = ('user', 'current_job_title', 'updated_at')

    # Fields to search by in the admin list view
    search_fields = ('user__username', 'user__email', 'current_job_title')

    # Filters to show on the right sidebar
    list_filter = ('created_at', 'marital_status', 'is_currently_smoking')

    # Make system-generated fields read-only
    readonly_fields = ('created_at', 'updated_at', 'llm_advice')

    # Organize the detail view into collapsible sections
    fieldsets = (
        ("User Information", {
            'fields': ('user',)
        }),
        ("1. Personal & Occupational Info", {
            'classes': ('collapse',),
            'fields': (
                ('marital_status', 'children_count'),
                'military_service_status',
                'medical_exemption_reason',
                'current_job_title',
                'current_job_duties',
                'previous_job_title',
                'previous_job_duties',
                'job_change_reason',
            )
        }),
        ("2. Occupational Hazard Assessment", {
            'classes': ('collapse',),
            'fields': (
                'physical_hazards',
                'chemical_hazards',
                'biological_hazards',
                'ergonomic_hazards',
                'psychological_hazards',
            )
        }),
        ("3. Medical History", {
            'classes': ('collapse',),
            'fields': (
                'has_disease_history',
                'disease_history_details',
                'does_symptoms_change_at_work',
                'do_colleagues_have_similar_symptoms',
                'does_symptoms_change_on_holidays',
                'has_allergies',
                'allergy_details',
                'has_hospitalization_history',
                'hospitalization_reason',
                'has_surgery_history',
                'surgery_details',
                'has_family_cancer_or_chronic_disease',
                'family_disease_details',
                'is_on_medication',
                'medication_details',
                ('is_currently_smoking', 'has_past_smoking_history'),
                'smoking_details',
                'hobbies',
                'has_occupational_accident_history',
                'accident_details',
            )
        }),
        ("4. Examination Notes", {
            'classes': ('collapse',),
            'fields': (
                'general_exam_notes',
                'eye_exam_notes',
                'skin_hair_nails_exam_notes',
                'ent_mouth_exam_notes',
                'head_neck_exam_notes',
                'lung_exam_notes',
                'cardiovascular_exam_notes',
                'abdomen_pelvis_exam_notes',
                'urinary_system_exam_notes',
                'musculoskeletal_exam_notes',
                'nervous_system_exam_notes',
                'mental_health_exam_notes',
            )
        }),
        ("5. Lab Tests & Paraclinical Results", {
            'classes': ('collapse',),
            'fields': (
                ('lab_test_file_1', 'lab_test_file_2', 'lab_test_file_3'),
                'spirometry_fvc',
                'spirometry_fev1',
                'spirometry_fev1_fvc_ratio',
                'spirometry_interpretation',
                'ecg_findings',
                'chest_xray_findings',
                'referral_notes',
            )
        }),
        ("AI Generated Advice", {
            'fields': ('llm_advice',)
        }),
        ("Timestamps", {
            'fields': (('created_at', 'updated_at'),)
        }),
    )