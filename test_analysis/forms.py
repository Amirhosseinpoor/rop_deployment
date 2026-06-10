# test_analysis/forms.py

from django import forms
from django.forms import modelformset_factory
from .models import HealthProfile, PreviousJob, Referral

from django.forms.widgets import TextInput
from django.utils.safestring import mark_safe

from test_analysis.constants import PROVINCE_CHOICES, NEIGHBORHOOD_CHOICES, INSURANCE_CHOICES
class SearchableSelect(TextInput):
    """
    Renders a text input with an associated <datalist> element.
    Pass `choices` as a list of (value, label) tuples.
    The 'list' attribute connects to the datalist's id.
    """
    template_name = 'django/forms/widgets/text.html'  # keep the input rendering

    def __init__(self, choices=None, attrs=None):
        super().__init__(attrs)
        self.choices = choices or []

    def render(self, name, value, attrs=None, renderer=None):
        # Build the input
        input_html = super().render(name, value, attrs, renderer)
        # Build the datalist
        id_ = attrs.get('id') if attrs else None
        list_id = f'{id_}_list' if id_ else f'{name}_list'
        options = ''.join(
            f'<option value="{option_value}">{label}</option>'
            for option_value, label in self.choices
        )
        datalist_html = f'<datalist id="{list_id}">{options}</datalist>'
        # Add the 'list' attribute to the input (it must point to the datalist id)
        input_html = input_html.replace('<input', f'<input list="{list_id}"')
        return mark_safe(input_html + datalist_html)


def apply_form_widget_classes(form):
    """
    A helper function to iterate over form fields and apply consistent CSS classes.
    """
    for field_name, field in form.fields.items():
        css_class = 'form-input'  # Default class
        if isinstance(field.widget, forms.CheckboxInput):
            css_class = 'form-checkbox'
        elif isinstance(field.widget, forms.FileInput):
            # Special, more detailed classes for file inputs
            css_class = 'form-input file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-violet-50 file:text-violet-700 hover:file:bg-violet-100'

        # Add the class to the widget's attributes
        field.widget.attrs['class'] = css_class


class HealthProfileForm(forms.ModelForm):
    """
    Form for creating and updating the main Health Profile.
    """

    class Meta:
        model = HealthProfile
        # Exclude fields that should not be edited by the user directly.
        exclude = ['user', 'llm_advice']

        # Use HTML5 date pickers for date fields for a better user experience.
        widgets = {
            'date_of_birth': forms.DateInput(attrs={'type': 'date'}),
            'current_job_start_date': forms.DateInput(attrs={'type': 'date'}),
            'exam_date': forms.DateInput(attrs={'type': 'date'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Apply CSS classes to all fields
        apply_form_widget_classes(self)


class PreviousJobForm(forms.ModelForm):
    """
    Form for a single previous job entry. Used within the formset.
    """

    class Meta:
        model = PreviousJob
        fields = ['title', 'duties', 'start_date', 'end_date', 'reason_for_leaving']
        widgets = {
            'start_date': forms.DateInput(attrs={'type': 'date'}),
            'end_date': forms.DateInput(attrs={'type': 'date'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Apply CSS classes to all fields
        apply_form_widget_classes(self)


class ReferralForm(forms.ModelForm):
    """
    Form for a single specialist referral entry. Used within the formset.
    """

    class Meta:
        model = Referral
        fields = ['date', 'reason', 'specialty', 'result']
        widgets = {
            'date': forms.DateInput(attrs={'type': 'date'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Apply CSS classes to all fields
        apply_form_widget_classes(self)


# ------------------------------------------------
# NEW FORMS FOR ROLE-BASED PROFILES
# ------------------------------------------------

class EmployeeProfileForm(forms.ModelForm):
    """
    Fields that the employee fills (personal, job, hazards, medical history,
    vital signs, lab/paraclinical tests).
    """

    class Meta:
        model = HealthProfile
        fields = [
            # Personal
            'father_name', 'national_id', 'date_of_birth', 'age', 'gender',
            'marital_status', 'children_count',
            'military_service_status', 'military_service_rank', 'medical_exemption_reason',
            'work_address', 'work_phone', 'living_province',  # new
            'neighborhood',  # new
            'insurance',
            # Current job
            'current_job_title', 'current_job_duties', 'current_job_start_date',
            # Hazards – physical
            'hazard_physical_noise', 'hazard_physical_vibration',
            'hazard_physical_non_ionizing_radiation', 'hazard_physical_ionizing_radiation',
            'hazard_physical_heat_stress', 'hazard_physical_other',
            # Hazards – chemical
            'hazard_chemical_dust', 'hazard_chemical_metal_fumes', 'hazard_chemical_solvents',
            'hazard_chemical_pesticides', 'hazard_chemical_acids_bases', 'hazard_chemical_gases',
            'hazard_chemical_other',
            # Hazards – biological
            'hazard_biological_bites', 'hazard_biological_bacteria', 'hazard_biological_virus',
            'hazard_biological_parasite', 'hazard_biological_other',
            # Hazards – ergonomic
            'hazard_ergonomic_prolonged_sitting_standing', 'hazard_ergonomic_repetitive_work',
            'hazard_ergonomic_heavy_lifting', 'hazard_ergonomic_poor_posture',
            'hazard_ergonomic_other',
            # Hazards – psychological
            'hazard_psychological_shift_work', 'hazard_psychological_stressors',
            'hazard_psychological_other',
            # Medical history
            'has_disease_history', 'disease_history_details',
            'does_symptoms_change_at_work', 'do_colleagues_have_similar_symptoms',
            'does_symptoms_change_on_holidays',
            'has_allergies', 'allergy_details',
            'has_hospitalization_history', 'hospitalization_reason',
            'has_surgery_history', 'surgery_details',
            'has_family_cancer_or_chronic_disease', 'family_disease_details',
            'is_on_medication', 'medication_details',
            'is_currently_smoking', 'smoking_details', 'has_past_smoking_history',
            'hobbies',
            'has_occupational_accident_history', 'accident_details',
            'has_absence_over_3_days', 'lives_near_industrial_center',
            'has_medical_commission_referral',
            'cigs_per_day', 'on_bp_meds', 'has_diabetes',
            # Vital signs (employee provides)
            'exam_date', 'exam_weight', 'exam_height', 'bmi',
            'exam_systolic_bp', 'exam_diastolic_bp', 'exam_pulse_rate',
            # Lab & paraclinical (employee provides)
            'lab_total_cholesterol', 'lab_glucose',
            'spirometry_fvc', 'spirometry_fev1', 'spirometry_fev1_fvc_ratio',
            'spirometry_fef_25_75', 'spirometry_pef',
            'spirometry_interpretation',
            'ecg_findings',
            'chest_xray_findings',
            'other_paraclinical_notes',
            'lab_test_files',
        ]
        widgets = {
            'date_of_birth': forms.DateInput(attrs={'type': 'date'}),
            'current_job_start_date': forms.DateInput(attrs={'type': 'date'}),
            'exam_date': forms.DateInput(attrs={'type': 'date'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from test_analysis.constants import PROVINCE_CHOICES, NEIGHBORHOOD_CHOICES

        # Set searchable widgets
        self.fields['living_province'].widget = SearchableSelect(
            choices=PROVINCE_CHOICES,
            attrs={'class': 'form-input', 'placeholder': 'جستجوی استان...'}
        )
        self.fields['neighborhood'].widget = SearchableSelect(
            choices=NEIGHBORHOOD_CHOICES,
            attrs={'class': 'form-input', 'placeholder': 'جستجوی محله...'}
        )
        # Insurance: plain text (will be changed later)
        self.fields['insurance'].widget = SearchableSelect(
            choices=INSURANCE_CHOICES,
            attrs={'class': 'form-input', 'placeholder': 'جستجوی بیمه...'}
        )
        apply_form_widget_classes(self)


class DoctorNotesForm(forms.ModelForm):
    OPINION_CHOICES = [
        ('fit', 'بلامانع برای کار'),
        ('conditional', 'مشروط'),
        ('unfit', 'عدم صلاحیت'),
    ]
    opinion_choice = forms.ChoiceField(
        choices=OPINION_CHOICES,
        widget=forms.RadioSelect,
        required=False,
        label='نظریه نهایی',
    )

    class Meta:
        model = HealthProfile
        fields = [
            'general_exam_notes', 'eye_exam_notes', 'skin_hair_nails_exam_notes',
            'ent_mouth_exam_notes', 'head_neck_exam_notes', 'lung_exam_notes',
            'cardiovascular_exam_notes', 'abdomen_pelvis_exam_notes',
            'urinary_system_exam_notes', 'musculoskeletal_exam_notes',
            'nervous_system_exam_notes', 'mental_health_exam_notes',
            'opinion_fit_conditions_details', 'opinion_unfit_reason',
            'medical_recommendations',
        ]
        # Remove the three boolean fields from the form – we handle them via opinion_choice

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-select the correct radio based on existing data
        instance = kwargs.get('instance')
        if instance:
            if instance.opinion_fit:
                self.fields['opinion_choice'].initial = 'fit'
            elif instance.opinion_fit_with_conditions:
                self.fields['opinion_choice'].initial = 'conditional'
            elif instance.opinion_unfit:
                self.fields['opinion_choice'].initial = 'unfit'

        # Apply widget classes (unchanged)
        apply_form_widget_classes(self)

    def clean(self):
        cleaned_data = super().clean()
        opinion = cleaned_data.get('opinion_choice')
        if not opinion:
            raise forms.ValidationError('لطفاً یکی از نظریه‌های نهایی را انتخاب کنید.')
        return cleaned_data

    def save(self, commit=True):
        profile = super().save(commit=False)
        opinion = self.cleaned_data.get('opinion_choice')
        # Reset all three booleans and set the selected one
        profile.opinion_fit = (opinion == 'fit')
        profile.opinion_fit_with_conditions = (opinion == 'conditional')
        profile.opinion_unfit = (opinion == 'unfit')
        if commit:
            profile.save()
        return profile


# --- Formsets ---
# Create formsets to handle multiple "PreviousJob" and "Referral" entries on the same page.

PreviousJobFormSet = modelformset_factory(
    PreviousJob,
    form=PreviousJobForm,
    extra=1,  # Show one extra empty form by default.
    can_delete=True  # Allow users to delete existing entries.
)

ReferralFormSet = modelformset_factory(
    Referral,
    form=ReferralForm,
    extra=1,
    can_delete=True
)
