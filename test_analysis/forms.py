# test_analysis/forms.py

from django import forms
from django.forms import modelformset_factory
from .models import HealthProfile, PreviousJob, Referral


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