from django import forms
from .models import HealthProfile


class HealthProfileForm(forms.ModelForm):
    class Meta:
        model = HealthProfile
        # We want all fields from the model except for the ones
        # that are set automatically by the system.
        exclude = ['user', 'llm_advice']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # You can add custom styling or attributes to your form fields here.
        for field_name, field in self.fields.items():

            # --- THE FIX IS HERE ---
            # Instead of checking for a potentially missing attribute,
            # we check the actual type of the widget. This is much safer.
            if isinstance(field.widget, forms.CheckboxInput):
                css_class = 'form-checkbox'
            else:
                # This will now correctly apply to TextInput, Textarea, Select, FileInput, etc.
                css_class = 'form-input'

            field.widget.attrs['class'] = css_class