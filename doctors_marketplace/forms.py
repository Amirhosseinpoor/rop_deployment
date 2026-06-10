# doctors_marketplace/forms.py
from django import forms
from .models import Doctor, DoctorKnowledge

class DoctorForm(forms.ModelForm):
    class Meta:
        model = Doctor
        fields = [
            "name","name_fa","slug",
            "specialization","specialization_fa",
            "persona","headline","headline_fa",
            "bio","bio_fa","tags_fa",
            "system_prompt","avatar","is_active",
        ]
        widgets = {
            "system_prompt": forms.Textarea(attrs={"rows": 8, "class": "w-full border rounded p-2"}),
            "bio": forms.Textarea(attrs={"rows": 5, "class": "w-full border rounded p-2"}),
            "bio_fa": forms.Textarea(attrs={"rows": 5, "class": "w-full border rounded p-2"}),
            "headline": forms.TextInput(attrs={"class": "w-full border rounded p-2"}),
            "headline_fa": forms.TextInput(attrs={"class": "w-full border rounded p-2"}),
            "tags_fa": forms.TextInput(attrs={"class": "w-full border rounded p-2","placeholder":"tag1,tag2,tag3"}),
            "name": forms.TextInput(attrs={"class":"w-full border rounded p-2"}),
            "name_fa": forms.TextInput(attrs={"class":"w-full border rounded p-2"}),
            "slug": forms.TextInput(attrs={"class":"w-full border rounded p-2","placeholder":"auto from name if empty"}),
        }

class KnowledgeUploadForm(forms.ModelForm):
    class Meta:
        model = DoctorKnowledge
        fields = ["title", "file"]
        widgets = {
            "title": forms.TextInput(attrs={"class":"w-full border rounded p-2"}),
        }
