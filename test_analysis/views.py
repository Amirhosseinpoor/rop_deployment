from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.conf import settings
from .forms import HealthProfileForm
from .models import HealthProfile
from openai import OpenAI
import markdown2  # Ensure this is imported


# It's highly recommended to load these from settings for security
# Make sure they are defined in your settings.py and .env file
# METIS_API_KEY = settings.OPENAI_API_KEY
# BASE_URL = settings.OPENAI_BASE_URL
METIS_API_KEY = 'tpsg-ba50J5QcVL0x9lPjjSj616bEQrCbrxC'
BASE_URL = "https://api.metisai.ir/openai/v1"
def format_profile_for_llm(profile):
    """
    Formats the user's profile data into a single string for the LLM.
    """
    prompt_data = f"User Profile for {profile.user.username}:\n\n"

    prompt_data += f"== Personal & Occupational Info ==\n"
    prompt_data += f"Marital Status: {profile.marital_status}\n"
    prompt_data += f"Children Count: {profile.children_count}\n"
    prompt_data += f"Current Job: {profile.current_job_title}\n"
    prompt_data += f"Hazards: Physical({profile.physical_hazards}), Chemical({profile.chemical_hazards}), Biological({profile.biological_hazards}), Ergonomic({profile.ergonomic_hazards}), Psychological({profile.psychological_hazards})\n\n"

    prompt_data += f"== Medical History ==\n"
    prompt_data += f"Disease History: {'Yes' if profile.has_disease_history else 'No'}. Details: {profile.disease_history_details}\n"
    prompt_data += f"Allergies: {'Yes' if profile.has_allergies else 'No'}. Details: {profile.allergy_details}\n"
    prompt_data += f"Currently Smoking: {'Yes' if profile.is_currently_smoking else 'No'}. Details: {profile.smoking_details}\n\n"

    prompt_data += f"== Examination Notes ==\n"
    prompt_data += f"General Notes: {profile.general_exam_notes}\n"
    prompt_data += f"Lungs Notes: {profile.lung_exam_notes}\n"
    prompt_data += f"Cardiovascular Notes: {profile.cardiovascular_exam_notes}\n"
    prompt_data += f"Nervous System Notes: {profile.nervous_system_exam_notes}\n"
    prompt_data += f"Mental Health Notes: {profile.mental_health_exam_notes}\n\n"

    prompt_data += f"== Paraclinical Findings ==\n"
    prompt_data += f"Spirometry: FVC={profile.spirometry_fvc}, FEV1={profile.spirometry_fev1}, Ratio={profile.spirometry_fev1_fvc_ratio}, Interp={profile.spirometry_interpretation}\n"
    prompt_data += f"ECG Findings: {profile.ecg_findings}\n"
    prompt_data += f"Chest X-Ray Findings: {profile.chest_xray_findings}\n\n"

    return prompt_data


def get_llm_advice(profile_text):
    """
    Sends the formatted profile to the GPT-4o-mini model via MetisAI
    and returns the generated health advice.
    """
    system_prompt = """
    You are a helpful AI assistant specialized in occupational health and general wellness.
    Based on the user's health profile provided, give personalized, actionable, and empathetic advice.
    Structure your advice with clear headings in Markdown. Focus on potential risks related to their job and lifestyle,
    and suggest improvements. Start by summarizing the key points from their profile.
    """

    try:
        client = OpenAI(
            api_key=METIS_API_KEY,
            base_url=BASE_URL
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": profile_text}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "There was an error generating advice. Please try again later."


@login_required
def create_or_update_health_profile(request):
    """
    Handles both displaying the form and processing the submission.
    """
    try:
        instance = HealthProfile.objects.get(user=request.user)
    except HealthProfile.DoesNotExist:
        instance = None

    if request.method == 'POST':
        form = HealthProfileForm(request.POST, request.FILES, instance=instance)
        if form.is_valid():
            profile = form.save(commit=False)
            profile.user = request.user

            profile_text_for_llm = format_profile_for_llm(profile)
            advice = get_llm_advice(profile_text_for_llm)
            profile.llm_advice = advice

            profile.save()
            return redirect('profile_detail')
    else:
        form = HealthProfileForm(instance=instance)

    return render(request, 'test_analysis/profile_form.html', {'form': form})


@login_required
def profile_detail_view(request):
    """
    Displays the user's profile information and the generated AI advice.
    It now converts the Markdown advice to HTML before rendering.
    """
    profile = get_object_or_404(HealthProfile, user=request.user)

    # Convert Markdown advice to HTML. If advice is empty, use a default message.
    markdown_text = profile.llm_advice or "### No Advice Generated\n\nThere was no advice generated for this profile yet. Please try submitting the form again."
    html_advice = markdown2.markdown(markdown_text)

    context = {
        'profile': profile,
        'html_advice': html_advice, # Pass the generated HTML to the template
    }
    return render(request, 'test_analysis/profile_detail.html', context)

    # Render the template with the correct context
    # return render(request, 'test_analysis/profile_detail.html', context)