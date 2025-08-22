# test_analysis/views.py

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.forms import modelformset_factory
from .forms import HealthProfileForm, PreviousJobFormSet, ReferralFormSet
from .models import HealthProfile, PreviousJob, Referral
from openai import OpenAI
import markdown2
from . import ai_pipeline


def format_profile_for_llm(profile):

    prompt_data = f"Analyze the following occupational health profile for {profile.user.username} and provide personalized wellness and safety advice.\n\n"
    # --- Section 1 & 2: Personal and Occupational Info ---
    prompt_data += "== Personal & Current Occupational Information ==\n"
    prompt_data += f"- Age: {profile.age}\n" if profile.age else ""
    prompt_data += f"- Date of Birth: {profile.date_of_birth}\n" if profile.date_of_birth else ""
    prompt_data += f"- Gender: {'1' if profile.gender == 'Male' else '0'}\n"  # Corrected for model
    prompt_data += f"- Marital Status: {profile.marital_status}\n" if profile.marital_status else ""
    prompt_data += f"- Children: {profile.children_count}\n"
    prompt_data += f"- Current Job: {profile.current_job_title}\n" if profile.current_job_title else ""
    prompt_data += f"- Current Job Duties: {profile.current_job_duties}\n\n" if profile.current_job_duties else "\n"
    if profile.province_of_residence:
        prompt_data += f"- Province of Residence: {profile.province_of_residence}\n"
    # (The rest of the function remains the same as your original)
    # ... (Keep the rest of the function as it was)
    # --- Section 4: Medical History ---
    prompt_data += "== Medical & Lifestyle History ==\n"
    if profile.has_disease_history:
        prompt_data += f"- History of significant disease: Yes. Details: {profile.disease_history_details}\n"
    prompt_data += f"- History of Diabetes: {'1' if profile.has_diabetes else '0'}\n"
    if profile.has_allergies:
        prompt_data += f"- History of allergies: Yes. Details: {profile.allergy_details}\n"
    if profile.has_surgery_history:
        prompt_data += f"- History of surgery: Yes. Details: {profile.surgery_details}\n"
    if profile.has_hospitalization_history:
        prompt_data += f"- History of hospitalization: Yes. Reason: {profile.hospitalization_reason}\n"
    if profile.is_on_medication:
        prompt_data += f"- Currently on medication: Yes. Details: {profile.medication_details}\n"
    prompt_data += f"- On blood pressure medication: {'1' if profile.on_bp_meds else '0'}\n"
    prompt_data += f"- Currently smokes: {'1' if profile.is_currently_smoking else '0'}\n"
    if profile.is_currently_smoking and profile.smoking_details:
        prompt_data += f"- Cigarettes per day: {profile.smoking_details}\n"  # Assuming smoking_details is cigsPerDay
    else:
        prompt_data += f"- Cigarettes per day: 0\n"

    # --- Section 5 & 6 & 7: Examination and Paraclinical ---
    prompt_data += "== Examination & Paraclinical Findings ==\n"
    if profile.exam_systolic_bp and profile.exam_diastolic_bp:
        prompt_data += f"- Systolic BP: {profile.exam_systolic_bp}\n"
        prompt_data += f"- Diastolic BP: {profile.exam_diastolic_bp}\n"
    prompt_data += f"- Heart Rate: {profile.exam_pulse_rate} bpm\n" if profile.exam_pulse_rate else ""
    prompt_data += f"- BMI: {profile.bmi}\n" if profile.bmi else ""
    prompt_data += f"- Total Cholesterol: {profile.lab_total_cholesterol} mg/dL\n" if profile.lab_total_cholesterol else ""
    prompt_data += f"- Glucose: {profile.lab_glucose} mg/dL\n" if profile.lab_glucose else ""

    # ... (any other fields you need for the prompt) ...
    return prompt_data


def get_llm_advice(profile_text, selected_model):

    try:
        final_report = ai_pipeline.run_health_analysis_pipeline(profile_text, selected_model)
        return final_report
    except Exception as e:
        print(f"🔥 CRITICAL ERROR calling the AI pipeline: {e}")
        return "متاسفانه در حال حاضر به دلیل یک خطای داخلی، امکان تولید گزارش وجود ندارد. لطفا بعداً دوباره تلاش کنید."


@login_required
def create_or_update_health_profile(request):
    """
    Handles both creating/updating the profile and its related jobs and referrals.
    """
    try:
        instance = HealthProfile.objects.get(user=request.user)
    except HealthProfile.DoesNotExist:
        instance = None

    if request.method == 'POST':
        form = HealthProfileForm(request.POST, request.FILES, instance=instance)
        job_formset = PreviousJobFormSet(request.POST, prefix='jobs', queryset=PreviousJob.objects.filter(
            profile=instance) if instance else PreviousJob.objects.none())
        referral_formset = ReferralFormSet(request.POST, prefix='referrals', queryset=Referral.objects.filter(
            profile=instance) if instance else Referral.objects.none())

        if form.is_valid() and job_formset.is_valid() and referral_formset.is_valid():
            profile = form.save(commit=False)
            profile.user = request.user
            profile.save()

            jobs = job_formset.save(commit=False)
            for job in jobs:
                job.profile = profile
                job.save()
            job_formset.save_m2m()

            referrals = referral_formset.save(commit=False)
            for referral in referrals:
                referral.profile = profile
                referral.save()
            referral_formset.save_m2m()

            # Generate the text summary from the saved profile
            profile_text_for_llm = format_profile_for_llm(profile)
            selected_model = request.POST.get('selected_model', 'cloud_gpt')
            profile.model_used_for_advice = selected_model
            # Call our NEW, powerful pipeline
            advice = get_llm_advice(profile_text_for_llm, selected_model)

            # Save the final report to the profile
            profile.llm_advice = advice
            profile.save()

            return redirect('profile_detail')
        else:
            # Your debugging code for form errors
            print("\n--- FORM VALIDATION FAILED ---")
            if form.errors: print("Main form errors:", form.errors)
            if job_formset.errors: print("Job Formset errors:", job_formset.errors)
            if referral_formset.errors: print("Referral Formset errors:", referral_formset.errors)
            print("------------------------------\n")
    else:
        form = HealthProfileForm(instance=instance)
        job_formset = PreviousJobFormSet(prefix='jobs', queryset=PreviousJob.objects.filter(
            profile=instance) if instance else PreviousJob.objects.none())
        referral_formset = ReferralFormSet(prefix='referrals', queryset=Referral.objects.filter(
            profile=instance) if instance else Referral.objects.none())

    context = {
        'form': form,
        'job_formset': job_formset,
        'referral_formset': referral_formset
    }
    return render(request, 'test_analysis/profile_form.html', context)


@login_required
def profile_detail_view(request):
    """
    Displays the complete profile, including the final report.
    """
    profile = get_object_or_404(HealthProfile, user=request.user)
    # The final report from the pipeline might already contain markdown
    html_advice = markdown2.markdown(profile.llm_advice or "No advice generated.")

    context = {
        'profile': profile,
        'html_advice': html_advice,
    }
    return render(request, 'test_analysis/profile_detail.html', context)