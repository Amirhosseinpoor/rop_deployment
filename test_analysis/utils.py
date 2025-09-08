# test_analysis/utils.py
def format_profile_for_llm(profile):
    prompt_data = (
        f"Analyze the following occupational health profile for {profile.user.username} "
        f"and provide personalized wellness and safety advice.\n\n"
    )
    prompt_data += "== Personal & Current Occupational Information ==\n"
    if profile.age: prompt_data += f"- Age: {profile.age}\n"
    if profile.date_of_birth: prompt_data += f"- Date of Birth: {profile.date_of_birth}\n"
    # جنسیت: مقدار ذخیره‌شده در مدل 'Male' یا 'Female' است
    prompt_data += f"- Gender: {'1' if profile.gender == 'Male' else '0'}\n"
    if profile.marital_status: prompt_data += f"- Marital Status: {profile.marital_status}\n"
    prompt_data += f"- Children: {profile.children_count}\n"
    if profile.current_job_title: prompt_data += f"- Current Job: {profile.current_job_title}\n"
    if profile.current_job_duties: prompt_data += f"- Current Job Duties: {profile.current_job_duties}\n"
    if profile.province_of_residence:
        prompt_data += f"- Province of Residence: {profile.province_of_residence}\n"

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

    # ✅ به‌جای smoking_details از فیلد عددی cigs_per_day استفاده کن
    prompt_data += f"- Cigarettes per day: {profile.cigs_per_day or 0}\n"

    prompt_data += "== Examination & Paraclinical Findings ==\n"
    if profile.exam_systolic_bp and profile.exam_diastolic_bp:
        prompt_data += f"- Systolic BP: {profile.exam_systolic_bp}\n"
        prompt_data += f"- Diastolic BP: {profile.exam_diastolic_bp}\n"
    if profile.exam_pulse_rate: prompt_data += f"- Heart Rate: {profile.exam_pulse_rate} bpm\n"
    if profile.bmi: prompt_data += f"- BMI: {profile.bmi}\n"
    if profile.lab_total_cholesterol:
        prompt_data += f"- Total Cholesterol: {profile.lab_total_cholesterol} mg/dL\n"
    if profile.lab_glucose:
        prompt_data += f"- Glucose: {profile.lab_glucose} mg/dL\n"

    return prompt_data
