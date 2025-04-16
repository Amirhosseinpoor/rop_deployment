from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from .forms import CustomUserCreationForm
from single_rop.models import PredictionLog as SinglePrediction
from double_rop.models import PredictionResult as DoublePrediction
from django.contrib.auth.decorators import login_required

from django.contrib.auth import authenticate, login

def custom_login(request):
    if request.user.is_authenticated:
        return redirect('dilemma')

    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)  # مهم!
            return redirect('dilemma')
        else:
            return render(request, 'usac/login.html', {'error': 'Invalid credentials'})

    return render(request, 'usac/login.html')


def signup_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = CustomUserCreationForm()
    return render(request, 'usac/signup.html', {'form': form})

@login_required(login_url='')
def dilemma_view(request):
    return render(request, 'usac/dilemma.html')

@login_required(login_url='')
def history_view(request):
    single = SinglePrediction.objects.filter(user=request.user)
    double = DoublePrediction.objects.filter(user=request.user)
    return render(request, 'usac/history.html', {'single_results': single, 'double_results': double})

import csv
from django.http import HttpResponse
from single_rop.models import PredictionLog
from double_rop.models import PredictionResult# مدل‌هاتو وارد کن

from django.contrib.auth.decorators import login_required


@login_required(login_url='')
def export_history_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="prediction_history.csv"'
    writer = csv.writer(response)

    # ------------------ SINGLE ROP SECTION ------------------
    writer.writerow(['==== ROP RECORDS ===='])
    writer = csv.writer(response)
    writer.writerow([
        'Type',
        'Username',
        'Email',
        'File Name',
        'Predicted Class',
        'Probability',
        'Corrected Class',
        'Review Comment',
        'Execution Time',
        'Timestamp',
        'Image URL'
    ])


    # ---- Single ROP ----
    single_logs = PredictionLog.objects.filter(user=request.user)
    for s in single_logs:
        writer.writerow([
            'ROP',
            s.user.username,
            s.user.email,
            s.file_name,
            s.predicted_class,
            f"{s.probability:.4f}",
            s.corrected_class or "",
            s.review_comment or "",
            s.execution_time,
            s.timestamp,
            request.build_absolute_uri(s.image_url) if s.image_url else ""
        ])
    writer.writerow([])
    # ---- Double ROP ----
    writer.writerow(['==== KC RECORDS ===='])
    writer.writerow([
        'Type',
        'Username',
        'Email',
        'Predicted Labels',
        'Probabilities',
        'Corrected Labels',
        'Review Comment',
        'Inference Time',
        'Timestamp',
        'Image URLs'
    ])
    double_logs = PredictionResult.objects.filter(user=request.user)
    for d in double_logs:
        image_urls = []
        if d.left_image:
            image_urls.append(request.build_absolute_uri(d.left_image.url))
        if d.right_image:
            image_urls.append(request.build_absolute_uri(d.right_image.url))

        writer.writerow([
            'KC',
            d.user.username,
            d.user.email,
            f"L: {d.left_label}, R: {d.right_label}, Z: {d.z_class_label}",
            f"L: {d.left_probability:.4f}, R: {d.right_probability:.4f}, Z: {d.z_class_probability:.4f}",
            f"Corrected → L: {d.corrected_left_label}, R: {d.corrected_right_label}, Z: {d.corrected_z_label}",
            d.review_comment or "",
            f"{d.inference_time:.4f}",
            d.created_at,
            " | ".join(image_urls)
        ])

    return response