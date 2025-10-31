# webapp/app/views.py
import json
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .utils import get_result
from .models import PredictionLog


def home(request):
    if request.method == "POST" and request.POST.get("feedback_mode"):
        corrected_class = request.POST.get("corrected_class")
        review_comment = request.POST.get("review_comment")
        corrected_stage = request.POST.get("stage_corrected_class")

        latest_prediction = PredictionLog.objects.filter(user=request.user).latest("timestamp")
        latest_prediction.corrected_class = corrected_class
        latest_prediction.review_comment = review_comment
        latest_prediction.stage_corrected_class = corrected_stage
        classification_status = 1

        if (
            corrected_class and corrected_class != latest_prediction.predicted_class
        ) or (
            corrected_stage and corrected_stage != latest_prediction.stage_class
        ):
            classification_status = -1

        latest_prediction.classification_status = classification_status
        latest_prediction.save()

    result = None
    error = None
    result_json = None

    if request.method == "POST":
        if "file" in request.FILES:
            uploaded_file = request.FILES["file"]
            try:
                result = get_result(image_file=uploaded_file, request=request)
                if result:
                    safe_result = {k: v for k, v in result.items() if k not in ("image_data",)}
                    result_json = json.dumps(safe_result)

            except Exception as ex:
                error = str(ex)
                print(f"Error during prediction: {error}")

    return render(request, "index.html", {"result": result, "error": error, "result_json": result_json})


@csrf_exempt
def predict(request):
    """
    Anonymous API endpoint: returns a JSON with diagnostic_context (dict + text)
    that your RAG service can pass directly as `prediction_context`.
    """
    if request.method == "POST" and "file" in request.FILES:
        uploaded_file = request.FILES["file"]
        try:
            result = get_result(image_file=uploaded_file, is_api=True, request=request)
            return JsonResponse(result)
        except Exception as ex:
            print(f"Error: {ex}")
            return JsonResponse({"error": str(ex)}, status=400)

    print("No file uploaded or invalid request.")
    return JsonResponse({"error": "No file uploaded or invalid request."}, status=400)
