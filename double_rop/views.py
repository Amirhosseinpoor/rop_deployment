from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .utils import get_prediction
import base64
import time


def home(request):
    result = None
    error = None

    if request.method == "POST":
        left_file = request.FILES.get("left_file")
        right_file = request.FILES.get("right_file")

        if left_file and right_file:
            try:
                # Read image bytes and create base64 previews
                left_bytes = left_file.read()
                right_bytes = right_file.read()

                # Encode images for display
                def encode_image(bytes_data):
                    return f"data:image/png;base64,{base64.b64encode(bytes_data).decode()}"

                # Get predictions with timing
                start_time = time.time()
                raw_result = get_prediction(left_bytes, right_bytes)
                inference_time = f"{(time.time() - start_time):.2f}s"

                # Structure result for template - FIXED KEYS HERE
                result = {
                    "predictions": {
                        "left_eye": {
                            "label": raw_result["left_eye"]["label"],
                            "probability": raw_result["left_eye"]["probability"]
                        },
                        "right_eye": {
                            "label": raw_result["right_eye"]["label"],
                            "probability": raw_result["right_eye"]["probability"]
                        },
                        "z_class": {
                            "label": raw_result["z_class"]["label"],
                            "probability": raw_result["z_class"]["probability"]
                        }
                    },
                    "image_data": {
                        "left": encode_image(left_bytes),
                        "right": encode_image(right_bytes)
                    },
                    "inference_time": inference_time
                }

            except Exception as ex:
                error = f"Prediction error: {str(ex)}"
        else:
            error = "Both left and right eye images are required"

    return render(request, "index_double.html", {
        "result": result,
        "error": error
    })


@csrf_exempt
def predict(request):
    if request.method == "POST":
        left_file = request.FILES.get("left_file")
        right_file = request.FILES.get("right_file")

        if not (left_file and right_file):
            return JsonResponse({"error": "Both images required"}, status=400)

        try:
            start_time = time.time()
            result = get_prediction(left_file.read(), right_file.read())
            result["inference_time"] = time.time() - start_time
            return JsonResponse(result)
        except Exception as ex:
            return JsonResponse({"error": str(ex)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)


from django.shortcuts import render

# Create your views here.