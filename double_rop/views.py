from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .utils import get_prediction
import base64
import time
from .models import PredictionResult


def home(request):
    result = None
    error = None

    if request.method == "POST":

        if "feedback_mode" in request.POST and request.user.is_authenticated:
            try:
                latest_result = PredictionResult.objects.filter(user=request.user).latest("created_at")
                latest_result.corrected_left_label = request.POST.get("corrected_left_label")
                latest_result.corrected_right_label = request.POST.get("corrected_right_label")
                latest_result.corrected_z_label = request.POST.get("corrected_z_label")
                latest_result.review_comment = request.POST.get("review_comment")
                classification_status = 1

                if (
                        latest_result.corrected_left_label and latest_result.corrected_left_label != latest_result.left_label) or \
                        (
                                latest_result.corrected_right_label and latest_result.corrected_right_label != latest_result.right_label) or \
                        (
                                latest_result.corrected_z_label and latest_result.corrected_z_label != latest_result.z_class_label):
                    classification_status = -1

                latest_result.classification_status = classification_status
                latest_result.save()

                result = {
                    "predictions": {
                        "left_eye": {
                            "label": latest_result.left_label,
                            "probability": latest_result.left_probability
                        },
                        "right_eye": {
                            "label": latest_result.right_label,
                            "probability": latest_result.right_probability
                        },
                        "z_class": {
                            "label": latest_result.z_class_label,
                            "probability": latest_result.z_class_probability
                        }
                    },
                    "image_data": {
                        "left": latest_result.left_image.url,
                        "right": latest_result.right_image.url
                    },
                    "inference_time": f"{latest_result.inference_time:.2f}s",
                    "left_image_url": latest_result.left_image_url,
                    "right_image_url": latest_result.right_image_url
                }

                return render(request, "index_double.html", {
                    "result": None,
                    "feedback_saved": True
                })

            except Exception as ex:
                error = f"Feedback error: {str(ex)}"
                return render(request, "index_double.html", {"error": error})

        # -------------------- فاز 1: پیش‌بینی --------------------
        left_file = request.FILES.get("left_file")
        right_file = request.FILES.get("right_file")

        if left_file and right_file:
            try:
                # Read image bytes and create base64 previews
                left_bytes = left_file.read()
                right_bytes = right_file.read()

                def encode_image(bytes_data):
                    return f"data:image/png;base64,{base64.b64encode(bytes_data).decode()}"

                start_time = time.time()
                raw_result = get_prediction(left_bytes, right_bytes)
                inference_time = f"{(time.time() - start_time):.2f}s"

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

                if request.user.is_authenticated:
                    result_obj = PredictionResult.objects.create(
                        user=request.user,
                        left_label=raw_result["left_eye"]["label"],
                        left_probability=raw_result["left_eye"]["probability"],
                        right_label=raw_result["right_eye"]["label"],
                        right_probability=raw_result["right_eye"]["probability"],
                        z_class_label=raw_result["z_class"]["label"],
                        z_class_probability=raw_result["z_class"]["probability"],
                        inference_time=float(inference_time.replace("s", "")),
                        left_image=left_file,
                        right_image=right_file,
                    )

                    from django.conf import settings
                    result_obj.left_image_url = request.build_absolute_uri(result_obj.left_image.url)
                    result_obj.right_image_url = request.build_absolute_uri(result_obj.right_image.url)
                    result_obj.save()

                    result["left_image_url"] = result_obj.left_image_url
                    result["right_image_url"] = result_obj.right_image_url

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


#hel
