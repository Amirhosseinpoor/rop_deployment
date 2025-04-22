from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.uploadedfile import InMemoryUploadedFile
from .utils import get_result
from .models import PredictionLog

# Import other necessary modules if needed

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


    """
    Handle GET and POST requests to display the main page with the form.
    For GET: Display the page.
    For POST: Process the uploaded image and show the result.
    """
    result = None
    error = None

    if request.method == "POST":
        # Check if a file is uploaded
        if "file" in request.FILES:
            uploaded_file = request.FILES["file"]  # Retrieve the uploaded file


            try:
                # Call the utils.get_result method to process the file
                result = get_result(image_file=uploaded_file,request=request)

            except Exception as ex:
                error = str(ex)  # Catch and store any error that occurs during prediction
                print(f"Error during prediction: {error}")  # Debugging print

    # Render the template with the result (or error, if any)
    return render(request, "index.html", {"result": result, "error": error})


@csrf_exempt
def predict(request):
    """
    Handle API POST requests to return JSON responses for predictions.
    """
    if request.method == "POST" and "file" in request.FILES:
        uploaded_file = request.FILES["file"]

        try:
            # Call the utils.get_result function to process the uploaded image
            result = get_result(image_file=uploaded_file, is_api=True)

            return JsonResponse(result)  # Return the prediction as a JSON response
        except Exception as ex:
            print(f"Error: {ex}")  # Debugging print
            return JsonResponse({"error": str(ex)}, status=400)

    # If no file is uploaded, return a bad request response
    print("No file uploaded or invalid request.")  # Debugging print
    return JsonResponse({"error": "No file uploaded or invalid request."}, status=400)

