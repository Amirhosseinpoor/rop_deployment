from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.uploadedfile import InMemoryUploadedFile
from .utils import get_result


# Import other necessary modules if needed

def home(request):
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
                result = get_result(image_file=uploaded_file)

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