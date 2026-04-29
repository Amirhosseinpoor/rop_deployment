# webapp/app/views.py
import json
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .utils import get_result
from .models import PredictionLog


# webapp/app/views.py
import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

def home(request):
    result = None
    results = None
    error = None
    result_json = None

    try:
        if request.method == "POST" and request.POST.get("feedback_mode"):
            # keep your feedback handling here if you have it
            pass

        if request.method == "POST":
            # IMPORTANT: inspect exactly what arrived
            files_debug = [f.name for f in request.FILES.getlist("files")]
            single_debug = request.FILES.get("file").name if "file" in request.FILES else None
            print("DEBUG request.FILES.getlist('files') ->", files_debug)
            print("DEBUG request.FILES['file'] ->", single_debug)

            file_list = []
            if request.FILES.getlist("files"):
                file_list = request.FILES.getlist("files")
            elif "file" in request.FILES:
                file_list = [request.FILES["file"]]

            if not file_list:
                error = "No files received. Make sure the input’s name is 'files' and the form has enctype='multipart/form-data'."

            else:
                from .utils import get_results_for_images
                result, results = get_results_for_images(file_list, request=request)

                # keep a safe json for the chat (no base64)
                safe_result = {k: v for k, v in result.items() if k not in ("image_data", "original_image_data")}
                result_json = json.dumps(safe_result)

    except Exception as ex:
        # Surface any pipeline errors to the page
        import traceback
        traceback.print_exc()
        error = str(ex)

    return render(request, "index2.html", {
        "result": result,
        "results": results,
        "error": error,
        "result_json": result_json
    })

@csrf_exempt
def predict(request):
    """
    Anonymous JSON API: now supports multiple files.
    Returns:
      {
        "aggregated": {...},      # worst-case / final labels
        "per_image": [ {...}, ... ]
      }
    """
    if request.method == "POST":
        files = request.FILES.getlist("files") or ([request.FILES["file"]] if "file" in request.FILES else [])
        if files:
            try:
                from .utils import get_results_for_images
                aggregated, per_image = get_results_for_images(files, request=request)
                return JsonResponse({"aggregated": aggregated, "per_image": per_image})
            except Exception as ex:
                print(f"Error: {ex}")
                return JsonResponse({"error": str(ex)}, status=400)

    print("No file(s) uploaded or invalid request.")
    return JsonResponse({"error": "No file(s) uploaded or invalid request."}, status=400)
