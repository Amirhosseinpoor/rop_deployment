# webapp/app/views.py
import json
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse, HttpResponseForbidden
from django.views.decorators.csrf import csrf_exempt

from .utils import get_result
from .models import PredictionLog


# webapp/app/views.py
import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


def _build_rop_review(target):
    """
    Reconstruct an employee's most recent ROP screening batch from the DB so a
    reviewer (doctor/manager) can see the stored result without re-uploading.

    Returns (result, results, result_json). If the employee has no logs it
    returns (None, None, None) so the page falls back to its empty state.
    """
    from datetime import timedelta

    logs = list(PredictionLog.objects.filter(user=target).order_by("-timestamp"))
    if not logs:
        return None, None, None

    newest = logs[0].timestamp
    batch = [lg for lg in logs if (newest - lg.timestamp) <= timedelta(minutes=5)]

    results = []
    for log in batch:
        results.append({
            "image_data": log.segmented_image_url or log.image_url,
            "original_image_data": log.image_url,
            "file_name": log.file_name,
            "inference_time": "",
            "predictions": {"class_name": log.predicted_class,
                            "class_prob": f"{log.probability:.3f}"},
            "stage_prediction": {"stage_name": log.stage_class or "",
                                 "stage_prob": f"{(log.stage_probability or 0):.3f}"},
            "zone_prediction": {"zone_name": log.zone_class or "",
                                "zone_prob": f"{(log.zone_probability or 0):.3f}"},
            "final_decision": log.final_decision or "",
            "corrected_class": log.corrected_class,
            "stage_corrected_class": log.stage_corrected_class,
            "zone_corrected_class": log.zone_corrected_class,
            "review_comment": log.review_comment,
        })

    first = results[0]
    plus = first["predictions"]["class_name"]
    stage = first["stage_prediction"]["stage_name"]
    zone = first["zone_prediction"]["zone_name"]

    # Recompute the final decision + clinical guidance from the stored labels
    # (the same logic a live prediction uses), so the reviewer sees the guidance.
    from .utils import get_guidance, compute_final_decision
    try:
        final_decision = first["final_decision"] or compute_final_decision(
            zone_label=zone, stage_label=stage, plus_label=plus)
    except Exception:
        final_decision = first["final_decision"] or ""
    try:
        g = get_guidance(zone, plus, stage, final_decision)
        guidance = {"title": g.get("title", ""), "text": g.get("text", "")}
    except Exception:
        guidance = {"title": "", "text": ""}

    diag_text = (
        f"Plus disease: {plus}. Stage: {stage}. Zone: {zone}. "
        f"Final decision: {final_decision}."
    )
    result = {
        "image_data": first["image_data"],
        "original_image_data": first["original_image_data"],
        "inference_time": "",
        "file_name": first["file_name"],
        "predictions": first["predictions"],
        "stage_prediction": first["stage_prediction"],
        "zone_prediction": first["zone_prediction"],
        "final_decision": final_decision,
        "guidance": guidance,
        "diagnostic_context_text": diag_text,
        "llm_diagnostic_text": diag_text,
        "worst_index": 0,
        "doctor_assessment": batch[0].doctor_assessment or "",
        "doctor_recommendation": batch[0].doctor_recommendation or "",
    }
    safe_result = {k: v for k, v in result.items()
                   if k not in ("image_data", "original_image_data")}
    result_json = json.dumps(safe_result)
    return result, results, result_json


def home(request):
    result = None
    results = None
    error = None
    result_json = None

    # ---- role-aware context (users may be anonymous on /rop/) ----
    from usac.models import UserProfile
    profile = getattr(request.user, "profile", None) if request.user.is_authenticated else None
    is_reviewer = bool(profile and profile.role in (UserProfile.ROLE_DOCTOR, UserProfile.ROLE_MANAGER))
    chat_mode = "doctor" if is_reviewer else "employee"
    viewed_user = None
    review_mode = False
    locked_employee = False

    # ---- REVIEW MODE: reviewer opens ?employee=<id> to see stored results ----
    employee_id = request.GET.get("employee")
    if request.method == "GET" and employee_id and is_reviewer:
        from django.contrib.auth.models import User
        target = User.objects.filter(id=employee_id).select_related("profile").first()
        if target is None:
            error = "Employee not found."
        else:
            tprof = getattr(target, "profile", None)
            allowed = False
            if tprof is not None:
                if profile.role == UserProfile.ROLE_DOCTOR:
                    allowed = bool(tprof.company_id) and tprof.company_id == profile.company_id
                elif profile.role == UserProfile.ROLE_MANAGER:
                    same_company = bool(tprof.company_id) and tprof.company_id == profile.company_id
                    manages = bool(tprof.company_id) and tprof.company.manager_id == request.user.id
                    allowed = same_company or manages
            if not allowed:
                return HttpResponseForbidden(
                    "You do not have permission to view this employee's results."
                )
            viewed_user = target
            review_mode = True
            result, results, result_json = _build_rop_review(target)
        return render(request, "index2.html", {
            "result": result,
            "results": results,
            "error": error,
            "result_json": result_json,
            "is_reviewer": is_reviewer,
            "chat_mode": chat_mode,
            "viewed_user": viewed_user,
            "review_mode": review_mode,
            "locked_employee": locked_employee,
            "can_edit_notes": bool(is_reviewer and viewed_user is not None),
        })

    # ---- EMPLOYEE LAST-RESULTS LOCK ----
    # Employees who already have a stored result never see the upload form and
    # cannot re-upload: they always see their own last results (read-only).
    is_employee = bool(profile and profile.role == UserProfile.ROLE_EMPLOYEE)
    employee_has_results = (
        is_employee and PredictionLog.objects.filter(user=request.user).exists()
    )
    posted_files = request.method == "POST" and bool(
        request.FILES.getlist("files") or "file" in request.FILES
    )
    if employee_has_results and (request.method == "GET" or posted_files):
        result, results, result_json = _build_rop_review(request.user)
        viewed_user = request.user
        locked_employee = True
        return render(request, "index2.html", {
            "result": result,
            "results": results,
            "error": error,
            "result_json": result_json,
            "is_reviewer": is_reviewer,
            "chat_mode": chat_mode,
            "viewed_user": viewed_user,
            "review_mode": review_mode,
            "locked_employee": locked_employee,
            "can_edit_notes": bool(is_reviewer and viewed_user is not None),
        })

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

                # notes keys so the template never KeyErrors on a fresh upload
                result.setdefault("doctor_assessment", "")
                result.setdefault("doctor_recommendation", "")

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
        "result_json": result_json,
        "is_reviewer": is_reviewer,
        "chat_mode": chat_mode,
        "viewed_user": viewed_user,
        "review_mode": review_mode,
        "locked_employee": locked_employee,
        "can_edit_notes": bool(is_reviewer and viewed_user is not None),
    })

def submit_feedback(request):
    """
    Save expert feedback (corrected labels + comment) for ONE specific uploaded
    image. The image is identified by 'feedback_image_name', which the UI sets
    from the image the reviewer chose. Returns JSON so the results page stays
    intact (submitted via fetch()).
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed."}, status=405)

    if not request.user.is_authenticated:
        return JsonResponse({"error": "Please log in to submit feedback."}, status=403)

    image_name = (request.POST.get("feedback_image_name") or "").strip()
    corrected_class = request.POST.get("corrected_class") or None
    stage_corrected = request.POST.get("stage_corrected_class") or None
    zone_corrected = request.POST.get("corrected_zone") or None
    comment = request.POST.get("review_comment") or None

    # Whose record does this feedback belong to? By default the submitter's own,
    # but a doctor/manager reviewing an employee (?employee=<id>) corrects THAT
    # employee's record.
    target = request.user
    employee_id = request.POST.get("employee_id")
    if employee_id:
        from usac.models import UserProfile
        from django.contrib.auth.models import User
        prof = getattr(request.user, "profile", None)
        if prof and prof.role in (UserProfile.ROLE_DOCTOR, UserProfile.ROLE_MANAGER):
            cand = User.objects.filter(id=employee_id).select_related("profile").first()
            tprof = getattr(cand, "profile", None) if cand else None
            if tprof:
                same_company = bool(tprof.company_id) and tprof.company_id == prof.company_id
                manages = bool(tprof.company_id) and tprof.company.manager_id == request.user.id
                if same_company or manages:
                    target = cand
                else:
                    return JsonResponse(
                        {"error": "You do not have permission to review this employee."},
                        status=403,
                    )

    # Match the most recent prediction for this user and (if provided) file name.
    logs = PredictionLog.objects.filter(user=target)
    if image_name:
        logs = logs.filter(file_name=image_name)
    log = logs.order_by("-timestamp").first()

    if log is None:
        return JsonResponse(
            {"error": "No matching prediction was found for the selected image."},
            status=404,
        )

    log.corrected_class = corrected_class
    log.stage_corrected_class = stage_corrected
    log.zone_corrected_class = zone_corrected
    log.review_comment = comment
    log.save(update_fields=[
        "corrected_class",
        "stage_corrected_class",
        "zone_corrected_class",
        "review_comment",
    ])

    return JsonResponse({"ok": True, "file_name": log.file_name})


@csrf_exempt
def save_notes(request):
    """
    Persist a reviewer's case notes (clinical assessment + recommendation) for a
    specific employee. Reviewer-only (doctor/manager) with the same
    same-company/managed-company permission used by submit_feedback. Saves to the
    newest log of that employee's latest batch. Returns JSON.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed."}, status=405)

    if not request.user.is_authenticated:
        return JsonResponse({"error": "Please log in."}, status=403)

    from usac.models import UserProfile
    from django.contrib.auth.models import User

    prof = getattr(request.user, "profile", None)
    if not (prof and prof.role in (UserProfile.ROLE_DOCTOR, UserProfile.ROLE_MANAGER)):
        return JsonResponse(
            {"error": "You do not have permission to save notes."}, status=403
        )

    employee_id = request.POST.get("employee_id")
    target = User.objects.filter(id=employee_id).select_related("profile").first()
    tprof = getattr(target, "profile", None) if target else None
    if tprof is None:
        return JsonResponse({"error": "Employee not found."}, status=404)

    same_company = bool(tprof.company_id) and tprof.company_id == prof.company_id
    manages = bool(tprof.company_id) and tprof.company.manager_id == request.user.id
    if not (same_company or manages):
        return JsonResponse(
            {"error": "You do not have permission to review this employee."},
            status=403,
        )

    log = PredictionLog.objects.filter(user=target).order_by("-timestamp").first()
    if log is None:
        return JsonResponse(
            {"error": "No prediction was found for this employee."}, status=404
        )

    log.doctor_assessment = request.POST.get("doctor_assessment") or ""
    log.doctor_recommendation = request.POST.get("doctor_recommendation") or ""
    log.save(update_fields=["doctor_assessment", "doctor_recommendation"])

    return JsonResponse({"ok": True})


# ---------------------------------------------------------------------------
# Chat (RAG) endpoint — served directly by Django so the assistant works with a
# plain `python manage.py runserver`, with no separate FastAPI/uvicorn service
# (and no Celery/Redis). The knowledge base is indexed once at startup (see
# single_rop/apps.py); each request only retrieves + generates. All the heavy
# lifting lives in single_rop/chat_service.py.
# ---------------------------------------------------------------------------
@csrf_exempt
def chat(request):
    """
    Chat endpoint. Accepts JSON:
      { query, chat_history, diagnostic_context_text, use_web }
    and returns:
      { answer, local_context_documents, web_context_documents, combined_context_truncated }
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed."}, status=405)

    try:
        payload = json.loads((request.body or b"").decode("utf-8") or "{}")
    except Exception:
        payload = {}

    query = (payload.get("query") or "").strip()
    if not query:
        return JsonResponse({"error": "Empty query."}, status=400)

    chat_history = payload.get("chat_history", "") or ""
    diagnostic_context_text = (payload.get("diagnostic_context_text") or "")[:4000]
    use_web = payload.get("use_web", True)
    attachments = payload.get("attachments") or []

    try:
        from . import chat_service
        result = chat_service.chat_answer(
            query=query,
            diagnostic_context_text=diagnostic_context_text,
            chat_history=chat_history,
            use_web=use_web,
            attachments=attachments,
        )
        return JsonResponse(result)
    except Exception as ex:
        import traceback
        traceback.print_exc()
        return JsonResponse(
            {"error": f"Failed to process query. Details: {ex}"},
            status=500,
        )


@csrf_exempt
def chat_stream(request):
    """
    Streaming chat endpoint (Server-Sent Events). Same JSON body as `chat`, but
    responds with a `text/event-stream` of newline-delimited events:
      data: {"type":"stage",  "stage":"searching", "label":"Searching the web", ...}
      data: {"type":"token",  "text":"…"}
      data: {"type":"done",   "sources":[…]}
    The UI uses these to show live progress and inline source citations.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed."}, status=405)

    try:
        payload = json.loads((request.body or b"").decode("utf-8") or "{}")
    except Exception:
        payload = {}

    query = (payload.get("query") or "").strip()
    if not query:
        return JsonResponse({"error": "Empty query."}, status=400)

    chat_history = payload.get("chat_history", "") or ""
    diagnostic_context_text = (payload.get("diagnostic_context_text") or "")[:4000]
    use_web = payload.get("use_web", True)
    attachments = payload.get("attachments") or []

    def event_stream():
        from . import chat_service
        try:
            for event in chat_service.chat_stream(
                query=query,
                diagnostic_context_text=diagnostic_context_text,
                chat_history=chat_history,
                use_web=use_web,
                attachments=attachments,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as ex:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(ex)})}\n\n"

    resp = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    resp["Cache-Control"] = "no-cache"
    resp["X-Accel-Buffering"] = "no"  # disable proxy buffering (nginx)
    return resp


@csrf_exempt
def chat_upload(request):
    """Attach an uploaded document (PDF/DOCX/TXT/MD) to THIS conversation.

    The file is not added to the shared knowledge base — it is a per-chat
    attachment the assistant reads to answer questions about it (ChatGPT-style).
    Returns a doc_id the client sends back with each message.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed."}, status=405)
    f = request.FILES.get("file")
    if not f:
        return JsonResponse({"error": "No file received."}, status=400)
    try:
        from . import chat_service
        info = chat_service.add_attachment(f.read(), f.name)
        return JsonResponse({"ok": True, **info})
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": f"Failed to read the document: {e}"}, status=500)


@csrf_exempt
def chat_transcribe(request):
    """Speech-to-text with automatic Persian/English detection (Whisper)."""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed."}, status=405)
    audio = request.FILES.get("audio")
    if not audio:
        return JsonResponse({"error": "No audio received."}, status=400)
    try:
        from . import chat_service
        result = chat_service.transcribe_audio(audio.read(), audio.name or "audio.webm")
        return JsonResponse({"ok": True, **result})
    except chat_service.RateLimited:
        return JsonResponse(
            {"error": "The voice service is busy right now (rate limit). "
                      "Please wait a moment and try again."},
            status=429)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse(
            {"error": "Couldn't transcribe the audio. Please try again."},
            status=500)


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
