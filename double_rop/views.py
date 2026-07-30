from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse, HttpResponseForbidden
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User
from .utils import get_prediction
import base64
import json
import time
from .models import PredictionResult
from usac.models import UserProfile


def _build_kc_result(latest):
    """Reconstruct a `result` dict from a stored `PredictionResult` so a reviewer
    (or an employee viewing their own last screening) can see it without
    re-uploading. Includes the doctor's case-notes."""
    return {
        "predictions": {
            "left_eye": {
                "label": latest.left_label,
                "probability": latest.left_probability,
            },
            "right_eye": {
                "label": latest.right_label,
                "probability": latest.right_probability,
            },
            "z_class": {
                "label": latest.z_class_label,
                "probability": latest.z_class_probability,
            },
        },
        "image_data": {
            "left": latest.left_image_url or (latest.left_image.url if latest.left_image else ""),
            "right": latest.right_image_url or (latest.right_image.url if latest.right_image else ""),
        },
        "inference_time": f"{latest.inference_time:.2f}s",
        "left_image_url": latest.left_image_url,
        "right_image_url": latest.right_image_url,
        "corrected_left_label": latest.corrected_left_label,
        "corrected_right_label": latest.corrected_right_label,
        "corrected_z_label": latest.corrected_z_label,
        "review_comment": latest.review_comment,
        "doctor_assessment": latest.doctor_assessment or "",
        "doctor_recommendation": latest.doctor_recommendation or "",
    }


def home(request):
    result = None
    error = None

    # ---- shared role-aware context ----
    profile = getattr(request.user, "profile", None) if request.user.is_authenticated else None
    is_reviewer = bool(profile and profile.role in (UserProfile.ROLE_DOCTOR, UserProfile.ROLE_MANAGER))
    is_employee = bool(profile and profile.role == UserProfile.ROLE_EMPLOYEE)
    viewed_user = None
    review_mode = False
    locked_employee = False

    # ---- REVIEW MODE: a doctor/manager opens ?employee=<id> ----
    employee_id = request.GET.get("employee")
    if request.method == "GET" and employee_id and is_reviewer:
        try:
            target = User.objects.get(pk=employee_id)
        except (User.DoesNotExist, ValueError, TypeError):
            return HttpResponseForbidden("Employee not found.")

        target_profile = getattr(target, "profile", None)
        allowed = False
        if target_profile is not None:
            if profile.role == UserProfile.ROLE_DOCTOR:
                allowed = bool(profile.company_id and target_profile.company_id == profile.company_id)
            elif profile.role == UserProfile.ROLE_MANAGER:
                same_company = bool(profile.company_id and target_profile.company_id == profile.company_id)
                manages = bool(target_profile.company and target_profile.company.manager_id == request.user.id)
                allowed = same_company or manages
        if not allowed:
            return HttpResponseForbidden("You do not have permission to view this record.")

        viewed_user = target
        review_mode = True
        try:
            latest = PredictionResult.objects.filter(user=target).latest("created_at")
            result = _build_kc_result(latest)
        except PredictionResult.DoesNotExist:
            result = None

        return render(request, "index_double2.html", {
            "result": result,
            "error": error,
            "is_reviewer": is_reviewer,
            "chat_mode": "doctor" if is_reviewer else "employee",
            "viewed_user": viewed_user,
            "review_mode": review_mode,
            "locked_employee": locked_employee,
            "can_edit_notes": is_reviewer and viewed_user is not None,
        })

    # ---- EMPLOYEE LAST-RESULTS LOCK (GET): show their own latest, no upload ----
    if request.method == "GET" and is_employee:
        latest = PredictionResult.objects.filter(user=request.user).order_by("-created_at").first()
        if latest is not None:
            result = _build_kc_result(latest)
            viewed_user = request.user
            locked_employee = True
            return render(request, "index_double2.html", {
                "result": result,
                "error": error,
                "is_reviewer": is_reviewer,
                "chat_mode": "doctor" if is_reviewer else "employee",
                "viewed_user": viewed_user,
                "review_mode": review_mode,
                "locked_employee": locked_employee,
                "can_edit_notes": is_reviewer and viewed_user is not None,
            })

    if request.method == "POST":

        # ---- EMPLOYEE LAST-RESULTS LOCK (POST): never re-predict; re-render ----
        if is_employee and "feedback_mode" not in request.POST:
            latest = PredictionResult.objects.filter(user=request.user).order_by("-created_at").first()
            if latest is not None:
                result = _build_kc_result(latest)
                viewed_user = request.user
                locked_employee = True
                return render(request, "index_double2.html", {
                    "result": result,
                    "error": error,
                    "is_reviewer": is_reviewer,
                    "chat_mode": "doctor" if is_reviewer else "employee",
                    "viewed_user": viewed_user,
                    "review_mode": review_mode,
                    "locked_employee": locked_employee,
                    "can_edit_notes": is_reviewer and viewed_user is not None,
                })

        if "feedback_mode" in request.POST and request.user.is_authenticated:
            try:
                # A doctor/manager reviewing an employee (employee_id) corrects THAT
                # employee's record; otherwise the submitter's own latest result.
                fb_target = request.user
                fb_employee_id = request.POST.get("employee_id")
                if fb_employee_id and profile and profile.role in (UserProfile.ROLE_DOCTOR, UserProfile.ROLE_MANAGER):
                    cand = User.objects.filter(pk=fb_employee_id).select_related("profile").first()
                    cprof = getattr(cand, "profile", None) if cand else None
                    if cprof and (
                        (bool(profile.company_id) and cprof.company_id == profile.company_id)
                        or (cprof.company and cprof.company.manager_id == request.user.id)
                    ):
                        fb_target = cand
                    else:
                        return HttpResponseForbidden("You do not have permission to review this employee.")
                latest_result = PredictionResult.objects.filter(user=fb_target).latest("created_at")
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
                    "right_image_url": latest_result.right_image_url,
                    "doctor_assessment": latest_result.doctor_assessment or "",
                    "doctor_recommendation": latest_result.doctor_recommendation or "",
                }

                return render(request, "index_double2.html", {
                    "result": None,
                    "feedback_saved": True
                })

            except Exception as ex:
                error = f"Feedback error: {str(ex)}"
                return render(request, "index_double2.html", {"error": error})

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
                    "inference_time": inference_time,
                    "doctor_assessment": "",
                    "doctor_recommendation": "",
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

    return render(request, "index_double2.html", {
        "result": result,
        "error": error,
        "is_reviewer": is_reviewer,
        "chat_mode": "doctor" if is_reviewer else "employee",
        "viewed_user": viewed_user,
        "review_mode": review_mode,
        "locked_employee": locked_employee,
        "can_edit_notes": is_reviewer and viewed_user is not None,
    })


@csrf_exempt
def save_notes(request):
    """Persist a reviewer's case-notes (clinical assessment + recommendation) onto
    an employee's latest KC screening. Reviewer-only, same-company/managed check."""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed."}, status=405)
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Please log in."}, status=403)

    profile = getattr(request.user, "profile", None)
    if not (profile and profile.role in (UserProfile.ROLE_DOCTOR, UserProfile.ROLE_MANAGER)):
        return JsonResponse({"error": "You do not have permission to edit notes."}, status=403)

    employee_id = request.POST.get("employee_id")
    try:
        target = User.objects.select_related("profile").get(pk=employee_id)
    except (User.DoesNotExist, ValueError, TypeError):
        return JsonResponse({"error": "Employee not found."}, status=404)

    target_profile = getattr(target, "profile", None)
    allowed = False
    if target_profile is not None:
        if profile.role == UserProfile.ROLE_DOCTOR:
            allowed = bool(profile.company_id and target_profile.company_id == profile.company_id)
        elif profile.role == UserProfile.ROLE_MANAGER:
            same_company = bool(profile.company_id and target_profile.company_id == profile.company_id)
            manages = bool(target_profile.company and target_profile.company.manager_id == request.user.id)
            allowed = same_company or manages
    if not allowed:
        return JsonResponse(
            {"error": "You do not have permission to review this employee."}, status=403)

    try:
        latest = PredictionResult.objects.filter(user=target).latest("created_at")
    except PredictionResult.DoesNotExist:
        return JsonResponse({"error": "No screening found for this employee."}, status=404)

    latest.doctor_assessment = request.POST.get("doctor_assessment") or ""
    latest.doctor_recommendation = request.POST.get("doctor_recommendation") or ""
    latest.save(update_fields=["doctor_assessment", "doctor_recommendation"])
    return JsonResponse({"ok": True})


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


# --------------------------------------------------------------------------- #
# KC chat assistant — same features as the ROP chat, grounded in KC knowledge.
# Served in-process; endpoints live under /kc/ (see double_rop/urls.py).
# --------------------------------------------------------------------------- #
@csrf_exempt
def chat(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed."}, status=405)
    try:
        payload = json.loads((request.body or b"").decode("utf-8") or "{}")
    except Exception:
        payload = {}
    query = (payload.get("query") or "").strip()
    if not query:
        return JsonResponse({"error": "Empty query."}, status=400)
    try:
        from . import chat_service
        return JsonResponse(chat_service.chat_answer(
            query=query,
            diagnostic_context_text=(payload.get("diagnostic_context_text") or "")[:4000],
            chat_history=payload.get("chat_history", "") or "",
            use_web=payload.get("use_web", True),
            attachments=payload.get("attachments") or [],
        ))
    except Exception as ex:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": f"Failed to process query. Details: {ex}"}, status=500)


@csrf_exempt
def chat_stream(request):
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
    resp["X-Accel-Buffering"] = "no"
    return resp


@csrf_exempt
def chat_upload(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed."}, status=405)
    f = request.FILES.get("file")
    if not f:
        return JsonResponse({"error": "No file received."}, status=400)
    try:
        from . import chat_service
        return JsonResponse({"ok": True, **chat_service.add_kc_attachment(f.read(), f.name)})
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": f"Failed to read the document: {e}"}, status=500)


@csrf_exempt
def chat_transcribe(request):
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
                      "Please wait a moment and try again."}, status=429)
    except Exception:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": "Couldn't transcribe the audio. Please try again."}, status=500)
