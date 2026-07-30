# doctors_marketplace/agents/whatsapp.py
"""Interaction agents that reach the patient over WhatsApp (WAHA): reminders and
proactive follow-ups. They persist a ScheduledMessage; the `deliver_due_messages`
command sends it at the right time. If no phone number is known, the agent asks
the model to collect the patient's WhatsApp number first."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from django.utils import timezone

from .base import BaseAgent, AgentContext, AgentResult
from .registry import register

log = logging.getLogger(__name__)


def _resolve_phone(args, ctx: AgentContext):
    """Explicit arg → prior reminder for this user → None (must ask)."""
    phone = (args.get("phone") or "").strip()
    if phone:
        return phone
    from ..models import ScheduledMessage
    user = getattr(ctx, "user", None)
    if user is not None and getattr(user, "is_authenticated", False):
        prev = (ScheduledMessage.objects.filter(user=user).exclude(phone="")
                .order_by("-created_at").first())
        if prev:
            return prev.phone
    return ""


def _parse_iso(when: str):
    when = (when or "").strip()
    if not when:
        return None
    try:
        dt = datetime.fromisoformat(when.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timezone.is_naive(dt):
        dt = timezone.make_aware(dt, timezone.get_current_timezone())
    return dt


@register
class SetReminderAgent(BaseAgent):
    key = "set_reminder"
    name = "Set a reminder"
    icon = "bell"
    category = "Follow-up (WhatsApp)"
    description = ("Schedule a one-off reminder (e.g. take medication, attend an appointment) "
                   "delivered to the patient on WhatsApp at a specific time. Convert the user's "
                   "natural-language time to an ISO-8601 datetime using the current time given to you.")
    input_desc = ("when = ISO-8601 datetime (e.g. 2026-07-26T09:00); message = the reminder text; "
                  "phone = WhatsApp number (ask the user if unknown).")
    output_desc = "Confirmation that the reminder was scheduled (or a request for the phone number)."
    example = ('User: "Remind me to take my antibiotic tomorrow at 8am." → when="2026-07-26T08:00", '
               'message="Time to take your antibiotic." → schedules a WhatsApp reminder.')
    stage_label = "Scheduling a reminder"
    run_order = 80
    parameters = {
        "type": "object",
        "properties": {
            "when": {"type": "string", "description": "ISO-8601 datetime, e.g. 2026-07-26T09:00."},
            "message": {"type": "string", "description": "The reminder text to send."},
            "phone": {"type": "string", "description": "WhatsApp number (digits, with country code)."},
        },
        "required": ["when", "message"],
    }

    def run(self, args, ctx):
        from ..models import ScheduledMessage
        dt = _parse_iso(args.get("when"))
        if dt is None:
            return AgentResult(content="I need a valid date/time. Convert it to ISO-8601 "
                                       "(YYYY-MM-DDTHH:MM) using the current time.", ok=False)
        message = (args.get("message") or "").strip()
        if not message:
            return AgentResult(content="What should the reminder say?", ok=False)
        phone = _resolve_phone(args, ctx)
        if not phone:
            return AgentResult(content="ASK_USER: I need the patient's WhatsApp number (with "
                                       "country code) to send this reminder. Ask for it, then retry.",
                               ok=False)
        sm = ScheduledMessage.objects.create(
            user=getattr(ctx, "user", None) if getattr(getattr(ctx, "user", None), "is_authenticated", False) else None,
            doctor=ctx.doctor, session=ctx.session, phone=phone, text=message,
            kind=ScheduledMessage.Kind.REMINDER, send_at=dt)
        local = timezone.localtime(dt)
        past = " (that time is in the past — it will send at the next delivery run)" if dt <= timezone.now() else ""
        out = f"Reminder scheduled for {local:%Y-%m-%d %H:%M} to WhatsApp {phone}{past}."
        log.info("REMINDER | #%s @ %s -> %s", sm.pk, local, phone)
        return AgentResult(content=out, display="Reminder set", ok=True)


@register
class ScheduleFollowUpAgent(BaseAgent):
    key = "schedule_followup"
    name = "Schedule a follow-up"
    icon = "calendar"
    category = "Follow-up (WhatsApp)"
    description = ("Schedule a proactive check-in a set time from now (minutes, hours, or days), "
                   "delivered on WhatsApp (e.g. 'how are your symptoms in 3 days?' or 'check on me "
                   "in 2 minutes'). Use to close the loop on a patient's care.")
    input_desc = ("Provide at least one of minutes / hours / days from now; message = optional "
                  "check-in text; phone = WhatsApp number (ask if unknown).")
    output_desc = "Confirmation that the follow-up was scheduled (or a request for the phone number)."
    example = ('User: "Check on me in 2 minutes." → minutes=2 → schedules a WhatsApp check-in 2 '
               'minutes from now. "Check on me in 3 days." → days=3.')
    stage_label = "Scheduling a follow-up"
    run_order = 82
    parameters = {
        "type": "object",
        "properties": {
            "minutes": {"type": "integer", "description": "Minutes from now."},
            "hours": {"type": "integer", "description": "Hours from now."},
            "days": {"type": "integer", "description": "Days from now."},
            "message": {"type": "string", "description": "Optional check-in message."},
            "phone": {"type": "string", "description": "WhatsApp number (digits, with country code)."},
        },
    }

    def run(self, args, ctx):
        from ..models import ScheduledMessage

        def _int(v):
            try:
                return max(0, int(v))
            except (TypeError, ValueError):
                return 0
        total_minutes = _int(args.get("days")) * 1440 + _int(args.get("hours")) * 60 + _int(args.get("minutes"))
        if total_minutes <= 0:
            return AgentResult(content="In how long should I follow up (e.g. 2 minutes, 3 days)?", ok=False)
        total_minutes = min(total_minutes, 90 * 1440)  # cap at 90 days
        phone = _resolve_phone(args, ctx)
        if not phone:
            return AgentResult(content="ASK_USER: I need the patient's WhatsApp number (with "
                                       "country code) to schedule this follow-up. Ask for it, then retry.",
                               ok=False)
        message = (args.get("message") or "").strip() or (
            "Hi — just checking in on how you're feeling. Have your symptoms changed? "
            "Reply here and let your care team know if anything is worse.")
        send_at = timezone.now() + timedelta(minutes=total_minutes)
        sm = ScheduledMessage.objects.create(
            user=getattr(ctx, "user", None) if getattr(getattr(ctx, "user", None), "is_authenticated", False) else None,
            doctor=ctx.doctor, session=ctx.session, phone=phone, text=message,
            kind=ScheduledMessage.Kind.FOLLOWUP, send_at=send_at)
        local = timezone.localtime(send_at)
        when = (f"{total_minutes} minute(s)" if total_minutes < 60 else
                f"{total_minutes // 60} hour(s)" if total_minutes < 1440 else
                f"{total_minutes // 1440} day(s)")
        out = f"Follow-up scheduled for {local:%Y-%m-%d %H:%M} (in {when}) to WhatsApp {phone}."
        log.info("FOLLOWUP | #%s in %smin -> %s", sm.pk, total_minutes, phone)
        return AgentResult(content=out, display="Follow-up set", ok=True)
