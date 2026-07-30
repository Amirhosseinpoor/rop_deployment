# doctors_marketplace/tests_agents.py
"""Fast, network-free test suite for the modular agent system.

External services (openFDA, PubMed, Serper, WAHA) and the LLM are mocked so the
suite is hermetic and quick. Run:
    python manage.py test doctors_marketplace.tests_agents
"""
from __future__ import annotations

import time
from unittest import mock

from django.test import TestCase
from django.utils import timezone

from doctors_marketplace import agents as agents_pkg
from doctors_marketplace.agents.base import AgentContext
from doctors_marketplace.models import Doctor, ScheduledMessage


def ctx(doctor=None, user=None, query=""):
    return AgentContext(doctor=doctor, session=None, user=user, query=query,
                        now_iso="2026-07-25T20:30")


class FakeResp:
    def __init__(self, json_data=None, text="", status=200):
        self._json = json_data or {}
        self.text = text
        self.status_code = status

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


# --------------------------------------------------------------------------- #
# Registry / metadata
# --------------------------------------------------------------------------- #
class RegistryTests(TestCase):
    def test_all_13_registered(self):
        self.assertEqual(len(agents_pkg.all_agents()), 13)

    def test_catalog_complete(self):
        for entry in agents_pkg.catalog():
            for field in ("key", "name", "description", "input", "output", "example", "category"):
                self.assertTrue(entry.get(field), f"{entry.get('key')} missing {field}")

    def test_valid_keys_filters_unknown(self):
        self.assertEqual(agents_pkg.valid_keys(["medical_calculator", "nope"]), ["medical_calculator"])

    def test_ordering_safety_first(self):
        got = [a.key for a in agents_pkg.get_agents(
            ["search_web", "red_flag_check", "medical_calculator"])]
        self.assertEqual(got[0], "red_flag_check")  # run_order 0

    def test_tool_schemas_valid(self):
        for a in agents_pkg.all_agents().values():
            s = a.tool_schema()
            self.assertEqual(s["type"], "function")
            self.assertEqual(s["function"]["name"], a.key)
            self.assertIn("parameters", s["function"])


# --------------------------------------------------------------------------- #
# Pure clinical agents
# --------------------------------------------------------------------------- #
class CalculatorTests(TestCase):
    def setUp(self):
        self.a = agents_pkg.get_agent("medical_calculator")

    def _run(self, calc, values, sex="male"):
        return self.a.run({"calc": calc, "values": values, "sex": sex}, ctx()).content

    def test_bmi(self):
        self.assertIn("26.1", self._run("bmi", {"weight_kg": 80, "height_cm": 175}))

    def test_egfr(self):
        self.assertIn("69", self._run("egfr", {"creatinine_mg_dl": 1.2, "age": 60}, "male"))

    def test_crcl(self):
        out = self._run("creatinine_clearance", {"age": 60, "weight_kg": 80, "creatinine_mg_dl": 1.2})
        self.assertIn("mL/min", out)

    def test_chads_vasc(self):
        out = self._run("chads_vasc", {"age": 72, "htn": 1, "diabetes": 1}, "female")
        self.assertIn("= 4", out)

    def test_anion_gap(self):
        self.assertIn("16", self._run("anion_gap", {"na": 140, "cl": 100, "hco3": 24}))

    def test_iv_rate(self):
        self.assertIn("42", self._run("iv_drip_rate", {"volume_ml": 1000, "time_min": 480, "drop_factor": 20}))

    def test_peds_dose(self):
        self.assertIn("300", self._run("pediatric_dose", {"weight_kg": 30, "mg_per_kg": 10}))

    def test_missing_input(self):
        r = self.a.run({"calc": "bmi", "values": {"weight_kg": 80}}, ctx())
        self.assertFalse(r.ok)

    def test_unknown_calc(self):
        self.assertFalse(self.a.run({"calc": "xyz", "values": {}}, ctx()).ok)

    def test_latency(self):
        t0 = time.perf_counter()
        for _ in range(50):
            self._run("bmi", {"weight_kg": 70, "height_cm": 170})
        self.assertLess(time.perf_counter() - t0, 0.5)  # 50 calcs well under 0.5s


class LabInterpreterTests(TestCase):
    def setUp(self):
        self.a = agents_pkg.get_agent("lab_interpreter")

    def test_low_female_hemoglobin(self):
        out = self.a.run({"test": "hemoglobin", "value": 10.2, "sex": "female"}, ctx()).content
        self.assertIn("LOW", out); self.assertIn("12.0", out)

    def test_high_glucose(self):
        self.assertIn("HIGH", self.a.run({"test": "glucose_fasting", "value": 140}, ctx()).content)

    def test_alias(self):
        self.assertIn("NORMAL", self.a.run({"test": "a1c", "value": 5.0}, ctx()).content)

    def test_unknown_test(self):
        self.assertFalse(self.a.run({"test": "unobtanium", "value": 1}, ctx()).ok)


class RedFlagTests(TestCase):
    def setUp(self):
        self.a = agents_pkg.get_agent("red_flag_check")

    def test_chest_pain_banner(self):
        r = self.a.run({"text": "crushing chest pain radiating to my arm"}, ctx())
        self.assertTrue(r.ui and r.ui.get("banner"))
        self.assertIn("EMERGENCY", r.content)

    def test_persian_emergency(self):
        r = self.a.run({"text": "درد قفسه سینه دارم"}, ctx())
        self.assertTrue(r.ui)

    def test_no_flag(self):
        r = self.a.run({"text": "what foods help digestion"}, ctx())
        self.assertIsNone(r.ui)

    def test_is_pre_pass_first(self):
        self.assertTrue(self.a.pre_pass)
        self.assertEqual(self.a.run_order, 0)


class TriageTests(TestCase):
    def test_triage_uses_llm(self):
        fake = mock.Mock()
        fake.chat.return_value = ('{"urgency":"urgent","rationale":"possible meningitis",'
                                  '"action":"same-day care","timeframe":"hours"}')
        with mock.patch("doctors_marketplace.services.llm.LLMClient", return_value=fake):
            r = agents_pkg.get_agent("symptom_triage").run({"symptoms": "fever and stiff neck"}, ctx())
        self.assertIn("urgent", r.content)


# --------------------------------------------------------------------------- #
# Retrieval agents (mocked)
# --------------------------------------------------------------------------- #
class RetrievalTests(TestCase):
    def test_kb_search(self):
        class Doc:
            def __init__(self, t, c): self.metadata = {"title": t}; self.page_content = c
        with mock.patch("doctors_marketplace.agents.retrieval.retrieve_context",
                        return_value=[Doc("Protocol", "stone protocol text")]):
            r = agents_pkg.get_agent("search_knowledge_base").run({"query": "stones"}, ctx(doctor=object()))
        self.assertEqual(len(r.sources), 1)
        self.assertEqual(r.sources[0]["type"], "local")

    def test_web_search(self):
        class Doc:
            def __init__(self): self.metadata = {"source": "https://x.org/a", "title": "A"}; self.page_content = "web text"
        with mock.patch("doctors_marketplace.agents.retrieval.websearch.serper_search",
                        return_value=[{"link": "https://x.org/a", "title": "A"}]), \
             mock.patch("doctors_marketplace.agents.retrieval.websearch.rank_web_pages",
                        return_value=[Doc()]):
            r = agents_pkg.get_agent("search_web").run({"query": "q"}, ctx())
        self.assertEqual(r.sources[0]["domain"], "x.org")

    def test_fetch_url(self):
        with mock.patch("doctors_marketplace.agents.retrieval.websearch.scrape_url",
                        return_value="page body text"):
            r = agents_pkg.get_agent("fetch_url").run({"url": "https://example.com/p"}, ctx())
        self.assertTrue(r.sources and r.sources[0]["url"] == "https://example.com/p")

    def test_fetch_url_rejects_non_url(self):
        self.assertFalse(agents_pkg.get_agent("fetch_url").run({"url": "not a url"}, ctx()).ok)

    def test_pubmed(self):
        def fake_get(url, params=None, timeout=None):
            if "esearch" in url:
                return FakeResp(json_data={"esearchresult": {"idlist": ["111"]}})
            if "esummary" in url:
                return FakeResp(json_data={"result": {"111": {
                    "title": "Metformin study", "fulljournalname": "NEJM",
                    "pubdate": "2023", "authors": [{"name": "Smith J"}]}}})
            return FakeResp(text="Abstract body here.")
        with mock.patch("doctors_marketplace.agents.retrieval.requests.get", side_effect=fake_get):
            r = agents_pkg.get_agent("search_pubmed").run({"query": "metformin"}, ctx())
        self.assertEqual(len(r.sources), 1)
        self.assertIn("pubmed", r.sources[0]["url"])


# --------------------------------------------------------------------------- #
# Drug agents (mocked openFDA)
# --------------------------------------------------------------------------- #
def _label(**sections):
    base = {"openfda": {"generic_name": [sections.pop("name", "drug")]}}
    base.update({k: [v] for k, v in sections.items()})
    return FakeResp(json_data={"results": [base]})


class DrugTests(TestCase):
    def test_drug_lookup(self):
        with mock.patch("doctors_marketplace.agents.drugs.requests.get",
                        return_value=_label(name="ibuprofen",
                                             indications_and_usage="pain relief",
                                             warnings="GI bleeding risk")):
            r = agents_pkg.get_agent("drug_lookup").run({"name": "ibuprofen"}, ctx())
        self.assertIn("pain relief", r.content)
        self.assertTrue(r.sources)

    def test_drug_lookup_not_found(self):
        with mock.patch("doctors_marketplace.agents.drugs.requests.get",
                        return_value=FakeResp(json_data={"results": []})):
            self.assertFalse(agents_pkg.get_agent("drug_lookup").run({"name": "zzz"}, ctx()).ok)

    def test_interactions_needs_two(self):
        self.assertFalse(agents_pkg.get_agent("drug_interactions").run({"drugs": ["warfarin"]}, ctx()).ok)

    def test_interactions(self):
        with mock.patch("doctors_marketplace.agents.drugs.requests.get",
                        return_value=_label(name="warfarin", drug_interactions="NSAIDs raise bleeding")):
            r = agents_pkg.get_agent("drug_interactions").run(
                {"drugs": ["warfarin", "ibuprofen"]}, ctx())
        self.assertIn("bleeding", r.content.lower())

    def test_contraindications_match(self):
        with mock.patch("doctors_marketplace.agents.drugs.requests.get",
                        return_value=_label(name="amoxicillin",
                                            contraindications="hypersensitivity to penicillin")):
            r = agents_pkg.get_agent("check_contraindications").run(
                {"drug": "amoxicillin", "profile": {"allergies": ["penicillin"]}}, ctx())
        self.assertIn("POTENTIAL MATCH", r.content)


# --------------------------------------------------------------------------- #
# WAHA + scheduling
# --------------------------------------------------------------------------- #
class WahaTests(TestCase):
    def test_not_configured(self):
        from doctors_marketplace.services import waha
        with mock.patch("doctors_marketplace.services.waha._env", return_value=None):
            ok, detail = waha.send_whatsapp("989121234567", "hi")
        self.assertFalse(ok)

    def test_send_builds_chatid(self):
        from doctors_marketplace.services import waha
        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["url"] = url; captured["json"] = json
            return FakeResp(status=201)
        with mock.patch("doctors_marketplace.services.waha._env",
                        side_effect=lambda *a, **k: "http://waha:3000" if a[0] == "WAHA_URL" else k.get("default")), \
             mock.patch("doctors_marketplace.services.waha.requests.post", side_effect=fake_post):
            ok, _ = waha.send_whatsapp("+98 912 123 4567", "hi")
        self.assertTrue(ok)
        self.assertEqual(captured["json"]["chatId"], "989121234567@c.us")
        self.assertTrue(captured["url"].endswith("/api/sendText"))


class SchedulingTests(TestCase):
    def setUp(self):
        from django.contrib.auth import get_user_model
        self.user = get_user_model().objects.create(username="p1")
        self.doctor = Doctor.objects.create(name="Doc", specialization="hypertension")

    def test_set_reminder_needs_phone(self):
        r = agents_pkg.get_agent("set_reminder").run(
            {"when": "2030-01-01T09:00", "message": "take pill"},
            ctx(doctor=self.doctor, user=self.user))
        self.assertFalse(r.ok)
        self.assertIn("ASK_USER", r.content)

    def test_set_reminder_creates(self):
        r = agents_pkg.get_agent("set_reminder").run(
            {"when": "2030-01-01T09:00", "message": "take pill", "phone": "989121234567"},
            ctx(doctor=self.doctor, user=self.user))
        self.assertTrue(r.ok)
        sm = ScheduledMessage.objects.get()
        self.assertEqual(sm.kind, "reminder")
        self.assertEqual(sm.phone, "989121234567")

    def test_followup_reuses_prior_phone(self):
        ScheduledMessage.objects.create(user=self.user, doctor=self.doctor, phone="989120000000",
                                        text="x", send_at=timezone.now())
        r = agents_pkg.get_agent("schedule_followup").run(
            {"days": 3}, ctx(doctor=self.doctor, user=self.user))
        self.assertTrue(r.ok)
        self.assertEqual(ScheduledMessage.objects.filter(kind="followup").first().phone, "989120000000")

    def test_deliver_command(self):
        from django.core.management import call_command
        from io import StringIO
        ScheduledMessage.objects.create(user=self.user, phone="989121234567", text="due",
                                        send_at=timezone.now() - timezone.timedelta(minutes=1))
        ScheduledMessage.objects.create(user=self.user, phone="989121234567", text="future",
                                        send_at=timezone.now() + timezone.timedelta(days=1))
        with mock.patch("doctors_marketplace.services.waha.is_configured", return_value=True), \
             mock.patch("doctors_marketplace.services.waha.send_whatsapp",
                        return_value=(True, "HTTP 201")) as sender:
            call_command("deliver_due_messages", stdout=StringIO())
        self.assertEqual(sender.call_count, 1)  # only the due one
        self.assertEqual(ScheduledMessage.objects.filter(status="sent").count(), 1)
        self.assertEqual(ScheduledMessage.objects.filter(status="pending").count(), 1)


# --------------------------------------------------------------------------- #
# Orchestration loop (mocked LLM) — modularity, ordering, no cross-conflict
# --------------------------------------------------------------------------- #
class FakeToolCall:
    def __init__(self, name, args, cid="c1"):
        self.id = cid
        self.function = mock.Mock(name=name)
        self.function.name = name
        self.function.arguments = args


class FakeMsg:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class OrchestrationTests(TestCase):
    def setUp(self):
        self.doctor = Doctor.objects.create(name="Doc", specialization="hypertension",
                                            agents=["red_flag_check", "medical_calculator"])

    def _fake_llm(self, complete_msgs, stream_text="Your BMI is 26.1."):
        fake = mock.Mock()
        fake.complete.side_effect = complete_msgs
        fake.chat_stream.return_value = iter([stream_text])
        fake.chat.return_value = stream_text
        return fake

    def test_greeting_skips_tools(self):
        fake = self._fake_llm([], stream_text="Hello! How can I help?")
        with mock.patch("doctors_marketplace.services.agent_runtime.LLMClient", return_value=fake):
            events = list(agents_pkg_run("hello", self.doctor))
        stages = [e["stage"] for e in events if e["type"] == "stage"]
        # red_flag pre-pass still runs, but no tool-decision LLM call was made
        self.assertEqual(fake.complete.call_count, 0)
        self.assertIn("red_flag_check", stages)
        self.assertTrue(any(e["type"] == "token" for e in events))

    def test_tool_call_flow_and_source_numbering(self):
        # Round 1: model calls medical_calculator; Round 2: no tool calls -> compose.
        calls = [FakeMsg(tool_calls=[FakeToolCall(
            "medical_calculator", '{"calc":"bmi","values":{"weight_kg":80,"height_cm":175}}')]),
            FakeMsg(content="")]
        fake = self._fake_llm(calls)
        with mock.patch("doctors_marketplace.services.agent_runtime.LLMClient", return_value=fake):
            events = list(agents_pkg_run("what is my BMI at 80kg 175cm", self.doctor))
        stages = [e["stage"] for e in events if e["type"] == "stage"]
        self.assertIn("medical_calculator", stages)
        self.assertTrue(any("26.1" in e.get("text", "") for e in events if e["type"] == "token"))

    def test_disabled_agent_not_exposed(self):
        # Only medical_calculator enabled -> web tool must not appear in the schema.
        self.doctor.agents = ["medical_calculator"]
        captured = {}
        def complete(messages, tools=None, tool_choice=None):
            captured["tools"] = [t["function"]["name"] for t in (tools or [])]
            return FakeMsg(content="")
        fake = mock.Mock()
        fake.complete.side_effect = complete
        fake.chat_stream.return_value = iter(["ok"])
        with mock.patch("doctors_marketplace.services.agent_runtime.LLMClient", return_value=fake):
            list(agents_pkg_run("hi there question please", self.doctor))
        self.assertIn("medical_calculator", captured["tools"])
        self.assertNotIn("search_web", captured["tools"])

    def test_emergency_banner_event(self):
        fake = self._fake_llm([FakeMsg(content="")], stream_text="Call emergency now.")
        with mock.patch("doctors_marketplace.services.agent_runtime.LLMClient", return_value=fake):
            events = list(agents_pkg_run("I have crushing chest pain and cannot breathe", self.doctor))
        self.assertTrue(any(e["type"] == "ui" and e.get("banner") for e in events))


def agents_pkg_run(query, doctor):
    from doctors_marketplace.services import agent_runtime
    return agent_runtime.run_agentic_answer(
        doctor=doctor, session=None, user=None, query=query,
        system_text="You are a helpful assistant.", history_msgs=[],
        user_content=query, enabled_keys=doctor.agents, now_iso="2026-07-25T20:30")
