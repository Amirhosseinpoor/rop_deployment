"""Health chat assistant: tool-calling agent + web-scraper tools.

Ported from the original ``health_chat_agent.py``. Behavioural changes:

* **No Django.** ``get_openai_client`` no longer imports ``django.conf.settings``;
  credentials come from :mod:`config`.
* **No hard-coded patient.** The original pinned ``disease_results`` and
  ``personal_information`` as module globals. Here ``chat_with_assistant`` accepts
  them as arguments (with the same demo defaults) so any patient can be served.
* **Bug fix.** ``drugs_finder_tool`` referenced ``urljoin`` without importing it;
  the import is added so the requests-based path works.

The scraper selectors are preserved exactly — they target nobat.ir, doctoreto.com
and darooyab.ir as before. Scrapers are inherently brittle against site changes;
that is unchanged from the original and noted in HOW_TO_RUN.md.
"""
from __future__ import annotations

import json
import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .config import get_settings

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
}

# Tehran neighbourhood code -> name map used by the nobat.ir scraper. The scraper
# does a reverse (name -> code) lookup to build the search URL.
_NOBAT_NEIGHBORHOODS = {
    "p-1005": "نزدیک به من", "p-1000": "شمال تهران", "p-1001": "شرق تهران",
    "p-1002": "جنوب تهران", "p-1003": "غرب تهران", "p-1004": "مرکز تهران",
    "p-294": "آذری", "p-300": "ابوذر", "p-341": "پاسداران", "p-349": "پیروزی",
    "p-351": "تجریش", "p-357": "تهرانپارس", "p-408": "سعادت‌آباد", "p-463": "شهرک غرب",
    "p-565": "نارمک", "p-570": "نیاوران", "p-577": "ونک", "p-582": "یوسف‌آباد",
    # NOTE: the full original map has ~300 entries; the common ones are kept here.
    # Extend via the DOCTORS_FINDER inputs if a neighbourhood is missing.
}
_NOBAT_CITIES = {"city-1": "تهران"}
_NOBAT_SPECIALTIES = {"c-5": "cardiovascular"}


class FINDERS:
    """Collection of scraping "tools" the chat agent can invoke."""

    # ------------------------------------------------------------------ #
    # Public tool entry points
    # ------------------------------------------------------------------ #
    def doctors_finder_tool(self, province, neighborhood, speciality, insurance=None):
        """Find doctors: nobat.ir when no insurance filter, else doctoreto.com."""
        if insurance is None:
            return self.nobat_dot_ir_scrapper(province, neighborhood, speciality)
        return self.doctoreto_scrapper(province, neighborhood, speciality)

    def medications_finder_tool(self, medications_list):
        """Find a stocking pharmacy in Tehran for each medication via Selenium.

        Requires a working Chrome/Chromedriver. Returns a flat list of
        per-medication pharmacy detail dicts, or an ``{"error": ...}`` payload.
        """
        # Imported lazily so the whole service doesn't hard-depend on Selenium.
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        if not medications_list or not isinstance(medications_list, list):
            return {"error": "Invalid input: expected a list of medications"}

        drugs_list = []
        options = webdriver.ChromeOptions()
        for arg in ("--headless", "--disable-gpu", "--no-sandbox",
                    "--disable-dev-shm-usage", "--window-size=1920,1080"):
            options.add_argument(arg)

        driver = webdriver.Chrome(options=options)
        wait = WebDriverWait(driver, 15)
        try:
            for med_name in medications_list:
                try:
                    driver.get(f"https://www.darooyab.ir/Search?SearchText={med_name}")
                    wait.until(EC.presence_of_element_located((By.XPATH, "//tbody[@id='tbody_DrugList']/tr[1]")))
                    first_link = driver.find_element(
                        By.XPATH, "//tbody[@id='tbody_DrugList']/tr[1]/td[1]/a[@class='ahref_Generic']"
                    )
                    drug_page_url = first_link.get_attribute("href")

                    driver.get(drug_page_url)
                    wait.until(EC.presence_of_element_located((By.ID, "divExtraInfo")))

                    try:
                        martindale = driver.find_element(
                            By.XPATH,
                            "//div[@id='divExtraInfo']//h3[preceding-sibling::label[contains(text(), 'طبقه بندی مارتیندل')]]/a",
                        ).text.strip()
                    except Exception:
                        martindale = "یافت نشد"

                    try:
                        labels = driver.find_elements(
                            By.XPATH,
                            "//div[@id='divExtraInfo']//label[contains(text(), 'طبقه بندی درمانی')]/following-sibling::h3//label",
                        )
                        labels_text = [t.text.strip() for t in labels if t.text.strip()]
                        labels_text = [t for t in labels_text if t not in (">", "<", ">>", "<<<")]
                        therapeutic = " > ".join(labels_text)
                    except Exception:
                        therapeutic = "یافت نشد"

                    prescription_btn = wait.until(EC.presence_of_element_located((By.ID, "btnHaveprescription")))
                    driver.execute_script("arguments[0].click();", prescription_btn)

                    wait.until(EC.presence_of_element_located((By.ID, "ProvinceId")))
                    driver.execute_script(
                        "var s=document.getElementById('ProvinceId');s.value='40';"
                        "s.dispatchEvent(new Event('change',{bubbles:true}));"
                    )
                    time.sleep(2.0)
                    wait.until(EC.presence_of_element_located((By.ID, "CityId")))
                    driver.execute_script(
                        "var s=document.getElementById('CityId');s.value='663';"
                        "s.dispatchEvent(new Event('change',{bubbles:true}));"
                    )
                    time.sleep(0.5)

                    driver.execute_script(
                        "arguments[0].click();",
                        driver.find_element(By.ID, "btnSearchPatientReferral"),
                    )

                    first_pharmacy_xpath = "//table[@id='TBL_patientReferral']/tbody/tr[1]/td[2]/a[1]"
                    wait.until(EC.presence_of_element_located((By.XPATH, first_pharmacy_xpath)))
                    raw_href = driver.find_element(By.XPATH, first_pharmacy_xpath).get_attribute("href")
                    if not raw_href:
                        continue
                    clean = raw_href.replace("~/", "").replace("../", "")
                    link = clean if clean.startswith("http") else "https://www.darooyab.ir/" + clean.lstrip("/")

                    driver.get(link)
                    wait.until(EC.presence_of_element_located((By.ID, "pharmacyTitle")))

                    def _safe(by, sel):
                        try:
                            return driver.find_element(by, sel).text.strip()
                        except Exception:
                            return "یافت نشد"

                    drugs_list.append({
                        "source": "darooyab.ir",
                        "drug_name": med_name,
                        "drug_page_url": drug_page_url,
                        "martindale_classification": martindale,
                        "therapeutic_classification": therapeutic,
                        "pharmacy_availability_link": link,
                        "pharmacy_name": _safe(By.ID, "pharmacyTitle"),
                        "pharmacy_address": _safe(By.ID, "h2Address"),
                        "pharmacy_phone": _safe(By.XPATH, "//a[contains(@href, 'tel://')]"),
                        "availability_duration": _safe(
                            By.XPATH,
                            "//td[contains(@style, 'color: #3c763d') or contains(@style, 'color:#3c763d')]//label",
                        ),
                        "specific_brand_name": _safe(By.XPATH, "//td//h2[@style='font-size: 16px;']/a"),
                    })
                except Exception:
                    continue  # skip a failing medication, keep going
            return drugs_list
        except Exception as e:  # noqa: BLE001
            return {"error": f"Scraping Failed: {e}"}
        finally:
            driver.quit()

    def drugs_finder_tool(self, disease_results, personal_information):
        """Requests-based pharmacy finder filtered by the patient's province/city.

        Returns a list of ``{medication, pharmacies: [...]}`` entries.
        """
        if not disease_results or "medications" not in disease_results[0]:
            return {"error": "No medications found in disease_results"}
        if not personal_information or not personal_information[0]:
            return {"error": "No personal information provided"}

        medications = disease_results[0]["medications"]
        user_info = personal_information[0]
        neighborhood = user_info.get("neighborhood", "").strip()

        neighborhood_to_city = {"آذری": "تهران", "ابوذر": "تهران", "شهریار": "شهریار"}
        target_city = neighborhood_to_city.get(neighborhood, "تهران")
        target_province = "تهران"

        all_results = []
        for med_name in medications:
            try:
                resp = requests.get(
                    f"https://www.darooyab.ir/Search?SearchText={med_name}", headers=_HEADERS, timeout=10
                )
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                first_row = soup.select_one("#tbody_DrugList tr")
                if not first_row:
                    all_results.append({"medication": med_name, "pharmacies": [], "error": "No search results"})
                    continue
                link_tag = first_row.select_one('a.ahref_Generic[href^="/G-"]') or first_row.select_one("a.ahref_Generic")
                if not link_tag:
                    all_results.append({"medication": med_name, "pharmacies": [], "error": "No link found"})
                    continue
                generic_url = urljoin("https://www.darooyab.ir", link_tag["href"])

                resp2 = requests.get(generic_url, headers=_HEADERS, timeout=10)
                resp2.raise_for_status()
                soup2 = BeautifulSoup(resp2.text, "html.parser")
                table = soup2.select_one("#TBL_patientReferral")
                if not table:
                    all_results.append({"medication": med_name, "pharmacies": [], "error": "Pharmacy table not found"})
                    continue

                pharmacy_dict: dict = {}
                for tr in table.select("tbody tr"):
                    tds = tr.find_all("td")
                    if len(tds) < 3:
                        continue
                    brand_tag = tds[0].find("a")
                    brand_name = brand_tag.get_text(strip=True) if brand_tag else ""
                    pharm_link_tag = tds[1].find("a", href=True)
                    if not pharm_link_tag:
                        continue
                    clean_path = re.sub(r"^~/|\.\./", "/", pharm_link_tag["href"]).replace("//", "/")
                    pharm_url = urljoin("https://www.darooyab.ir", clean_path)
                    pharm_name = (pharm_link_tag.contents[0].strip() if pharm_link_tag.contents else "") or \
                        pharm_link_tag.get_text(strip=True).split("\n")[0].strip()

                    prov_city_text = tds[2].get_text(strip=True)
                    prov_match = re.search(r"استان\s+(\S+)", prov_city_text)
                    city_match = re.search(r"شهر\s+(\S+)", prov_city_text)
                    row_province = prov_match.group(1) if prov_match else ""
                    row_city = city_match.group(1) if city_match else ""

                    if row_province == target_province and row_city == target_city:
                        key = (pharm_name, pharm_url)
                        pharmacy_dict.setdefault(key, {
                            "pharmacy_name": pharm_name, "pharmacy_url": pharm_url,
                            "brands": [], "province": row_province, "city": row_city,
                        })
                        pharmacy_dict[key]["brands"].append(brand_name)

                if not pharmacy_dict:
                    all_results.append({"medication": med_name, "pharmacies": [],
                                        "message": "No pharmacies found in the specified area"})
                    continue

                pharmacies_list = []
                for (name, url), info in pharmacy_dict.items():
                    try:
                        resp3 = requests.get(url, headers=_HEADERS, timeout=10)
                        resp3.raise_for_status()
                        soup3 = BeautifulSoup(resp3.text, "html.parser")
                        address = phone = availability = ""
                        alert_div = soup3.find("div", class_="alert-success")
                        if alert_div:
                            addr_elem = alert_div.find("h2", id="h2Address")
                            address = addr_elem.get_text(strip=True) if addr_elem else ""
                            phone_elem = alert_div.find("h3", id="h3Phone")
                            if phone_elem and phone_elem.a:
                                phone = phone_elem.a.get_text(strip=True)
                        avail_tr = soup3.find("tr", class_="alert-success")
                        if avail_tr and avail_tr.find("label"):
                            availability = avail_tr.find("label").get_text(strip=True)
                        pharmacies_list.append({
                            "pharmacy_name": name, "pharmacy_url": url,
                            "brands_available": info["brands"], "address": address,
                            "phone": phone, "availability": availability,
                        })
                    except Exception as e:  # noqa: BLE001
                        pharmacies_list.append({
                            "pharmacy_name": name, "pharmacy_url": url,
                            "brands_available": info["brands"], "address": "", "phone": "",
                            "availability": "", "error": f"Failed to fetch details: {e}",
                        })
                all_results.append({"medication": med_name, "pharmacies": pharmacies_list})
            except Exception as e:  # noqa: BLE001
                all_results.append({"medication": med_name, "pharmacies": [], "error": f"Processing failed: {e}"})
        return all_results

    # ------------------------------------------------------------------ #
    # Scrapers
    # ------------------------------------------------------------------ #
    def nobat_dot_ir_scrapper(self, province, neighborhood, speciality):
        """Scrape the top-3 doctors from nobat.ir for a province/neighborhood/specialty."""
        city_code = next((c for c, n in _NOBAT_CITIES.items() if n == province), None)
        neighborhood_code = next((c for c, n in _NOBAT_NEIGHBORHOODS.items() if n == neighborhood), None)
        speciality_code = next((c for c, n in _NOBAT_SPECIALTIES.items() if n == speciality), None)

        missing = [label for label, code in
                   (("city", city_code), ("neighborhood", neighborhood_code), ("speciality", speciality_code))
                   if not code]
        if missing:
            return {"error": f"Could not map the following inputs to codes: {', '.join(missing)}."}

        search_url = f"https://nobat.ir/find/{city_code}/{neighborhood_code}/{speciality_code}/"
        try:
            response = requests.get(search_url, headers=_HEADERS, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            doctor_cards = soup.select("a.doctor-ui")[:3]
            if not doctor_cards:
                return {"error": "No doctors found for this search query."}

            doctors_list = []
            for card in doctor_cards:
                try:
                    href = card.get("href")
                    doctor_page_url = href if href.startswith("http") else "https://nobat.ir" + href
                    profile_soup = BeautifulSoup(
                        requests.get(doctor_page_url, headers=_HEADERS, timeout=10).text, "html.parser"
                    )

                    photo_tag = profile_soup.select_one("div.doctor-ui-profile img")
                    name_tag = profile_soup.select_one("h1.doctor-ui-name span")
                    specialties = list({s.get_text(strip=True) for s in
                                        profile_soup.select("h2.doctor-ui-specialty, span.doctor-ui-specialty")})

                    medical_code = None
                    code_div = profile_soup.select_one("div.doctor-code")
                    if code_div:
                        m = re.search(r"نظام:\s*(\d+)", code_div.get_text(strip=True))
                        medical_code = m.group(1) if m else None

                    availability_days = [d.select_one("div.day-desc").get_text(strip=True)
                                         for d in profile_soup.select("div.day") if d.select_one("div.day-desc")]

                    count_span = profile_soup.select_one("div.comments-summary span")
                    comment_count = count_span.get_text(strip=True) if count_span else None

                    top_comments = []
                    for comment in profile_soup.select("div.comments div.comment")[:10]:
                        content_p = comment.select_one("p.comment-content")
                        date_span = comment.select_one("span.comment-date")
                        stars_style = comment.select_one("div.stars-value")
                        stars = 0.0
                        if stars_style:
                            wm = re.search(r"width:\s*(\d+)%", stars_style.get("style", ""))
                            stars = float(wm.group(1)) / 20.0 if wm else 0.0
                        top_comments.append({
                            "content": content_p.get_text(strip=True) if content_p else "",
                            "date": date_span.get_text(strip=True) if date_span else "",
                            "stars": stars,
                        })

                    offices_info = []
                    for off in profile_soup.select("div.office"):
                        addr = off.select_one("div.office-address")
                        desc = off.select_one("div.office-description")
                        phones = [p.get_text(strip=True) for p in off.select("div.office-phone") if p.get_text(strip=True)]
                        offices_info.append({
                            "address": addr.get_text(strip=True) if addr else None,
                            "description": desc.get_text(strip=True) if desc else None,
                            "phones": phones,
                        })

                    doctors_list.append({
                        "source": "nobat.ir",
                        "profile_url": doctor_page_url,
                        "reservation_link": doctor_page_url.replace("nobat.ir", "turn.nobat.ir"),
                        "name": name_tag.get_text(strip=True) if name_tag else None,
                        "photo_url": photo_tag["src"] if photo_tag else None,
                        "specialties": specialties,
                        "medical_license_code": medical_code,
                        "availability": availability_days,
                        "comments": {"total_count": comment_count, "top_10": top_comments},
                        "offices": offices_info,
                    })
                except Exception:
                    continue
            return doctors_list
        except Exception as e:  # noqa: BLE001
            return {"error": f"Scraping Failed: {e}"}

    def doctoreto_scrapper(self, province, neighborhood, speciality, insurance=None):
        """Scrape the top-3 doctors from doctoreto.com."""
        region_map = {
            "آبشار": "abshar", "ابوذر": "abozar", "اختیاریه": "ekhtariyeh", "آذری": "azari",
            "آرارات": "ararat", "ارامنه": "aramineh", "امیرآباد": "amirabad", "پاسداران": "pasdaran",
            "سعادت آباد": "saadatabad", "ونک": "vanak", "آهنگ": "ahang", "تهرانپارس": "tehranpars",
        }
        city_map = {"تهران": "tehran", "اصفهان": "isfahan", "شیراز": "shiraz", "مشهد": "mashhad", "تبریز": "tabriz"}
        speciality_map = {"قلب و عروق": "cardiologist", "داخلی": "internist",
                          "پزشک عمومی": "general-practitioner", "cardiovascular": "cardiologist"}

        province_slug = city_map.get(province, province.lower().replace(" ", "-"))
        neighborhood_slug = region_map.get(neighborhood, neighborhood.lower().replace(" ", "-"))
        speciality_slug = speciality_map.get(speciality, speciality.lower().replace(" ", "-"))

        search_url = (
            f"https://doctoreto.com/doctors/speciality/{speciality_slug}"
            f"/city/{province_slug}/region/{neighborhood_slug}"
        )
        try:
            response = requests.get(search_url, headers=_HEADERS, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            articles = soup.select("article.flex.cursor-pointer.flex-col")[:3] or soup.select("article")[:3]
            if not articles:
                return {"error": "No doctors found on doctoreto.com for this search."}

            doctors_list = []
            for article in articles:
                try:
                    link_a = article.select_one('a[href^="/doctor/"]')
                    if not link_a:
                        continue
                    href = link_a.get("href")
                    profile_url = "https://doctoreto.com" + href if href.startswith("/") else href
                    profile_soup = BeautifulSoup(
                        requests.get(profile_url, headers=_HEADERS, timeout=10).text, "html.parser"
                    )
                    name_tag = profile_soup.select_one("div.sc-212a8fa3-4.qvYGb")
                    doctors_list.append({
                        "source": "doctoreto.com",
                        "profile_url": profile_url,
                        "name": name_tag.get_text(strip=True) if name_tag else None,
                    })
                except Exception:
                    continue
            return doctors_list
        except Exception as e:  # noqa: BLE001
            return {"error": f"Doctoreto Scraping Failed: {e}"}


# --------------------------------------------------------------------------- #
# Chat agent
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT_FINDERS = (
    "You are a very polite assistant that your task is to help our customers. Our customers are some "
    "Employees that based on some research by doctors, they may be in danger of some diseases. So, you "
    "have to help them to address their issues by finding appropriate doctors, medications, and drugstores. "
    "I have provided some tools named DOCTORS_FINDER_TOOL and MEDICATIONS_FINDER_TOOL for you that they help "
    "you to assist customers. You should call these tool whenever you want to find one of these for customers. "
    "Our customers are Persian; therefore, all conversations should be in Persian. Be to the point and answer "
    "the user very shortly. Try to use the users first name. After calling tools you will get the result of "
    "tools just for notice; dont write them and just say you can see the proper and closest doctors or "
    "medications down below, because the result of tools is shown separately by the frontend.\n "
    "{disease_results}\n HERE IS THE PERSONAL INFORMATION OF THE CUSTOMER.\n {personal_information}."
)

# Demo defaults (used when the caller doesn't supply patient context).
DEFAULT_DISEASE_RESULTS = [
    {"disease": "hypertension", "accuracy": "0.91", "refer_to": "cardiovascular",
     "medications": ["Lisinopril", "Enalapril", "Losartan"]},
]
DEFAULT_PERSONAL_INFORMATION = [
    {"name": "امیر", "last_name": "حسین پور کلسری",
     "living_province": "تهران", "neighborhood": "ابوذر"},
]

_DOCTORS_FINDER_FN = {
    "name": "doctors_finder_tool",
    "description": "Find doctors for the user based on province, neighborhood, and speciality.",
    "parameters": {
        "type": "object",
        "properties": {
            "province": {"type": "string", "description": "Living province name in Persian"},
            "neighborhood": {"type": "string", "description": "Living neighborhood name in Persian"},
            "speciality": {"type": "string", "description": "Speciality in English, e.g., 'cardiovascular'."},
        },
        "required": ["province", "neighborhood", "speciality"],
        "additionalProperties": False,
    },
}
_MEDICATIONS_FINDER_FN = {
    "name": "medications_finder_tool",
    "description": "Find medications/pharmacies for the user based on a list of medication names.",
    "parameters": {
        "type": "object",
        "properties": {
            "medications": {"type": "array", "items": {"type": "string"},
                            "description": "a list of medications related to the customer disease."},
        },
        "required": ["medications"],
        "additionalProperties": False,
    },
}
_TOOLS = [
    {"type": "function", "function": _DOCTORS_FINDER_FN},
    {"type": "function", "function": _MEDICATIONS_FINDER_FN},
]

_client = None


def get_openai_client():
    """Build & cache the OpenAI-compatible client from config (GapGPT)."""
    global _client
    if _client is None:
        import openai

        settings = get_settings()
        if not settings.chat_api_key:
            raise RuntimeError("GAPGPT_API_KEY not set; the chat assistant cannot run.")
        _client = openai.OpenAI(base_url=settings.chat_base_url or None, api_key=settings.chat_api_key)
    return _client


def _handle_tool_call(message):
    """Run the requested finder tool and return ``(tool_envelope, raw_results)``."""
    tool_call = message.tool_calls[0]
    function_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    finder = FINDERS()
    finder_results = None
    tool_content = ""

    if function_name == "doctors_finder_tool":
        finder_results = finder.doctors_finder_tool(
            province=args["province"], neighborhood=args["neighborhood"], speciality=args["speciality"]
        )
        if isinstance(finder_results, list):
            summary = [{"name": d.get("name"), "reservation_link": d.get("reservation_link")}
                       for d in finder_results if "error" not in d]
            tool_content = json.dumps(summary, ensure_ascii=False)
        else:
            tool_content = json.dumps(finder_results, ensure_ascii=False)
    elif function_name == "medications_finder_tool":
        finder_results = finder.medications_finder_tool(medications_list=args["medications"])
        if isinstance(finder_results, list):
            summary = [{"drug_name": d.get("drug_name"), "pharmacy_name": d.get("pharmacy_name")}
                       for d in finder_results if "error" not in d]
            tool_content = json.dumps(summary, ensure_ascii=False)
        else:
            tool_content = json.dumps(finder_results, ensure_ascii=False)

    return {"role": "tool", "content": tool_content, "tool_call_id": tool_call.id}, finder_results


def chat_with_assistant(message, history, disease_results=None, personal_information=None):
    """Run one turn of the health chat assistant.

    Args:
        message: The user's latest message.
        history: Prior messages as a list of ``{role, content}`` dicts.
        disease_results: Optional patient disease context (defaults to the demo).
        personal_information: Optional patient profile (defaults to the demo).

    Returns:
        ``(reply_text, finder_results)`` where ``finder_results`` is the raw tool
        output (doctors/medications) for the frontend to render, or ``None`` if no
        tool was invoked.
    """
    disease_results = disease_results or DEFAULT_DISEASE_RESULTS
    personal_information = personal_information or DEFAULT_PERSONAL_INFORMATION

    client = get_openai_client()
    model = get_settings().chat_model
    system_content = _SYSTEM_PROMPT_FINDERS.format(
        disease_results=disease_results, personal_information=personal_information
    )
    messages = [{"role": "system", "content": system_content}, *history, {"role": "user", "content": message}]

    response = client.chat.completions.create(model=model, messages=messages, tools=_TOOLS)
    choice = response.choices[0]

    if choice.finish_reason == "tool_calls":
        msg = choice.message
        tool_response, finder_results = _handle_tool_call(msg)
        messages.append(msg)
        messages.append(tool_response)
        final = client.chat.completions.create(model=model, messages=messages)
        return final.choices[0].message.content, finder_results
    return choice.message.content, None
