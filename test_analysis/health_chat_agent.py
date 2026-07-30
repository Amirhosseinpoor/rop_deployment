import json
import re
import logging
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

# ------------------------------------------------------------------
# Logging setup — verbose terminal logging for the health chat agent.
# A dedicated StreamHandler guarantees records reach the terminal/stdout
# even if Django's own LOGGING config does not capture this logger.
# ------------------------------------------------------------------
logger = logging.getLogger("test_analysis.health_chat_agent")
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    ))
    logger.addHandler(_console_handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

load_dotenv()
gapgpt_base_url = os.getenv("GAPGPT_BASE_URL")
gapgpt_api_key = os.getenv("GAPGPT_API_KEY")
gapgpt_model = os.getenv("GAPGPT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

logger.info("📦 health_chat_agent module imported | GAPGPT_BASE_URL set=%s | GAPGPT_API_KEY set=%s | GAPGPT_MODEL=%s",
            bool(gapgpt_base_url), bool(gapgpt_api_key), gapgpt_model)

# ------------------------------
# Nobat.ir constants
# ------------------------------
NOBAT_CITIES = {
    "city-1": "تهران"
}

NOBAT_NEIGHBORHOODS = {
    "p-1005": "نزدیک به من", "p-1000": "شمال تهران", "p-1001": "شرق تهران",
    "p-1002": "جنوب تهران", "p-1003": "غرب تهران", "p-1004": "مرکز تهران",
    "p-292": "آجودانیه", "p-293": "آذربایجان", "p-294": "آذری",
    "p-295": "آرارات", "p-296": "آرژانتین", "p-298": "آهنگ",
    "p-299": "ابراهیم‌آباد", "p-300": "ابوذر", "p-301": "احتشامیه",
    "p-302": "اختیاریه", "p-303": "ارامنه", "p-304": "ارم",
    "p-305": "ازگل", "p-306": "استاد معین", "p-308": "اسفندیاری",
    "p-309": "اسکندری", "p-310": "اسلام‌‌آباد", "p-311": "افسریه",
    "p-312": "اقدسیه", "p-313": "اکباتان", "p-314": "المهدی",
    "p-315": "الهیه", "p-316": "امام حسین(ع)", "p-317": "امامزاده قاسم",
    "p-318": "امانیه", "p-319": "امیر بهادر", "p-320": "امیرآباد",
    "p-321": "امیریه", "p-322": "اندیشه", "p-323": "اوقاف",
    "p-324": "اوین", "p-325": "ایرانشهر", "p-326": "بازار",
    "p-327": "باغ آذری", "p-328": "باغ رضوان", "p-329": "باقرخان",
    "p-330": "بریانک", "p-331": "بلوار کشاورز", "p-332": "بوستان ولایت",
    "p-334": "بهاران", "p-336": "بهجت‌آباد", "p-337": "بهمن‌ یار",
    "p-338": "پارک آزادگان", "p-339": "پارک شهر", "p-340": "پارک لاله",
    "p-341": "پاسداران", "p-343": "پامنار", "p-345": "پردیسان",
    "p-346": "پرند", "p-347": "پل رومی", "p-348": "پونک",
    "p-349": "پیروزی", "p-350": "پیکان شهر", "p-351": "تجریش",
    "p-352": "توحید", "p-353": "تهران‌سر", "p-354": "تهران‌نو",
    "p-355": "تهران‌ویلا", "p-357": "تهرانپارس", "p-358": "جردن",
    "p-359": "جماران", "p-360": "جمال‌زاده", "p-361": "جمشیدیه",
    "p-362": "جمهوری", "p-366": "جوادیه", "p-367": "جوانمرد قصاب",
    "p-368": "جی", "p-369": "جیحون", "p-370": "چهارصد دستگاه",
    "p-371": "چیتگر", "p-372": "چیتگر شمالی", "p-373": "چیذر",
    "p-375": "حسن‌آباد باقرفر", "p-376": "حسن‌آباد شمالی", "p-377": "حشمتیه",
    "p-378": "حکیمیه", "p-379": "خانی‌آباد", "p-380": "خزانه",
    "p-381": "خلیج فارس", "p-382": "خورین", "p-383": "دارآباد",
    "p-384": "دانشگاه تهران", "p-385": "دبستان", "p-386": "دربند",
    "p-387": "دردشت", "p-388": "درکه", "p-389": "دروازه شمیران",
    "p-390": "دروس", "p-391": "دریان‌نو‍", "p-392": "دزاشیب",
    "p-393": "دولاب", "p-394": "دولت‌آباد", "p-395": "ده‌ونک",
    "p-396": "دهکده المپیک", "p-397": "دیلمان جنوبی", "p-398": "رباط کریم",
    "p-400": "زرگنده", "p-401": "زعفرانیه", "p-402": "زهتابی",
    "p-403": "زینبیه", "p-404": "سازمان آب", "p-405": "سازمان برنامه",
    "p-406": "ستارخان", "p-407": "سرسبز", "p-408": "سعادت‌آباد",
    "p-409": "سعدآباد", "p-410": "سلسبیل", "p-411": "سلیمانیه",
    "p-412": "سنگلج", "p-413": "سوهانک", "p-414": "سهروردی جنوبی",
    "p-415": "سهروردی شمالی", "p-416": "سید خندان", "p-417": "شاپور",
    "p-418": "شادآباد", "p-419": "شادمان", "p-420": "شاهین",
    "p-421": "شریف‌آباد", "p-422": "شکوفه", "p-423": "شمس‌آباد",
    "p-424": "شمیران‌نو", "p-425": "شوش", "p-427": "شهر زیبا",
    "p-429": "شهرآرا", "p-432": "شهرک آپادانا", "p-433": "شهرک آزادی",
    "p-434": "شهرک آسمان", "p-435": "شهرک ابوذر", "p-436": "شهرک ارغوان",
    "p-437": "شهرک استقلال", "p-438": "شهرک امام خمینی", "p-439": "شهرک امید",
    "p-440": "شهرک امیرالمومنین", "p-441": "شهرک انصار", "p-442": "شهرک بخارایی",
    "p-443": "شهرک بعثت", "p-444": "شهرک پارس", "p-445": "شهرک پاسداران",
    "p-446": "شهرک پرواز", "p-447": "شهرک تختی", "p-448": "شهرک چشمه",
    "p-449": "شهرک دانشگاه", "p-450": "شهرک دریا", "p-451": "شهرک رضوان",
    "p-452": "شهرک رضویه", "p-453": "شهرک ژاندارمری", "p-454": "شهرک شاهین",
    "p-455": "شهرک شریعتی", "p-456": "شهرک شریفی", "p-457": "شهرک شهربانی",
    "p-458": "شهرک شهرداری", "p-459": "شهرک شهید مفتح", "p-460": "شهرک طالقانی",
    "p-461": "شهرک عباسپور", "p-462": "شهرک عباسی", "p-463": "شهرک غرب",
    "p-464": "شهرک غزالی", "p-465": "شهرک فاطمه الزهرا", "p-467": "شهرک فرهنگیان",
    "p-468": "شهرک فرهنگیان غرب", "p-469": "شهرک قدس", "p-470": "شهرک کاظمیه",
    "p-471": "شهرک کمالی", "p-472": "شهرک کوهسار", "p-473": "شهرک کیانشهر",
    "p-474": "شهرک گلستان", "p-475": "شهرک گلستان غربی", "p-476": "شهرک گلها",
    "p-477": "شهرک محلاتی", "p-478": "شهرک مسلمین", "p-479": "شهرک نفت",
    "p-480": "شهرک والفجر", "p-481": "شهرک ولیعصر", "p-482": "شهرک ویلاشهر",
    "p-483": "شهرک هجرت", "p-485": "شیان", "p-486": "شیخ الرئیس",
    "p-487": "شیخ هادی", "p-488": "شیرازی", "p-489": "صادقیه",
    "p-490": "صد دستگاه", "p-491": "ضرابخانه", "p-492": "طالقانی",
    "p-493": "طرشت", "p-494": "ظفر", "p-495": "ظهیرآباد",
    "p-496": "عباس‌آباد", "p-497": "عبدل‌آباد", "p-498": "عشرت‌آباد",
    "p-499": "علم و صنعت", "p-500": "علی‌آباد جنوبی", "p-501": "علی‌آباد شمالی",
    "p-502": "فاطمی", "p-503": "فدک", "p-504": "فرح‌آباد",
    "p-505": "فرحزاد", "p-506": "فرمانیه", "p-507": "فرودگاه مهر‌آباد",
    "p-508": "فشم", "p-509": "فلاح", "p-512": "قزل قلعه",
    "p-513": "قصر", "p-514": "قصر فیروزه ۱", "p-515": "قصر فیروزه ۲",
    "p-517": "قلهک", "p-518": "قنات‌کوثر", "p-519": "قیامدشت",
    "p-520": "قیطریه", "p-521": "کاروان", "p-522": "کاشانک",
    "p-523": "کاظم‌آباد", "p-525": "کامرانیه", "p-526": "کتابخانه ملی ایران",
    "p-527": "کریم‌آباد", "p-528": "کن", "p-529": "کوهک",
    "p-530": "کوی اسلام", "p-531": "کوی بیمه", "p-532": "کوی دانشگاه",
    "p-533": "کوی فردوس", "p-534": "کوی مهرآباد", "p-535": "کوی هفدهم شهریور",
    "p-536": "گاندی", "p-537": "گلاب دره", "p-538": "گمرک",
    "p-539": "گیشا (کوی نصر)", "p-540": "لواسان", "p-541": "لویزان",
    "p-542": "مبارک‌آباد", "p-543": "مبارک‌آباد بهشتی", "p-544": "مجموعه ورزشگاه آزادی",
    "p-545": "مجید‌آباد", "p-546": "مجیدیه", "p-547": "محمودیه",
    "p-548": "مرادآباد", "p-549": "مرزداران", "p-550": "مسعودیه",
    "p-551": "مشیریه", "p-553": "ملک‌آباد", "p-554": "منصوریه",
    "p-555": "منظریه", "p-556": "منیریه", "p-557": "مولوی",
    "p-558": "مهرآباد جنوبی", "p-559": "مهران", "p-560": "میدان آزادی",
    "p-561": "میدان انقلاب", "p-562": "میدان حر", "p-563": "میدان ولیعصر",
    "p-564": "میرداماد", "p-565": "نارمک", "p-566": "نازی‌آباد",
    "p-567": "نظام‌آباد", "p-568": "نعمت‌آباد", "p-569": "نواب",
    "p-570": "نیاوران", "p-571": "نیرو هوایی", "p-572": "وحیدیه",
    "p-574": "وردآورد", "p-575": "وصفنارد", "p-576": "ولنجک",
    "p-577": "ونک", "p-578": "هاشم‌آباد", "p-579": "هفت‌ حوض",
    "p-580": "یاخچی‌آباد", "p-581": "یافت‌آباد", "p-582": "یوسف‌آباد",
    "p-583": "شهر ری", "p-584": "سنائی", "p-586": "باغ فیض",
    "p-587": "بلوار فردوس", "p-588": "هفت تیر", "p-590": "جنت آباد",
    "p-591": "شهران", "p-592": "شریعتی", "p-593": "هروی",
    "p-595": "رسالت", "p-596": "ملاصدرا"
}

NOBAT_SPECIALTIES = {
    "c-5": "cardiovascular",
    # add more as needed
}


class FINDERS():
    def doctors_finder_tool(self, province, neighborhood, speciality, insurance=None):
        logger.info("🔧 doctors_finder_tool() | province=%r | neighborhood=%r | speciality=%r | insurance=%r",
                    province, neighborhood, speciality, insurance)
        if insurance is None:
            logger.debug("   ↳ no insurance → using nobat.ir scraper")
            nobat_dot_ir_dr_info = self.nobat_dot_ir_scrapper(province, neighborhood, speciality)
            logger.info("   ✅ nobat.ir scraper returned %s",
                        f"{len(nobat_dot_ir_dr_info)} doctors" if isinstance(nobat_dot_ir_dr_info, list)
                        else nobat_dot_ir_dr_info)
            return nobat_dot_ir_dr_info
        else:
            logger.debug("   ↳ insurance provided → using doctoreto scraper")
            doctoreto_dr_info = self.doctoreto_scrapper(province, neighborhood, speciality)
            logger.info("   ✅ doctoreto scraper returned %s",
                        f"{len(doctoreto_dr_info)} doctors" if isinstance(doctoreto_dr_info, list)
                        else doctoreto_dr_info)
            return doctoreto_dr_info

    def medications_finder_tool(self, medications_list):
            """
            For each medication in disease_results, finds available pharmacies in Tehran
            using Selenium Headless, extracts details, and returns a clean, flat list of results.
            """
            import time
            from selenium import webdriver
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            logger.info("🔧 medications_finder_tool() | medications_list=%r", medications_list)
            if not medications_list or not isinstance(medications_list, list):
                logger.warning("   ❌ invalid input (not a non-empty list)")
                return {"error": "Invalid input: expected a list of medications"}

            medications = medications_list
            drugs_list = []
            logger.info("   🌐 launching headless Chrome (Selenium) for %d medication(s)", len(medications))

            # کانفیگ بروزر سلنیوم به صورت Headless
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--window-size=1920,1080')

            driver = webdriver.Chrome(options=options)
            wait = WebDriverWait(driver, 15)

            try:
                for med_name in medications:
                    try:
                        logger.info("   💊 processing medication: %r", med_name)
                        search_url = f"https://www.darooyab.ir/Search?SearchText={med_name}"
                        logger.debug("      ↳ GET %s", search_url)
                        driver.get(search_url)

                        # ۱. انتظار برای ردیف اول جدول جستجو
                        wait.until(EC.presence_of_element_located((By.XPATH, "//tbody[@id='tbody_DrugList']/tr[1]")))
                        first_link_element = driver.find_element(By.XPATH, "//tbody[@id='tbody_DrugList']/tr[1]/td[1]/a[@class='ahref_Generic']")
                        drug_page_url = first_link_element.get_attribute("href")

                        # ۲. ورود به صفحه اختصاصی دارو
                        driver.get(drug_page_url)
                        wait.until(EC.presence_of_element_located((By.ID, "divExtraInfo")))

                        # ۳. استخراج طبقه‌بندی‌ها
                        try:
                            martindale_element = driver.find_element(By.XPATH, "//div[@id='divExtraInfo']//h3[preceding-sibling::label[contains(text(), 'طبقه بندی مارتیندل')]]/a")
                            martindale_classification = martindale_element.text.strip()
                        except:
                            martindale_classification = "یافت نشد"

                        try:
                            therapeutic_labels = driver.find_elements(By.XPATH, "//div[@id='divExtraInfo']//label[contains(text(), 'طبقه بندی درمانی')]/following-sibling::h3//label")
                            labels_text = [label.text.strip() for label in therapeutic_labels if label.text.strip()]
                            labels_text = [text for text in labels_text if text not in ('>', '<', '>>', '<<<')]
                            therapeutic_classification = " > ".join(labels_text)
                        except:
                            therapeutic_classification = "یافت نشد"

                        # ۴. باز کردن مودال/بخش موجودی داروخانه‌ها
                        prescription_btn = wait.until(EC.presence_of_element_located((By.ID, "btnHaveprescription")))
                        driver.execute_script("arguments[0].click();", prescription_btn)

                        # ۵. تنظیم استان و شهر روی تهران با تزریق جاوا اسکریپت
                        wait.until(EC.presence_of_element_located((By.ID, "ProvinceId")))
                        js_select_province = """
                        var provSelect = document.getElementById('ProvinceId');
                        provSelect.value = '40';
                        var event = new Event('change', { bubbles: true });
                        provSelect.dispatchEvent(event);
                        """
                        driver.execute_script(js_select_province)
                        time.sleep(2.0)  # لود ایجکس شهرها

                        wait.until(EC.presence_of_element_located((By.ID, "CityId")))
                        js_select_city = """
                        var citySelect = document.getElementById('CityId');
                        citySelect.value = '663';
                        var event = new Event('change', { bubbles: true });
                        citySelect.dispatchEvent(event);
                        """
                        driver.execute_script(js_select_city)
                        time.sleep(0.5)

                        # ۶. کلیک روی دکمه جستجوی موجودی
                        search_btn = driver.find_element(By.ID, "btnSearchPatientReferral")
                        driver.execute_script("arguments[0].click();", search_btn)

                        # ۷. استخراج لینک اولین داروخانه
                        first_pharmacy_link_xpath = "//table[@id='TBL_patientReferral']/tbody/tr[1]/td[2]/a[1]"
                        wait.until(EC.presence_of_element_located((By.XPATH, first_pharmacy_link_xpath)))
                        pharmacy_element = driver.find_element(By.XPATH, first_pharmacy_link_xpath)
                        raw_href = pharmacy_element.get_attribute("href")

                        if raw_href:
                            clean_href = raw_href.replace('~/', '').replace('../', '')
                            if not clean_href.startswith("http"):
                                pharmacy_availability_link = "https://www.darooyab.ir/" + clean_href.lstrip('/')
                            else:
                                pharmacy_availability_link = clean_href
                        else:
                            continue  # اگر لینکی نبود برو سراغ داروی بعدی

                        # ۸. ورود به صفحه اختصاصی داروخانه و استخراج جزییات نهایی
                        driver.get(pharmacy_availability_link)
                        wait.until(EC.presence_of_element_located((By.ID, "pharmacyTitle")))

                        try:
                            pharmacy_name = driver.find_element(By.ID, "pharmacyTitle").text.strip()
                        except:
                            pharmacy_name = "یافت نشد"

                        try:
                            pharmacy_address = driver.find_element(By.ID, "h2Address").text.strip()
                        except:
                            pharmacy_address = "یافت نشد"

                        try:
                            pharmacy_phone = driver.find_element(By.XPATH, "//a[contains(@href, 'tel://')]").text.strip()
                        except:
                            pharmacy_phone = "یافت نشد"

                        try:
                            availability_duration = driver.find_element(By.XPATH, "//td[contains(@style, 'color: #3c763d') or contains(@style, 'color:#3c763d')]//label").text.strip()
                        except:
                            availability_duration = "یافت نشد"

                        try:
                            specific_brand_name = driver.find_element(By.XPATH, "//td//h2[@style='font-size: 16px;']/a").text.strip()
                        except:
                            specific_brand_name = "یافت نشد"

                        # فرمت خروجی استاندارد و هماهنگ (دقیقاً مشابه الگوی ساختاریافته پزشکان)
                        drug_info = {
                            "source": "darooyab.ir",
                            "drug_name": med_name,
                            "drug_page_url": drug_page_url,
                            "martindale_classification": martindale_classification,
                            "therapeutic_classification": therapeutic_classification,
                            "pharmacy_availability_link": pharmacy_availability_link,
                            "pharmacy_name": pharmacy_name,
                            "pharmacy_address": pharmacy_address,
                            "pharmacy_phone": pharmacy_phone,
                            "availability_duration": availability_duration,
                            "specific_brand_name": specific_brand_name
                        }
                        drugs_list.append(drug_info)
                        logger.info("      ✅ scraped med=%r | pharmacy=%r | availability=%r",
                                    med_name, pharmacy_name, availability_duration)

                    except Exception as inner_e:
                        # نادیده گرفتن خطای یک دارو و ادامه فرآیند برای داروهای بعدی لیست
                        logger.warning("      ⚠️  skipping med=%r due to error: %s", med_name, inner_e)
                        continue

                logger.info("   ✅ medications_finder_tool done | %d drug record(s) collected", len(drugs_list))
                return drugs_list

            except Exception as e:
                logger.exception("   🔥 medications_finder_tool scraping failed: %s", e)
                return {"error": f"Scraping Failed: {str(e)}"}

            finally:
                logger.debug("   🧹 quitting Selenium driver")
                driver.quit()

    def drugs_finder_tool(self, disease_results, personal_information):
        """
        For each medication in disease_results, find pharmacies that stock it,
        filtered by the user's province and neighborhood (city).
        Returns a list of medications with pharmacy details including address, phone, and availability.
        """
        logger.info("🔧 drugs_finder_tool() | disease_results=%r | personal_information=%r",
                    disease_results, personal_information)
        if not disease_results or 'medications' not in disease_results[0]:
            logger.warning("   ❌ no medications found in disease_results")
            return {"error": "No medications found in disease_results"}
        if not personal_information or not personal_information[0]:
            logger.warning("   ❌ no personal information provided")
            return {"error": "No personal information provided"}

        medications = disease_results[0]['medications']
        user_info = personal_information[0]
        province = user_info.get('living_province', '').strip()
        neighborhood = user_info.get('neighborhood', '').strip()
        logger.info("   📍 province=%r | neighborhood=%r | medications=%r", province, neighborhood, medications)

        # Mapping of neighborhood (or city) to the expected city name in the pharmacy table
        # The table displays "شهر X". We'll build a dictionary of neighborhood -> city name
        # Fallback: if neighborhood not found, use "تهران" as the city.
        NEIGHBORHOOD_TO_CITY = {
            "آذری": "تهران",
            "ابوذر": "تهران",
            "شهریار": "شهریار",
            # Add more as needed, e.g.:
            # "آجودانیه": "تهران",
            # "پاسداران": "تهران",
            # ...
        }
        target_city = NEIGHBORHOOD_TO_CITY.get(neighborhood, "تهران")  # default to Tehran
        target_province = "تهران"  # we assume province is always تهران based on input
        logger.debug("   ↳ target_province=%r | target_city=%r", target_province, target_city)

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        all_results = []

        for med_name in medications:
            try:
                logger.info("   💊 [requests] processing medication: %r", med_name)
                # Step 1: Get the generic drug page URL (same as medications_finder_tool)
                search_url = f"https://www.darooyab.ir/Search?SearchText={med_name}"
                logger.debug("      ↳ GET %s", search_url)
                resp = requests.get(search_url, headers=headers, timeout=100)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, 'html.parser')

                first_row = soup.select_one('#tbody_DrugList tr')
                if not first_row:
                    logger.warning("      ⚠️  no search results for med=%r", med_name)
                    all_results.append({"medication": med_name, "pharmacies": [], "error": "No search results"})
                    continue

                link_tag = first_row.select_one('a.ahref_Generic[href^="/G-"]') or first_row.select_one(
                    'a.ahref_Generic')
                if not link_tag:
                    all_results.append({"medication": med_name, "pharmacies": [], "error": "No link found"})
                    continue
                generic_url = urljoin("https://www.darooyab.ir", link_tag['href'])

                # Step 2: Fetch the generic page and extract the pharmacy table
                resp2 = requests.get(generic_url, headers=headers, timeout=100)
                resp2.raise_for_status()
                soup2 = BeautifulSoup(resp2.text, 'html.parser')

                # The table is inside div#haveprescription -> div#lstPatientRefferalContent
                table = soup2.select_one('#TBL_patientReferral')
                if not table:
                    all_results.append({"medication": med_name, "pharmacies": [], "error": "Pharmacy table not found"})
                    continue

                # Step 3: Parse rows and filter by province/city
                pharmacy_dict = {}  # key: (pharmacy_name, pharmacy_link) -> list of brands
                for tr in table.select('tbody tr'):
                    tds = tr.find_all('td')
                    if len(tds) < 3:
                        continue

                    # Brand name (first td)
                    brand_tag = tds[0].find('a')
                    brand_name = brand_tag.get_text(strip=True) if brand_tag else ""

                    # Pharmacy info (second td)
                    pharm_link_tag = tds[1].find('a', href=True)
                    if not pharm_link_tag:
                        continue
                    pharm_relative_url = pharm_link_tag['href']
                    # Fix the ~/../../ pattern: replace ~/ with / and then navigate up
                    # Simple approach: remove leading ~ and all ../ sequences, then prepend /
                    clean_path = re.sub(r'^~/|\.\./', '/', pharm_relative_url).replace('//', '/')
                    pharm_url = urljoin("https://www.darooyab.ir", clean_path)

                    # Pharmacy name (text before <br> inside the a tag)
                    pharm_name = pharm_link_tag.contents[0].strip() if pharm_link_tag.contents else ""
                    if not pharm_name:
                        pharm_name = pharm_link_tag.get_text(strip=True).split('\n')[0].strip()

                    # Province and city (third td)
                    prov_city_text = tds[2].get_text(strip=True)
                    # Example: "استان تهران شهر تهران" or "استان تهران شهر شهریار"
                    prov_match = re.search(r'استان\s+(\S+)', prov_city_text)
                    city_match = re.search(r'شهر\s+(\S+)', prov_city_text)
                    row_province = prov_match.group(1) if prov_match else ""
                    row_city = city_match.group(1) if city_match else ""

                    # Filter
                    if row_province == target_province and row_city == target_city:
                        key = (pharm_name, pharm_url)
                        if key not in pharmacy_dict:
                            pharmacy_dict[key] = {
                                "pharmacy_name": pharm_name,
                                "pharmacy_url": pharm_url,
                                "brands": [],
                                "province": row_province,
                                "city": row_city
                            }
                        pharmacy_dict[key]["brands"].append(brand_name)

                if not pharmacy_dict:
                    logger.info("      ℹ️  no pharmacies in area for med=%r", med_name)
                    all_results.append({"medication": med_name, "pharmacies": [],
                                        "message": "No pharmacies found in the specified area"})
                    continue

                logger.debug("      ↳ %d unique pharmacies matched for med=%r", len(pharmacy_dict), med_name)
                # Step 4: For each unique pharmacy, fetch details (address, phone, availability)
                pharmacies_list = []
                for (name, url), info in pharmacy_dict.items():
                    try:
                        resp3 = requests.get(url, headers=headers, timeout=100)
                        resp3.raise_for_status()
                        soup3 = BeautifulSoup(resp3.text, 'html.parser')

                        # Address and phone
                        address = ""
                        phone = ""
                        alert_div = soup3.find('div', class_='alert-success')
                        if alert_div:
                            addr_elem = alert_div.find('h2', id='h2Address')
                            if addr_elem:
                                address = addr_elem.get_text(strip=True)
                            phone_elem = alert_div.find('h3', id='h3Phone')
                            if phone_elem and phone_elem.a:
                                phone = phone_elem.a.get_text(strip=True)

                        # Availability
                        availability = ""
                        avail_tr = soup3.find('tr', class_='alert-success')
                        if avail_tr:
                            avail_label = avail_tr.find('label')
                            if avail_label:
                                availability = avail_label.get_text(strip=True)

                        pharmacies_list.append({
                            "pharmacy_name": name,
                            "pharmacy_url": url,
                            "brands_available": info["brands"],
                            "address": address,
                            "phone": phone,
                            "availability": availability
                        })
                    except Exception as e:
                        pharmacies_list.append({
                            "pharmacy_name": name,
                            "pharmacy_url": url,
                            "brands_available": info["brands"],
                            "address": "",
                            "phone": "",
                            "availability": "",
                            "error": f"Failed to fetch details: {str(e)}"
                        })

                logger.info("      ✅ med=%r → %d pharmacy detail record(s)", med_name, len(pharmacies_list))
                all_results.append({
                    "medication": med_name,
                    "pharmacies": pharmacies_list
                })

            except Exception as e:
                logger.warning("      ⚠️  processing failed for med=%r: %s", med_name, e)
                all_results.append({
                    "medication": med_name,
                    "pharmacies": [],
                    "error": f"Processing failed: {str(e)}"
                })

        logger.info("   ✅ drugs_finder_tool done | %d medication result(s)", len(all_results))
        return all_results

    def nobat_dot_ir_scrapper(self, province, neighborhood, speciality):
        """
        Scrapes the top 3 doctors from nobat.ir based on:
        - province name (e.g., "تهران")
        - neighborhood name (e.g., "آذری")
        - specialty name (e.g., "cardiovascular" or "قلب و عروق")
        """
        logger.info("🕷️  nobat_dot_ir_scrapper() | province=%r | neighborhood=%r | speciality=%r",
                    province, neighborhood, speciality)
        # 1. Constants (unchanged)
        nobat_dot_ir_constants = {
            "cities": {"city-1": "تهران"},
            "specialties": {"c-5": "cardiovascular"},
            "neighborhoods": {
                "p-1005": "نزدیک به من", "p-1000": "شمال تهران", "p-1001": "شرق تهران",
                "p-1002": "جنوب تهران", "p-1003": "غرب تهران", "p-1004": "مرکز تهران",
                "p-292": "آجودانیه", "p-293": "آذربایجان", "p-294": "آذری",
                "p-295": "آرارات", "p-296": "آرژانتین", "p-298": "آهنگ",
                "p-299": "ابراهیم‌آباد", "p-300": "ابوذر", "p-301": "احتشامیه",
                "p-302": "اختیاریه", "p-303": "ارامنه", "p-304": "ارم",
                "p-305": "ازگل", "p-306": "استاد معین", "p-308": "اسفندیاری",
                "p-309": "اسکندری", "p-310": "اسلام‌‌آباد", "p-311": "افسریه",
                "p-312": "اقدسیه", "p-313": "اکباتان", "p-314": "المهدی",
                "p-315": "الهیه", "p-316": "امام حسین(ع)", "p-317": "امامزاده قاسم",
                "p-318": "امانیه", "p-319": "امیر بهادر", "p-320": "امیرآباد",
                "p-321": "امیریه", "p-322": "اندیشه", "p-323": "اوقاف",
                "p-324": "اوین", "p-325": "ایرانشهر", "p-326": "بازار",
                "p-327": "باغ آذری", "p-328": "باغ رضوان", "p-329": "باقرخان",
                "p-330": "بریانک", "p-331": "بلوار کشاورز", "p-332": "بوستان ولایت",
                "p-334": "بهاران", "p-336": "بهجت‌آباد", "p-337": "بهمن‌ یار",
                "p-338": "پارک آزادگان", "p-339": "پارک شهر", "p-340": "پارک لاله",
                "p-341": "پاسداران", "p-343": "پامنار", "p-345": "پردیسان",
                "p-346": "پرند", "p-347": "پل رومی", "p-348": "پونک",
                "p-349": "پیروزی", "p-350": "پیکان شهر", "p-351": "تجریش",
                "p-352": "توحید", "p-353": "تهران‌سر", "p-354": "تهران‌نو",
                "p-355": "تهران‌ویلا", "p-357": "تهرانپارس", "p-358": "جردن",
                "p-359": "جماران", "p-360": "جمال‌زاده", "p-361": "جمشیدیه",
                "p-362": "جمهوری", "p-366": "جوادیه", "p-367": "جوانمرد قصاب",
                "p-368": "جی", "p-369": "جیحون", "p-370": "چهارصد دستگاه",
                "p-371": "چیتگر", "p-372": "چیتگر شمالی", "p-373": "چیذر",
                "p-375": "حسن‌آباد باقرفر", "p-376": "حسن‌آباد شمالی", "p-377": "حشمتیه",
                "p-378": "حکیمیه", "p-379": "خانی‌آباد", "p-380": "خزانه",
                "p-381": "خلیج فارس", "p-382": "خورین", "p-383": "دارآباد",
                "p-384": "دانشگاه تهران", "p-385": "دبستان", "p-386": "دربند",
                "p-387": "دردشت", "p-388": "درکه", "p-389": "دروازه شمیران",
                "p-390": "دروس", "p-391": "دریان‌نو‍", "p-392": "دزاشیب",
                "p-393": "دولاب", "p-394": "دولت‌آباد", "p-395": "ده‌ونک",
                "p-396": "دهکده المپیک", "p-397": "دیلمان جنوبی", "p-398": "رباط کریم",
                "p-400": "زرگنده", "p-401": "زعفرانیه", "p-402": "زهتابی",
                "p-403": "زینبیه", "p-404": "سازمان آب", "p-405": "سازمان برنامه",
                "p-406": "ستارخان", "p-407": "سرسبز", "p-408": "سعادت‌آباد",
                "p-409": "سعدآباد", "p-410": "سلسبیل", "p-411": "سلیمانیه",
                "p-412": "سنگلج", "p-413": "سوهانک", "p-414": "سهروردی جنوبی",
                "p-415": "سهروردی شمالی", "p-416": "سید خندان", "p-417": "شاپور",
                "p-418": "شادآباد", "p-419": "شادمان", "p-420": "شاهین",
                "p-421": "شریف‌آباد", "p-422": "شکوفه", "p-423": "شمس‌آباد",
                "p-424": "شمیران‌نو", "p-425": "شوش", "p-427": "شهر زیبا",
                "p-429": "شهرآرا", "p-432": "شهرک آپادانا", "p-433": "شهرک آزادی",
                "p-434": "شهرک آسمان", "p-435": "شهرک ابوذر", "p-436": "شهرک ارغوان",
                "p-437": "شهرک استقلال", "p-438": "شهرک امام خمینی", "p-439": "شهرک امید",
                "p-440": "شهرک امیرالمومنین", "p-441": "شهرک انصار", "p-442": "شهرک بخارایی",
                "p-443": "شهرک بعثت", "p-444": "شهرک پارس", "p-445": "شهرک پاسداران",
                "p-446": "شهرک پرواز", "p-447": "شهرک تختی", "p-448": "شهرک چشمه",
                "p-449": "شهرک دانشگاه", "p-450": "شهرک دریا", "p-451": "شهرک رضوان",
                "p-452": "شهرک رضویه", "p-453": "شهرک ژاندارمری", "p-454": "شهرک شاهین",
                "p-455": "شهرک شریعتی", "p-456": "شهرک شریفی", "p-457": "شهرک شهربانی",
                "p-458": "شهرک شهرداری", "p-459": "شهرک شهید مفتح", "p-460": "شهرک طالقانی",
                "p-461": "شهرک عباسپور", "p-462": "شهرک عباسی", "p-463": "شهرک غرب",
                "p-464": "شهرک غزالی", "p-465": "شهرک فاطمه الزهرا", "p-467": "شهرک فرهنگیان",
                "p-468": "شهرک فرهنگیان غرب", "p-469": "شهرک قدس", "p-470": "شهرک کاظمیه",
                "p-471": "شهرک کمالی", "p-472": "شهرک کوهسار", "p-473": "شهرک کیانشهر",
                "p-474": "شهرک گلستان", "p-475": "شهرک گلستان غربی", "p-476": "شهرک گلها",
                "p-477": "شهرک محلاتی", "p-478": "شهرک مسلمین", "p-479": "شهرک نفت",
                "p-480": "شهرک والفجر", "p-481": "شهرک ولیعصر", "p-482": "شهرک ویلاشهر",
                "p-483": "شهرک هجرت", "p-485": "شیان", "p-486": "شیخ الرئیس",
                "p-487": "شیخ هادی", "p-488": "شیرازی", "p-489": "صادقیه",
                "p-490": "صد دستگاه", "p-491": "ضرابخانه", "p-492": "طالقانی",
                "p-493": "طرشت", "p-494": "ظفر", "p-495": "ظهیرآباد",
                "p-496": "عباس‌آباد", "p-497": "عبدل‌آباد", "p-498": "عشرت‌آباد",
                "p-499": "علم و صنعت", "p-500": "علی‌آباد جنوبی", "p-501": "علی‌آباد شمالی",
                "p-502": "فاطمی", "p-503": "فدک", "p-504": "فرح‌آباد",
                "p-505": "فرحزاد", "p-506": "فرمانیه", "p-507": "فرودگاه مهر‌آباد",
                "p-508": "فشم", "p-509": "فلاح", "p-512": "قزل قلعه",
                "p-513": "قصر", "p-514": "قصر فیروزه ۱", "p-515": "قصر فیروزه ۲",
                "p-517": "قلهک", "p-518": "قنات‌کوثر", "p-519": "قیامدشت",
                "p-520": "قیطریه", "p-521": "کاروان", "p-522": "کاشانک",
                "p-523": "کاظم‌آباد", "p-525": "کامرانیه", "p-526": "کتابخانه ملی ایران",
                "p-527": "کریم‌آباد", "p-528": "کن", "p-529": "کوهک",
                "p-530": "کوی اسلام", "p-531": "کوی بیمه", "p-532": "کوی دانشگاه",
                "p-533": "کوی فردوس", "p-534": "کوی مهرآباد", "p-535": "کوی هفدهم شهریور",
                "p-536": "گاندی", "p-537": "گلاب دره", "p-538": "گمرک",
                "p-539": "گیشا (کوی نصر)", "p-540": "لواسان", "p-541": "لویزان",
                "p-542": "مبارک‌آباد", "p-543": "مبارک‌آباد بهشتی", "p-544": "مجموعه ورزشگاه آزادی",
                "p-545": "مجید‌آباد", "p-546": "مجیدیه", "p-547": "محمودیه",
                "p-548": "مرادآباد", "p-549": "مرزداران", "p-550": "مسعودیه",
                "p-551": "مشیریه", "p-553": "ملک‌آباد", "p-554": "منصوریه",
                "p-555": "منظریه", "p-556": "منیریه", "p-557": "مولوی",
                "p-558": "مهرآباد جنوبی", "p-559": "مهران", "p-560": "میدان آزادی",
                "p-561": "میدان انقلاب", "p-562": "میدان حر", "p-563": "میدان ولیعصر",
                "p-564": "میرداماد", "p-565": "نارمک", "p-566": "نازی‌آباد",
                "p-567": "نظام‌آباد", "p-568": "نعمت‌آباد", "p-569": "نواب",
                "p-570": "نیاوران", "p-571": "نیرو هوایی", "p-572": "وحیدیه",
                "p-574": "وردآورد", "p-575": "وصفنارد", "p-576": "ولنجک",
                "p-577": "ونک", "p-578": "هاشم‌آباد", "p-579": "هفت‌ حوض",
                "p-580": "یاخچی‌آباد", "p-581": "یافت‌آباد", "p-582": "یوسف‌آباد",
                "p-583": "شهر ری", "p-584": "سنائی", "p-586": "باغ فیض",
                "p-587": "بلوار فردوس", "p-588": "هفت تیر", "p-590": "جنت آباد",
                "p-591": "شهران", "p-592": "شریعتی", "p-593": "هروی",
                "p-595": "رسالت", "p-596": "ملاصدرا"
            }
        }

        # 2. Map the names to codes
        city_code = None
        for code, name in nobat_dot_ir_constants["cities"].items():
            if name == province:
                city_code = code
                break

        neighborhood_code = None
        for code, name in nobat_dot_ir_constants["neighborhoods"].items():
            if name == neighborhood:
                neighborhood_code = code
                break

        speciality_code = None
        for code, name in nobat_dot_ir_constants["specialties"].items():
            if name == speciality:
                speciality_code = code
                break

        # 3. Validate codes
        missing = []
        if not city_code: missing.append("city")
        if not neighborhood_code: missing.append("neighborhood")
        if not speciality_code: missing.append("speciality")

        logger.debug("   ↳ mapped codes | city=%s | neighborhood=%s | speciality=%s",
                     city_code, neighborhood_code, speciality_code)
        if missing:
            logger.warning("   ❌ could not map inputs to codes: %s", missing)
            return {
                "error": f"Could not map the following inputs to codes: {', '.join(missing)}. Please check your constants."}

        # 4. Build the search URL
        search_url = f"https://nobat.ir/find/{city_code}/{neighborhood_code}/{speciality_code}/"
        logger.info("   🌐 nobat.ir search URL: %s", search_url)

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        try:
            # 5. Fetch search results page
            response = requests.get(search_url, headers=headers, timeout=100)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # 6. Find the top 3 doctor cards
            doctor_cards = soup.select('a.doctor-ui')[:3]
            logger.info("   📋 found %d doctor card(s) on nobat.ir", len(doctor_cards))
            if not doctor_cards:
                logger.warning("   ❌ no doctors found for this query")
                return {"error": "No doctors found for this search query."}

            doctors_list = []

            for card in doctor_cards:
                try:
                    doctor_page_href = card.get('href')
                    logger.debug("      ↳ fetching doctor profile: %s", doctor_page_href)
                    if not doctor_page_href.startswith('http'):
                        doctor_page_url = 'https://nobat.ir' + doctor_page_href
                    else:
                        doctor_page_url = doctor_page_href

                    # 7. Fetch the Doctor's Profile Page
                    profile_response = requests.get(doctor_page_url, headers=headers, timeout=100)
                    profile_response.raise_for_status()
                    profile_soup = BeautifulSoup(profile_response.text, 'html.parser')

                    # 8. Extract Information
                    # A. Photo, Name, Specialities, Medical Code
                    photo_tag = profile_soup.select_one('div.doctor-ui-profile img')
                    photo_url = photo_tag['src'] if photo_tag else None

                    name_tag = profile_soup.select_one('h1.doctor-ui-name span')
                    name = name_tag.get_text(strip=True) if name_tag else None

                    specialty_tags = profile_soup.select('h2.doctor-ui-specialty, span.doctor-ui-specialty')
                    specialties = list(set([s.get_text(strip=True) for s in specialty_tags]))

                    medical_code = None
                    code_div = profile_soup.select_one('div.doctor-code')
                    if code_div:
                        code_text = code_div.get_text(strip=True)
                        code_match = re.search(r'نظام:\s*(\d+)', code_text)
                        if code_match:
                            medical_code = code_match.group(1)

                    # B. Reservation Data
                    office_ids = []
                    first_office = profile_soup.select_one('div.office')
                    if first_office:
                        office_ids.append(first_office.get('data-officeid'))

                    # C. Availability Days
                    availability_days = []
                    day_elements = profile_soup.select('div.day')
                    if day_elements:
                        for day in day_elements:
                            desc = day.select_one('div.day-desc')
                            if desc:
                                availability_days.append(desc.get_text(strip=True))

                    # D. Comments (Count and Top 10)
                    comment_count = None
                    count_span = profile_soup.select_one('div.comments-summary span')
                    if count_span:
                        comment_count = count_span.get_text(strip=True)

                    top_comments = []
                    comment_containers = profile_soup.select('div.comments div.comment')
                    for i, comment in enumerate(comment_containers[:10]):
                        content_p = comment.select_one('p.comment-content')
                        content = content_p.get_text(strip=True) if content_p else ""

                        date_span = comment.select_one('span.comment-date')
                        date = date_span.get_text(strip=True) if date_span else ""

                        stars_val_style = comment.select_one('div.stars-value')
                        stars = 0.0
                        if stars_val_style:
                            style_attr = stars_val_style.get('style', '')
                            width_match = re.search(r'width:\s*(\d+)%', style_attr)
                            if width_match:
                                stars = float(width_match.group(1)) / 20.0

                        top_comments.append({
                            "content": content,
                            "date": date,
                            "stars": stars
                        })

                    # E. Biography
                    biography = ""
                    bio_div = profile_soup.select_one('div.doctor-bio-content')
                    if bio_div:
                        bio_parts = [p.get_text(strip=True) for p in
                                     bio_div.find_all(['p', 'div', 'span'], recursive=True) if
                                     p.get_text(strip=True)]
                        biography = "\n".join(bio_parts)

                    # F. Office Information
                    offices_info = []
                    all_offices = profile_soup.select('div.office')
                    for off in all_offices:
                        address_div = off.select_one('div.office-address')
                        address = address_div.get_text(strip=True) if address_div else None

                        description_div = off.select_one('div.office-description')
                        description = description_div.get_text(strip=True) if description_div else None

                        phone_divisions = off.select('div.office-phone')
                        phones = [p.get_text(strip=True) for p in phone_divisions if p.get_text(strip=True)]

                        offices_info.append({
                            "address": address,
                            "description": description,
                            "phones": phones
                        })

                    reservation_link = doctor_page_url.replace('nobat.ir', 'turn.nobat.ir')

                    # Build doctor info dict
                    doctor_info = {
                        "source": "nobat.ir",
                        "profile_url": doctor_page_url,
                        "reservation_link": reservation_link,
                        "name": name,
                        "photo_url": photo_url,
                        "specialties": specialties,
                        "medical_license_code": medical_code,
                        "reservation_data": {"office_ids": office_ids},
                        "availability": availability_days,
                        "comments": {
                            "total_count": comment_count,
                            "top_10": top_comments
                        },
                        "biography": biography,
                        "offices": offices_info
                    }
                    doctors_list.append(doctor_info)
                    logger.info("      ✅ scraped doctor: name=%r | code=%r | offices=%d",
                                name, medical_code, len(offices_info))

                except Exception as e:
                    # Skip this doctor but continue with others
                    logger.warning("      ⚠️  skipping a doctor card due to error: %s", e)
                    continue

            # Return list of doctors (up to 3)
            logger.info("   ✅ nobat_dot_ir_scrapper done | %d doctor(s)", len(doctors_list))
            return doctors_list

        except Exception as e:
            logger.exception("   🔥 nobat.ir scraping failed: %s", e)
            return {"error": f"Scraping Failed: {str(e)}"}

    def doctor_doctor_scrapper(self, province, neighborhood, speciality):
        pass

    def doctoreto_scrapper(self, province, neighborhood, speciality, insurance=None):
        """
        Scrapes the top 3 doctors from doctoreto.com.
        Extracts all requested fields except phones and first available appointment.
        Comments now correctly fetch content, recommendation, and wait time.
        """
        logger.info("🕷️  doctoreto_scrapper() | province=%r | neighborhood=%r | speciality=%r | insurance=%r",
                    province, neighborhood, speciality, insurance)

        # Mappings (extend as needed)
        DOCTORETO_REGION_MAPPING = {
            "آبشار": "abshar", "ابوذر": "abozar", "ابوذر (بسیج)": "abozar-basij",
            "اختیاریه": "ekhtariyeh", "آذری": "azari", "آرارات": "ararat",
            "ارامنه": "aramineh", "امیرآباد": "amirabad", "پاسداران": "pasdaran",
            "سعادت آباد": "saadatabad", "ونک": "vanak", "آهنگ": "ahang",
            "تهرانپارس": "tehranpars",
        }
        DOCTORETO_CITY_MAPPING = {
            "تهران": "tehran", "اصفهان": "isfahan", "شیراز": "shiraz",
            "مشهد": "mashhad", "تبریز": "tabriz",
        }
        DOCTORETO_SPECIALITY_MAPPING = {
            "قلب و عروق": "cardiologist", "داخلی": "internist",
            "پزشک عمومی": "general-practitioner",
            "cardiovascular": "cardiologist",
        }

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # ---------- 1. Map inputs to URL slugs ----------
        province_slug = DOCTORETO_CITY_MAPPING.get(province, province.lower().replace(' ', '-'))
        neighborhood_slug = DOCTORETO_REGION_MAPPING.get(neighborhood, neighborhood.lower().replace(' ', '-'))
        speciality_slug = DOCTORETO_SPECIALITY_MAPPING.get(speciality, speciality.lower().replace(' ', '-'))
        logger.debug("   ↳ slugs | speciality=%r | city=%r | region=%r",
                     speciality_slug, province_slug, neighborhood_slug)
        if not all([province_slug, neighborhood_slug, speciality_slug]):
            logger.warning("   ❌ could not map inputs to doctoreto slugs")
            return {"error": "Could not map one of province, neighborhood, or speciality to a valid Doctoreto slug."}

        search_url = f"https://doctoreto.com/doctors/speciality/{speciality_slug}/city/{province_slug}/region/{neighborhood_slug}"
        logger.info("   🌐 doctoreto search URL: %s", search_url)
        try:
            # ---------- 2. Fetch search results page ----------
            response = requests.get(search_url, headers=headers, timeout=100)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the top 3 doctor articles
            cards_container = soup.select_one('div.flex.flex-col.gap-3.px-\\[15px\\].lg\\:gap-4.lg\\:px-0')
            articles = []
            if cards_container:
                articles = cards_container.select('article')[:3]
            else:
                articles = soup.select('article.flex.cursor-pointer.flex-col')[:3]

            logger.info("   📋 found %d doctor article(s) on doctoreto.com", len(articles))
            if not articles:
                logger.warning("   ❌ no doctors found on doctoreto.com")
                return {"error": "No doctors found on doctoreto.com for this search."}

            doctors_list = []

            for article in articles:
                try:
                    # Profile link
                    profile_link_a = article.select_one('a.sc-fc0eedc0-3.xkJsg') or article.select_one(
                        'a[href^="/doctor/"]')
                    if not profile_link_a:
                        continue
                    href = profile_link_a.get('href')
                    profile_page_url = 'https://doctoreto.com' + href if href.startswith('/') else href
                    logger.debug("      ↳ fetching doctoreto profile: %s", profile_page_url)

                    # ---------- 3. Fetch profile page ----------
                    profile_response = requests.get(profile_page_url, headers=headers, timeout=100)
                    profile_response.raise_for_status()
                    profile_soup = BeautifulSoup(profile_response.text, 'html.parser')

                    # ---------- 4. Extract all fields ----------
                    # (A) Photo
                    photo_tag = profile_soup.select_one('div.sc-732d5513-1 img') or profile_soup.select_one(
                        'div.sc-732d5513-0 img')
                    photo_url = None
                    if photo_tag and photo_tag.get('src'):
                        src = photo_tag['src']
                        photo_url = 'https://doctoreto.com' + src if src.startswith('/') else src

                    # (B) Name
                    name_tag = profile_soup.select_one('div.sc-212a8fa3-4.qvYGb')
                    name = name_tag.get_text(strip=True) if name_tag else None

                    # (C) Specialties
                    specialties = []
                    header_spec = profile_soup.select_one('div.sc-212a8fa3-7 div.mb-3.text-12.font-medium')
                    if header_spec:
                        specialties.append(header_spec.get_text(strip=True))
                    specialty_section = profile_soup.select_one('div.sc-910b97e5-0')
                    if specialty_section:
                        for item in specialty_section.select('div.sc-50a608cd-1 div.sc-910b97e5-2'):
                            s = item.get_text(strip=True)
                            if s and s not in specialties:
                                specialties.append(s)

                    # (D) Medical license code
                    medical_code = None
                    license_div = profile_soup.select_one('div.sc-212a8fa3-12.jPJuqw')
                    if license_div:
                        match = re.search(r'کد نظام پزشکی\s*:\s*(\d+)', license_div.get_text(strip=True))
                        if match:
                            medical_code = match.group(1)

                    # (E) Rating stats
                    rating = None
                    rating_count = None
                    recommendation_rate = None
                    successful_appointments = None
                    average_wait_time = None
                    review_breakdown = {}

                    stats_bar = profile_soup.select_one('div.sc-8fc57d2d-0')
                    if stats_bar:
                        first_stat = stats_bar.select_one('div.sc-8fc57d2d-3')
                        if first_stat:
                            rating_text = first_stat.get_text(strip=True)
                            rating_match = re.search(r'^([\d.]+)', rating_text)
                            if rating_match:
                                rating = rating_match.group(1)
                            count_match = re.search(r'\((\d+)\s*نظر\)', rating_text)
                            if count_match:
                                rating_count = count_match.group(1)
                        rec_block = stats_bar.select_one('div.sc-8fc57d2d-3:nth-of-type(2)')
                        if rec_block:
                            rec_text = rec_block.get_text(strip=True)
                            rec_match = re.search(r'(\d+)%', rec_text) or re.search(r'(\d+)\s*٪', rec_text)
                            if rec_match:
                                recommendation_rate = rec_match.group(1) + '%'
                        succ_block = stats_bar.select('div.sc-8fc57d2d-3')[-1] if len(
                            stats_bar.select('div.sc-8fc57d2d-3')) >= 3 else None
                        if succ_block:
                            succ_text = succ_block.get_text(strip=True)
                            succ_match = re.search(r'(\d+)\s*نوبت موفق', succ_text)
                            if succ_match:
                                successful_appointments = succ_match.group(1)

                    rating_container = profile_soup.select_one('div.sc-28976a9b-0')
                    if rating_container:
                        for item in rating_container.select('div.mb-5'):
                            label_span = item.select_one('span')
                            score_span = item.select_one('span.flex.items-center')
                            if label_span and score_span:
                                label = label_span.get_text(strip=True).rstrip(':')
                                score = score_span.get_text(strip=True).split()[0]
                                review_breakdown[label] = score
                        wait_block = rating_container.select_one('div.sc-28976a9b-2:last-child .sc-28976a9b-3')
                        if wait_block:
                            wait_text = wait_block.get_text(strip=True)
                            if 'دقیقه' in wait_text:
                                average_wait_time = wait_text

                    # (F) Services
                    services = []
                    services_section = profile_soup.select_one('div.sc-a595fbf3-0')
                    if services_section:
                        for span in services_section.select('span.sc-18058bc8-0.sc-a595fbf3-2'):
                            t = span.get_text(strip=True)
                            if t:
                                services.append(t)

                    # (G) Offices (only name and address, no phones or appointments)
                    offices = []
                    office_section = profile_soup.select_one('div.sc-8025bd4f-0') or profile_soup.select_one(
                        'div.sc-4e6b0e39-0')
                    if office_section:
                        office_name_tag = office_section.select_one('div.sc-8025bd4f-2') or office_section.select_one(
                            'div.sc-4e6b0e39-1')
                        office_name = office_name_tag.get_text(strip=True) if office_name_tag else None

                        address = None
                        addr_line = office_section.select_one('div.sc-8025bd4f-4') or office_section.select_one(
                            'div.sc-4e6b0e39-2')
                        if addr_line:
                            full_addr = addr_line.get_text(strip=True)
                            if 'آدرس:' in full_addr:
                                address = full_addr.split('آدرس:')[-1].strip()
                            else:
                                addr_span = addr_line.select_one('span:not(.sc-8025bd4f-7):not(.sc-8025bd4f-8)')
                                if addr_span:
                                    address = addr_span.get_text(strip=True)

                        offices.append({
                            "name": office_name,
                            "address": address,
                            "phones": [],
                            "first_available_appointment": None
                        })

                    # (H) Top-10 Comments
                    top_comments = []
                    comment_containers = profile_soup.select('div.sc-49be1e1-1.byVpll')
                    for i, container in enumerate(comment_containers[:10]):
                        user_elem = container.select_one('div.sc-49be1e1-5')
                        user_name = ""
                        if user_elem:
                            full_user_text = user_elem.get_text(strip=True)
                            user_name = full_user_text.split('(')[0].strip()

                        date_elem = container.select_one('span.sc-49be1e1-3')
                        date = ""
                        if date_elem:
                            date = date_elem.get_text(strip=True).strip('()')

                        stars = 0
                        star_container = container.select_one('div.sc-cf98647b-1')
                        if star_container:
                            stars = len(star_container.select('svg'))

                        content_elem = container.select_one('div.sc-49be1e1-9.jYJAaF')
                        content = content_elem.get_text(strip=True) if content_elem else ""

                        rec_elem = container.select_one('div.sc-49be1e1-8.loXbFH')
                        rec_text = rec_elem.get_text(strip=True) if rec_elem else ""

                        wait_elem = container.select_one('div.sc-49be1e1-8.iRTwKL')
                        wait_time = ""
                        if wait_elem:
                            txt = wait_elem.get_text(strip=True)
                            if 'زمان انتظار' in txt:
                                wait_time = txt.split('زمان انتظار:')[-1].strip()

                        top_comments.append({
                            "user": user_name,
                            "date": date,
                            "stars": stars,
                            "content": content,
                            "recommendation": rec_text,
                            "wait_time": wait_time
                        })

                    # Build doctor info dictionary
                    doctor_info = {
                        "source": "doctoreto.com",
                        "profile_url": profile_page_url,
                        "name": name,
                        "photo_url": photo_url,
                        "specialties": specialties,
                        "medical_license_code": medical_code,
                        "rating": {
                            "overall": rating,
                            "count": rating_count,
                            "breakdown": review_breakdown
                        },
                        "recommendation_rate": recommendation_rate,
                        "successful_appointments": successful_appointments,
                        "average_wait_time": average_wait_time,
                        "services": services,
                        "offices": offices,
                        "comments": {
                            "top_10": top_comments
                        }
                    }
                    doctors_list.append(doctor_info)
                    logger.info("      ✅ scraped doctoreto doctor: name=%r | code=%r | rating=%r",
                                name, medical_code, rating)

                except Exception as e:
                    # Skip this doctor and continue with next
                    logger.warning("      ⚠️  skipping a doctoreto article due to error: %s", e)
                    continue

            # Return list of doctors (up to 3)
            logger.info("   ✅ doctoreto_scrapper done | %d doctor(s)", len(doctors_list))
            return doctors_list

        except Exception as e:
            logger.exception("   🔥 doctoreto scraping failed: %s", e)
            return {"error": f"Doctoreto Scraping Failed: {str(e)}"}


# ------------------------------
# Chat logic (same as your code)
# ------------------------------
system_prompt_finders = (
    'You are a very polite assistant that your task is to help our customers. Our customers are some Employees that based on some research by doctors, they may be in danger of some diseases. So, you have to help them to address their issues by finding appropriate doctors, medications, and drugstores. I have provided some tools named DOCTORS_FINDER_TOOL and MEDICATIONS_FINDER_TOOL for you that they help you to assist customers. You should call these tool whenever you want to find one of these for customers. for calling these tools you need to provide some required information which is mandatory for using those tools. Our customers are Persian; therefore, all conversations should be in Persian. You can start with a great greetings, yourself introduction, and how you can assist the customers. Be to the point and answer the user very shortly. Try to use the users first name in your conversation. Related to the conversation, go step by step exactly like this, hello Im Your intelligent assistant. I hope you are doing well. The result above shows that you may in danger of X(X is a disease that may threaten the customer). I can find proper doctors and medications to address this issue. So, now tell me how I can help you. Wait till they answer. After they answer, ask them to confirm these three information, province, neighborhood. Additionally, you can tell them if they need services from other provinces or neighborhoods, we can do this also. And then, call the proper tool. After you have done one of these duties, you can ask the customer that they need other services. After calling tools you will get the result of tools just for notice. dont write them and just say you can see the proper and closest doctors or medications down below. because I will show the result of tools by HTML CSS separately. However, if they have any question from the results provide adequate responses. THESE ARE THE RESULTS of DOCTORS PREDICTION, THE ACCURACY OF PREDICTIONS, THE SPECIALITY THAT THE CUSTOMERS NEED TO REFER TO, AND A LIST OF MEDICATIONS THEY NEED TO USE TO ADDRESS THEIR DISEASE .\n {disease_results}(when the customer asks you to find me medications, use the disease_results and medications, and you dont need to ask him for acceptance and also questioning).\n HERE IS THE PERSONAL INFORMATION OF THE CUSTOMER. \n {personal_information}.'
)
# Include disease_results and personal_information (hardcoded for test)
disease_results = [
    {
        'disease': 'hypertension',
        'accuracy': '0.91',
        'refer_to': 'cardiovascular',
        'medications': ['Lisinopril', 'Enalapril', 'Losartan']
    },
]

personal_information = [{
    'name': 'امیر',
    'last_name': 'حسین پور کلسری',
    'living_province': 'تهران',
    'neighborhood': 'ابوذر',
}]

OPENAI_CLIENT = None  # Will be set in view


def get_openai_client():
    """Initialize if not already done (using settings or env)."""
    global OPENAI_CLIENT
    if OPENAI_CLIENT is None:
        logger.info("🔌 initializing OpenAI client | base_url=%s", gapgpt_base_url)
        import openai
        from django.conf import settings

        OPENAI_CLIENT = openai.OpenAI(
            base_url=gapgpt_base_url,
            api_key=gapgpt_api_key,
        )
        logger.info("   ✅ OpenAI client initialized")
    else:
        logger.debug("🔌 reusing cached OpenAI client")
    return OPENAI_CLIENT


doctors_finder_tool_prompt = {
    "type": "function",
    "function": {
        "name": "doctors_finder_tool",
        "description": "Find doctors for the user based on province, neighborhood, and speciality. After calling this tool you will get the result of it just for notice. dont write them and just say you can see the proper and closest doctors down below.",
        "parameters": {
            "type": "object",
            "properties": {
                "province": {"type": "string", "description": "Living province name in Persian"},
                "neighborhood": {"type": "string", "description": "Living neighborhood name in Persian"},
                "speciality": {"type": "string",
                               "description": "Speciality in English, e.g., 'cardiovascular' in English."},
            },
            "required": ["province", "neighborhood", "speciality"],
            "additionalProperties": False
        }
    }
}
medications_finder_tool_prompt = {
    "type": "function",
    "function": {
        "name": "medications_finder_tool",
        "description": "Find medications for the user based on given medications name. After calling this tool you will get the result of it just for notice. dont write them and just say you can see the proper medications down below.",
        "parameters": {
            "type": "object",
            "properties": {
                "medications": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "a list of medications related to the customer disease."
                },
            },
            "required": ["medications"],
            "additionalProperties": False
        }
    }
}

tools = [
    {"type": "function", "function": doctors_finder_tool_prompt["function"]},
    {"type": "function", "function": medications_finder_tool_prompt["function"]}
]


def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    function_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    logger.info("🛠️  handle_tool_call() | function=%r | args=%r", function_name, args)
    finder = FINDERS()

    # متغیرهای خروجی پیش‌فرض
    finder_results = None
    tool_content = ""

    # سناریو اول: فراخوانی ابزار پزشکان
    if function_name == "doctors_finder_tool":
        finder_results = finder.doctors_finder_tool(
            province=args['province'],
            neighborhood=args['neighborhood'],
            speciality=args['speciality'],
        )
        if isinstance(finder_results, list):
            summary = []
            for doc in finder_results:
                if 'error' not in doc:
                    summary.append({
                        "name": doc.get("name"),
                        "reservation_link": doc.get("reservation_link")
                    })
            tool_content = json.dumps(summary, ensure_ascii=False)
        else:
            tool_content = json.dumps(finder_results, ensure_ascii=False)

    # سناریو دوم: فراخوانی ابزار داروها
    elif function_name == "medications_finder_tool":
        finder_results = finder.medications_finder_tool(
            medications_list=args['medications']
        )
        if isinstance(finder_results, list):
            summary = []
            for drug in finder_results:
                if 'error' not in drug:
                    # طبق دستور شما، فقط نام دارو و نام داروخانه ذخیره می‌شود
                    summary.append({
                        "drug_name": drug.get("drug_name"),
                        "pharmacy_name": drug.get("pharmacy_name")
                    })
            tool_content = json.dumps(summary, ensure_ascii=False)
        else:
            tool_content = json.dumps(finder_results, ensure_ascii=False)

    logger.info("🛠️  handle_tool_call result | function=%r | summary_content=%s",
                function_name, tool_content)
    # ساختار پاسخ استاندارد برای ارسال به API جهت اطلاع مدل
    response = {
        "role": "tool",
        "content": tool_content,
        "tool_call_id": tool_call.id
    }
    return response, finder_results


def chat_with_assistant(message, history):
    logger.info("💬 chat_with_assistant() | user_message_len=%d | history_turns=%d",
                len(message or ""), len(history or []))
    client = get_openai_client()
    system_content = system_prompt_finders.format(
        disease_results=disease_results,
        personal_information=personal_information
    )
    messages = [{"role": "system", "content": system_content}] + history + [{"role": "user", "content": message}]

    logger.info("   📤 sending chat completion request | model=%s | messages=%d | tools=%d",
                gapgpt_model, len(messages), len(tools))
    response = client.chat.completions.create(
        model=gapgpt_model,
        messages=messages,
        tools=tools
    )
    choice = response.choices[0]
    logger.info("   📥 first response | finish_reason=%s", choice.finish_reason)

    # هندل کردن اجرای ابزارها در صورت درخواست مدل
    if choice.finish_reason == "tool_calls":
        msg = choice.message
        logger.info("   🔧 model requested tool call → executing")

        # اجرای ابزار مناسب با توجه به شرط نوشته شده در متد قبلی
        tool_response, finder_results = handle_tool_call(msg)

        # اضافه کردن پیام درخواست ابزار مدل و سپس پاسخ ابزار به آرایه سابقه پیام‌ها
        messages.append(msg)
        messages.append(tool_response)

        # دریافت پاسخ نهایی متنی از مدل بر اساس دیتای خلاصه شده ابزار
        logger.info("   📤 sending follow-up completion (with tool result) | messages=%d", len(messages))
        final_response = client.chat.completions.create(
            model=gapgpt_model,
            messages=messages
        )
        final_text = final_response.choices[0].message.content
        logger.info("   ✅ final reply ready (len=%d) | finder_results=%s",
                    len(final_text or ""),
                    f"{len(finder_results)} items" if isinstance(finder_results, list) else type(finder_results).__name__)
        return final_text, finder_results
    else:
        logger.info("   ✅ direct reply (no tool) | len=%d", len(choice.message.content or ""))
        return choice.message.content, None


# ============================================================
#  DOCTOR / MANAGER FORM-FILLING ASSISTANT
#  ------------------------------------------------------------
#  Unlike the employee finder bot above, this agent does NOT
#  search for doctors/medications. Instead it acts as a clinical
#  drafting assistant: given a free-text prompt from the
#  examining physician (or manager) plus the patient's profile,
#  it drafts every field of the occupational-health examination
#  record — physical-exam notes per body system, the final
#  fitness opinion, conditions / unfit reasons, medical
#  recommendations and specialist referrals — each with a clear
#  rationale. The physician always reviews and edits before save.
# ============================================================

# Maps each draftable exam-note field (the model field name) to its Persian label.
DOCTOR_ASSIST_EXAM_FIELDS = {
    "general_exam_notes": "معاینه عمومی",
    "head_neck_exam_notes": "سر و گردن",
    "eye_exam_notes": "چشم",
    "ent_mouth_exam_notes": "گوش، حلق، بینی و دهان",
    "lung_exam_notes": "ریه",
    "cardiovascular_exam_notes": "قلب و عروق",
    "abdomen_pelvis_exam_notes": "شکم و لگن",
    "urinary_system_exam_notes": "دستگاه ادراری",
    "musculoskeletal_exam_notes": "اسکلتی-عضلانی",
    "nervous_system_exam_notes": "سیستم عصبی",
    "mental_health_exam_notes": "سلامت روان",
    "skin_hair_nails_exam_notes": "پوست، مو و ناخن",
}

doctor_assist_system_prompt = (
    "شما دستیار هوشمند و ارشد مستندسازی بالینی برای 'پزشکان متخصص طب کار' هستید. "
    "کاربر شما یک پزشک معاینه‌کننده (یا مدیر سلامت شغلی) است که در حال تکمیل فرم معاینات دوره‌ای طب کار برای یک پرونده پرسنلی است. "
    "شما نقش پیش‌نویس‌کننده پرونده پزشکی را دارید؛ پزشک تمامی فیلدهای پیشنهادی شما را مرور، ویرایش و نهایی می‌کند. "
    "\n\n"
    "ورودی‌های شما شامل موارد زیر است:\n"
    "۱. PATIENT CONTEXT: اطلاعات کامل پرونده شامل مشخصات فردی بیمار، شغل و مواجهات زیان‌آور شغلی، سوابق پزشکی و شغلی، علائم حیاتی و معاینات اولیه، نتایج کامل پاراکلینیک (اسپیرومتری، ECG، CXR)، آزمایش‌های خون/ادرار/آدیکشن، مدارک و اسناد پزشکی بارگذاری‌شده (Medical Tests با تمامی پنل‌ها و فاکتورها)، نتیجه غربالگری آنمی چشم، و گزارش کامل تحلیل ریسک هوش مصنوعی.\n"
    "۲. PHYSICIAN REQUEST: دستور یا توضیح آزاد پزشک.\n"
    "\n\n"
    "تشخیص دقیق حالت درخواست پزشک (MODE DETECTION):\n"
    "پزشک ممکن است یکی از دو حالت زیر را درخواست کند:\n"
    "حالت ۱. تکمیل کامل پرونده (FULL DRAFT MODE):\n"
    "  - زمانی که پزشک می‌گوید 'پرونده را پر کن'، 'همه فیلدها را تکمیل کن'، 'پیش‌نویس کامل بنویس' یا یک دستور عمومی برای تکمیل پرونده می‌دهد.\n"
    "  - در این حالت: is_targeted = false، requires_approval = false، لیست target_fields خالی است و تمام فیلدهای فرم پیش‌نویس می‌شوند.\n"
    "\n"
    "حالت ۲. اصلاح/تکمیل فیلد(های) خاص (TARGETED FIELD MODIFICATION MODE):\n"
    "  - زمانی که پزشک فقط درباره یک یا چند ارگان یا فیلد خاص صحبت می‌کند (مثلاً 'در سر و گردن ضایعه لمس شد اینو ثبت کن'، 'فقط فیلد ریه را اصلاح کن'، 'فشار خون در فیلد قلب ثبت شود'، 'توصیه‌ها را تغییر بده').\n"
    "  - در این حالت: is_targeted = true، requires_approval = true (مگر اینکه در پیام تایید قبلی داده شده باشد)، و اسامی فیلدهای هدف دقیقاً در لیست target_fields قرار می‌گیرند (مثلاً [\"head_neck_exam_notes\"]).\n"
    "  - در approval_prompt_text متنی شفاف به زبان فارسی بنویسید که از پزشک سوال می‌کند: 'آیا تایید می‌فرمایید که فیلد(های) [نام فیلدها به فارسی] با متن زیر بروزرسانی شود؟'.\n"
    "  - در proposed_changes متنی که فقط باید در فیلد(های) هدف قرار گیرد قرار دهید.\n"
    "\n"
    "اسامی کلیدهای فیلدها جهت ارجاع دقیق در target_fields و proposed_changes:\n"
    "- general_exam_notes: معاینه عمومی\n"
    "- head_neck_exam_notes: سر، گردن و حواس (Head, neck & senses)\n"
    "- eye_exam_notes: چشم\n"
    "- ent_mouth_exam_notes: گوش، حلق، بینی و دهان\n"
    "- lung_exam_notes: ریه و دستگاه تنفس\n"
    "- cardiovascular_exam_notes: قلب و عروق\n"
    "- abdomen_pelvis_exam_notes: شکم و لگن\n"
    "- urinary_system_exam_notes: دستگاه ادراری\n"
    "- musculoskeletal_exam_notes: اسکلتی-عضلانی\n"
    "- nervous_system_exam_notes: سیستم عصبی\n"
    "- mental_health_exam_notes: سلامت روان\n"
    "- skin_hair_nails_exam_notes: پوست، مو و ناخن\n"
    "- opinion_choice / opinion_fit_conditions_details / opinion_unfit_reason: نظریه نهایی\n"
    "- medical_recommendations: توصیه‌های پزشکی\n"
    "- referrals: ارجاعات تخصصی\n"
    "\n\n"
    "قواعد نگارش پزشکی:\n"
    "۱. لحن نگارش باید ۱۰۰٪ مانند یک پزشک متخصص طب کار در پرونده‌های رسمی معاینات دوره‌ای باشد. از هرگونه ادبیات هوش مصنوعی پرهیز کنید.\n"
    "۲. جملات هر بخش مستقیم و مختصر (۱ تا ۳ جمله) باشد.\n"
    "۳. نظریه نهایی (opinion_choice) باید دقیقاً یکی از 'fit'، 'conditional' یا 'unfit' باشد.\n"
    "۴. در صورت وجود مواجهه یا خطرات شغلی، ارجاعات تخصصی کاربردی (۱ الی ۳ ارجاع) در referrals پیشنهاد شود. هشدار بسیار مهم: اگر در PATIENT CONTEXT ارجاعاتی از قبل ثبت شده است، به هیچ عنوان ارجاعی با همان تخصص یا تخصص مشابه تولید نکنید (مثلاً اگر 'چشم' وجود دارد، 'متخصص چشم' را اضافه نکنید). فقط ارجاعات کاملاً جدید را در لیست قرار دهید.\n"
    "\n\n"
    "پاسخ شما باید منحصراً یک JSON معتبر باشد که دقیقاً کلیدهای زیر را داشته باشد:\n"
    "اگر پیام پزشک صرفاً سلام، احوالپرسی یا یک سوال عمومی است که نیازی به پر کردن یا تغییر فرم ندارد، فقط دو کلید زیر را برگردانید:\n"
    "{\n"
    '  "is_chat_only": true,\n'
    '  "chat_message": "پاسخ متنی شما به پزشک..."\n'
    "}\n\n"
    "اما اگر پزشک دستور پر کردن، تغییر، یا اصلاح فرم را داده است، باید به صورت زیر پاسخ دهید:\n"
    "{\n"
    '  "is_chat_only": false,\n'
    '  "is_targeted": bool,\n'
    '  "target_fields": [str],\n'
    '  "requires_approval": bool,\n'
    '  "approval_prompt_text": str,\n'
    '  "proposed_changes": { "field_name": "proposed text..." },\n'
    '  "general_exam_notes": str,\n'
    '  "head_neck_exam_notes": str,\n'
    '  "eye_exam_notes": str,\n'
    '  "ent_mouth_exam_notes": str,\n'
    '  "lung_exam_notes": str,\n'
    '  "cardiovascular_exam_notes": str,\n'
    '  "abdomen_pelvis_exam_notes": str,\n'
    '  "urinary_system_exam_notes": str,\n'
    '  "musculoskeletal_exam_notes": str,\n'
    '  "nervous_system_exam_notes": str,\n'
    '  "mental_health_exam_notes": str,\n'
    '  "skin_hair_nails_exam_notes": str,\n'
    '  "opinion_choice": "fit" | "conditional" | "unfit",\n'
    '  "opinion_fit_conditions_details": str,\n'
    '  "opinion_unfit_reason": str,\n'
    '  "medical_recommendations": str,\n'
    '  "referrals": [ { "specialty": str, "reason": str, "result": str, "date": str } ],\n'
    '  "explanation": str\n'
    "}"
)


def _extract_json_object(text):
    """Robustly pull the first JSON object out of an LLM response."""
    logger.debug("🧩 _extract_json_object() | text_len=%d", len(text or ""))
    if not text:
        logger.warning("   ⚠️  empty text → None")
        return None
    cleaned = text.strip()
    # strip ``` / ```json fences if present
    if cleaned.startswith("```"):
        logger.debug("   ↳ stripping code fences")
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
        logger.debug("   ✅ parsed JSON directly")
        return parsed
    except Exception:
        logger.debug("   ↳ direct json.loads failed, trying outermost { ... }")
    # fall back: grab the outermost { ... }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start:end + 1])
            logger.debug("   ✅ parsed JSON from outermost braces")
            return parsed
        except Exception:
            logger.warning("   ⚠️  could not parse JSON from braces → None")
            return None
    logger.warning("   ⚠️  no JSON object found → None")
    return None


def doctor_assist_assistant(prompt, profile_context, history=None):
    """
    Draft the occupational-health examination record for a physician/manager.

    Args:
        prompt: free-text instruction from the physician.
        profile_context: a readable text block describing the patient.
        history: optional prior [{role, content}] turns for iterative refinement.

    Returns:
        (suggestions_dict, error_str). suggestions_dict is None on failure.
    """
    logger.info("🩺 doctor_assist_assistant() | prompt_len=%d | context_len=%d | history_turns=%d",
                len(prompt or ""), len(profile_context or ""), len(history or []))
    client = get_openai_client()
    user_content = (
        f"PATIENT CONTEXT:\n{profile_context}\n\n"
        f"PHYSICIAN REQUEST:\n{prompt}\n\n"
        "Draft the full record now as the JSON object specified."
    )
    messages = (
        [{"role": "system", "content": doctor_assist_system_prompt}]
        + (history or [])
        + [{"role": "user", "content": user_content}]
    )

    # Prefer enforced JSON output; gracefully fall back if the proxy rejects it.
    try:
        logger.info("   📤 requesting completion with response_format=json_object | model=%s", gapgpt_model)
        response = client.chat.completions.create(
            model=gapgpt_model,
            messages=messages,
            response_format={"type": "json_object"},
        )
    except Exception as json_mode_err:
        logger.warning("   ⚠️  json_object mode rejected (%s), retrying without response_format", json_mode_err)
        try:
            response = client.chat.completions.create(
                model=gapgpt_model,
                messages=messages,
            )
        except Exception as e:
            logger.exception("   🔥 AI engine connection failed: %s", e)
            return None, f"خطا در ارتباط با موتور هوش مصنوعی: {e}"

    content = response.choices[0].message.content
    logger.info("   📥 model response received | content_len=%d", len(content or ""))
    data = _extract_json_object(content)
    if not isinstance(data, dict):
        logger.warning("   ❌ model response not parseable as dict")
        return None, "پاسخ مدل قابل پردازش نبود. لطفاً دوباره و با توضیح دقیق‌تر تلاش کنید."

    # Normalise / sanitise the result so the front-end can rely on it.
    suggestions = {}
    suggestions["is_targeted"] = bool(data.get("is_targeted"))
    suggestions["requires_approval"] = bool(data.get("requires_approval"))
    suggestions["target_fields"] = [str(f) for f in (data.get("target_fields") or []) if isinstance(f, str)]
    suggestions["approval_prompt_text"] = (data.get("approval_prompt_text") or "").strip()
    suggestions["proposed_changes"] = data.get("proposed_changes") if isinstance(data.get("proposed_changes"), dict) else {}

    for field in DOCTOR_ASSIST_EXAM_FIELDS:
        suggestions[field] = (data.get(field) or "").strip()

    choice = (data.get("opinion_choice") or "").strip().lower()
    if choice not in ("fit", "conditional", "unfit"):
        choice = ""
    suggestions["opinion_choice"] = choice
    suggestions["opinion_fit_conditions_details"] = (data.get("opinion_fit_conditions_details") or "").strip()
    suggestions["opinion_unfit_reason"] = (data.get("opinion_unfit_reason") or "").strip()
    suggestions["medical_recommendations"] = (data.get("medical_recommendations") or "").strip()

    referrals = []
    seen_specs = set()
    for ref in (data.get("referrals") or []):
        if not isinstance(ref, dict):
            continue
        spec = (ref.get("specialty") or "").strip()
        reas = (ref.get("reason") or "").strip()
        if not spec and not reas:
            continue
        spec_clean = spec or "متخصص طب کار"
        spec_key = spec_clean.lower()
        if spec_key in seen_specs:
            continue
        seen_specs.add(spec_key)
        referrals.append({
            "specialty": spec_clean,
            "reason": reas or "ارزیابی و معاینه تکمیلی دوره‌ای",
            "result": (ref.get("result") or "").strip() or "در انتظار ارزیابی بالینی",
            "date": (ref.get("date") or "").strip(),
        })
    suggestions["referrals"] = referrals
    suggestions["explanation"] = (data.get("explanation") or "").strip()

    logger.info("   ✅ doctor_assist_assistant done | is_targeted=%s | opinion_choice=%r | referrals=%d | recommendations_len=%d",
                suggestions.get("is_targeted"), suggestions.get("opinion_choice"), len(referrals),
                len(suggestions.get("medical_recommendations") or ""))
    return suggestions, None


# ============================================================
#  DOCTOR RESEARCH CHAT  (knowledge base + web + read-URL +
#  PubMed + medications)
#  ------------------------------------------------------------
#  A tool-calling research assistant for a DOCTOR (or MANAGER)
#  reviewing an employee's ROP / KC eye-screening case. It runs a
#  bounded tool-calling loop against GAPGPT (OpenAI-compatible)
#  and returns (answer_markdown, sources_list) where each source
#  is {id, title, url, domain} — the same shape the ROP chat and
#  the marketplace chat use for their inline [n] citation cards.
#
#  Web-search / KB retrieval helpers are reused from
#  ``single_rop.chat_service`` when importable (retrieve_local,
#  _serper_search, _scrape_url, _domain); otherwise it falls back
#  to ``doctors_marketplace.services.websearch``. Every tool is
#  crash-safe: a failure returns a short error string to the model
#  instead of raising out of the loop.
# ============================================================

DOCTOR_RESEARCH_SYSTEM_PROMPT = (
    "You are a meticulous clinical RESEARCH assistant working for a DOCTOR (or an "
    "occupational-health manager) who is reviewing an employee's automated eye-screening "
    "case — either ROP (Retinopathy of Prematurity) or KC (Keratoconus). The doctor uses "
    "you to research clinical guidelines, recent evidence, and medications so they can "
    "make and justify a decision. Your user is a licensed clinician, so precise, technical, "
    "evidence-based answers are expected and appropriate.\n\n"
    "You have these tools — use them proactively before answering any factual/clinical "
    "question, and prefer authoritative sources:\n"
    "  • search_knowledge_base(query): the clinic's local ROP/KC knowledge base.\n"
    "  • search_web(query): general web search for guidelines and current information.\n"
    "  • fetch_url(url): read the readable text of a specific web page.\n"
    "  • search_pubmed(query): search peer-reviewed literature on PubMed (returns PMIDs + links).\n"
    "  • medication_lookup(names): look up drug classification / info for medications.\n\n"
    "RULES:\n"
    "1) Ground every clinical claim, number, guideline, or dosage in a tool result. "
    "Never invent facts and NEVER invent citations, PMIDs, URLs, or study titles.\n"
    "2) Cite the sources you actually used with inline markers like [1], [2] that correspond "
    "to the sources returned by the tools; put each marker next to the specific claim it "
    "supports. Do not write your own 'References' list — the app renders source cards from the "
    "returned sources.\n"
    "3) If the tools do not yield enough evidence, say so plainly rather than guessing.\n"
    "4) Always take the provided CASE CONTEXT into account and tailor the research to it.\n"
    "5) Answer in clear Markdown. Be concise but complete; lead with the clinically actionable "
    "point, then the supporting evidence."
)


def _dr_get_web_helpers():
    """
    Resolve the (kb_retrieve, web_search, url_scrape, domain_of) helpers, preferring
    ``single_rop.chat_service`` and falling back to
    ``doctors_marketplace.services.websearch``. Any missing piece is returned as None
    so callers degrade gracefully. Never raises.
    """
    kb_retrieve = web_search = url_scrape = domain_of = None
    try:
        from single_rop import chat_service as _cs
        kb_retrieve = getattr(_cs, "retrieve_local", None)
        web_search = getattr(_cs, "_serper_search", None)
        url_scrape = getattr(_cs, "_scrape_url", None)
        domain_of = getattr(_cs, "_domain", None)
        logger.info("🔬 doctor_research | reusing single_rop.chat_service helpers "
                    "(kb=%s web=%s scrape=%s)",
                    bool(kb_retrieve), bool(web_search), bool(url_scrape))
    except Exception as e:
        logger.warning("🔬 doctor_research | single_rop.chat_service unavailable: %s", e)

    # Fall back to the marketplace web-search module for anything missing.
    if not (web_search and url_scrape and domain_of):
        try:
            from doctors_marketplace.services import websearch as _ws
            web_search = web_search or getattr(_ws, "serper_search", None)
            url_scrape = url_scrape or getattr(_ws, "scrape_url", None)
            domain_of = domain_of or getattr(_ws, "domain_of", None)
            logger.info("🔬 doctor_research | using doctors_marketplace.services.websearch fallback")
        except Exception as e:
            logger.warning("🔬 doctor_research | websearch fallback unavailable: %s", e)

    if domain_of is None:
        from urllib.parse import urlparse

        def domain_of(url):
            try:
                return urlparse(url).netloc.replace("www.", "")
            except Exception:
                return url or ""

    return kb_retrieve, web_search, url_scrape, domain_of


DOCTOR_RESEARCH_TOOLS = [
    {"type": "function", "function": {
        "name": "search_knowledge_base",
        "description": "Search the clinic's local ROP/KC knowledge base for relevant guideline "
                       "passages and decision rules. Use this first for questions about the "
                       "clinic's screening/decision criteria.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"], "additionalProperties": False,
        },
    }},
    {"type": "function", "function": {
        "name": "search_web",
        "description": "Search the web for clinical guidelines, definitions, or up-to-date "
                       "information. Returns a list of {title, url, snippet}.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The web search query."}},
            "required": ["query"], "additionalProperties": False,
        },
    }},
    {"type": "function", "function": {
        "name": "fetch_url",
        "description": "Fetch and extract the readable text of a specific web page (e.g. a "
                       "result from search_web or search_pubmed) to read it in detail.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The absolute URL to read."}},
            "required": ["url"], "additionalProperties": False,
        },
    }},
    {"type": "function", "function": {
        "name": "search_pubmed",
        "description": "Search PubMed (NCBI) for peer-reviewed medical literature. Returns a "
                       "list of {title, journal, year, pmid, url}. Use for evidence/study lookups.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The PubMed search query."}},
            "required": ["query"], "additionalProperties": False,
        },
    }},
    {"type": "function", "function": {
        "name": "medication_lookup",
        "description": "Look up classification and information for one or more medications by name.",
        "parameters": {
            "type": "object",
            "properties": {"names": {
                "type": "array", "items": {"type": "string"},
                "description": "List of medication names to look up.",
            }},
            "required": ["names"], "additionalProperties": False,
        },
    }},
]


def _dr_tool_search_knowledge_base(query, kb_retrieve):
    """Local KB search. Returns (text_for_model, source_dicts)."""
    if not kb_retrieve:
        return "The local knowledge base is not available in this environment.", []
    try:
        docs = kb_retrieve(query, k=5) or []
    except Exception as e:
        logger.warning("🔬 search_knowledge_base failed: %s", e)
        return f"search_knowledge_base error: {e}", []
    if not docs:
        return "No relevant passages found in the local knowledge base.", []
    lines, sources = [], []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        title = meta.get("title") or _pretty_title(meta.get("source", "knowledge base"))
        content = (getattr(d, "page_content", "") or "")[:800]
        sources.append({"title": title, "url": None, "domain": "knowledge base"})
        lines.append(f"[{title}] {content}")
    return "\n---\n".join(lines), sources


def _dr_tool_search_web(query, web_search, domain_of):
    """Web search. Returns (text_for_model, source_dicts)."""
    if not web_search:
        return "Web search is not available in this environment.", []
    try:
        organic = web_search(query) or []
    except Exception as e:
        logger.warning("🔬 search_web failed: %s", e)
        return f"search_web error: {e}", []
    results, sources = [], []
    for r in organic[:6]:
        url = r.get("link") or r.get("url")
        if not url:
            continue
        title = r.get("title") or domain_of(url)
        snippet = r.get("snippet") or ""
        results.append({"title": title, "url": url, "snippet": snippet})
        sources.append({"title": title, "url": url, "domain": domain_of(url)})
    if not results:
        return "No web results found.", []
    return json.dumps(results, ensure_ascii=False), sources


def _dr_tool_fetch_url(url, url_scrape, domain_of):
    """Fetch readable page text. Returns (text_for_model, source_dicts)."""
    if not url:
        return "No URL provided.", []
    text = ""
    try:
        if url_scrape:
            text = url_scrape(url, char_limit=4000) or ""
        else:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
                tag.decompose()
            text = " ".join(soup.get_text(separator=" ").split())[:4000]
    except Exception as e:
        logger.warning("🔬 fetch_url failed for %s: %s", url, e)
        return f"fetch_url error for {url}: {e}", []
    if not text:
        return f"Could not extract readable text from {url}.", []
    src = [{"title": domain_of(url), "url": url, "domain": domain_of(url)}]
    return text[:4000], src


def _dr_tool_search_pubmed(query):
    """Query NCBI E-utilities (no API key). Returns (text_for_model, source_dicts)."""
    if not query:
        return "No query provided.", []
    try:
        esearch = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "retmode": "json", "retmax": 5, "term": query},
            timeout=10, headers={"User-Agent": "Mozilla/5.0"},
        )
        esearch.raise_for_status()
        ids = (esearch.json().get("esearchresult", {}) or {}).get("idlist", []) or []
        if not ids:
            return "No PubMed results found.", []
        esummary = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "pubmed", "retmode": "json", "id": ",".join(ids)},
            timeout=10, headers={"User-Agent": "Mozilla/5.0"},
        )
        esummary.raise_for_status()
        result = esummary.json().get("result", {}) or {}
    except Exception as e:
        logger.warning("🔬 search_pubmed failed: %s", e)
        return f"search_pubmed error: {e}", []

    articles, sources = [], []
    for pmid in ids:
        rec = result.get(pmid) or {}
        if not rec:
            continue
        title = rec.get("title") or f"PubMed {pmid}"
        journal = rec.get("fulljournalname") or rec.get("source") or ""
        pubdate = rec.get("pubdate") or ""
        year = pubdate.split(" ")[0] if pubdate else ""
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        articles.append({"title": title, "journal": journal, "year": year,
                         "pmid": pmid, "url": url})
        sources.append({"title": title, "url": url, "domain": "pubmed.ncbi.nlm.nih.gov"})
    if not articles:
        return "No PubMed article details found.", []
    return json.dumps(articles, ensure_ascii=False), sources


def _dr_tool_medication_lookup(names):
    """
    Lightweight drug-info lookup (classification) via darooyab.ir — a fast, crash-safe
    variant of the FINDERS drug tools (which target pharmacy availability and are much
    heavier). Returns (text_for_model, source_dicts).
    """
    from urllib.parse import urljoin
    if isinstance(names, str):
        names = [names]
    if not names or not isinstance(names, list):
        return "No medication names provided.", []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    out, sources = [], []
    for name in names[:5]:
        try:
            resp = requests.get(f"https://www.darooyab.ir/Search?SearchText={name}",
                                headers=headers, timeout=12)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            row = soup.select_one("#tbody_DrugList tr")
            link = row.select_one("a.ahref_Generic") if row else None
            if not link or not link.get("href"):
                out.append({"medication": name, "info": "not found"})
                continue
            drug_url = urljoin("https://www.darooyab.ir", link["href"])
            page = requests.get(drug_url, headers=headers, timeout=12)
            page.raise_for_status()
            dsoup = BeautifulSoup(page.text, "html.parser")
            info = dsoup.select_one("#divExtraInfo")
            classification = ""
            if info:
                classification = " ".join(info.get_text(separator=" ").split())[:600]
            out.append({"medication": name, "url": drug_url,
                        "classification": classification or "no classification found"})
            sources.append({"title": f"{name} (darooyab.ir)", "url": drug_url,
                            "domain": "darooyab.ir"})
        except Exception as e:
            logger.warning("🔬 medication_lookup failed for %r: %s", name, e)
            out.append({"medication": name, "error": str(e)})
    return json.dumps(out, ensure_ascii=False), sources


def _dr_register_sources(new_sources, registry, ordered):
    """Dedupe sources by url (or title for KB) and assign stable 1-based ids."""
    for s in new_sources or []:
        key = s.get("url") or ("kb::" + (s.get("title") or ""))
        if key in registry:
            continue
        sid = len(ordered) + 1
        entry = {"id": sid, "title": s.get("title") or "source",
                 "url": s.get("url"), "domain": s.get("domain")}
        registry[key] = entry
        ordered.append(entry)


def doctor_research_chat(message, history, context_text=""):
    """
    Research assistant for a doctor/manager reviewing an employee's ROP/KC case.

    Runs a bounded (≈4 round) GAPGPT tool-calling loop with tools for the local
    knowledge base, web search, URL reading, PubMed and medication lookup.

    Args:
        message: the doctor's latest message (str).
        history: prior turns as a list of {role, content} dicts (may be empty/None).
        context_text: a text summary of the case under review (str).

    Returns:
        (answer_markdown, sources_list) where sources_list items are
        {id, title, url, domain}. Never raises.
    """
    logger.info("🔬 doctor_research_chat() | message_len=%d | history_turns=%d | context_len=%d",
                len(message or ""), len(history or []), len(context_text or ""))

    kb_retrieve, web_search, url_scrape, domain_of = _dr_get_web_helpers()

    def run_tool(name, args):
        try:
            if name == "search_knowledge_base":
                return _dr_tool_search_knowledge_base(args.get("query", ""), kb_retrieve)
            if name == "search_web":
                return _dr_tool_search_web(args.get("query", ""), web_search, domain_of)
            if name == "fetch_url":
                return _dr_tool_fetch_url(args.get("url", ""), url_scrape, domain_of)
            if name == "search_pubmed":
                return _dr_tool_search_pubmed(args.get("query", ""))
            if name == "medication_lookup":
                return _dr_tool_medication_lookup(args.get("names", []))
            return f"Unknown tool: {name}", []
        except Exception as e:  # last-resort guard — tools must never crash the loop
            logger.warning("🔬 tool %r raised: %s", name, e)
            return f"{name} error: {e}", []

    try:
        client = get_openai_client()
    except Exception as e:
        logger.exception("🔬 doctor_research_chat | could not init LLM client: %s", e)
        return ("I couldn't reach the language model just now. Please try again in a moment.", [])

    case_block = (f"CASE CONTEXT (the case under review):\n{context_text}\n\n"
                  if context_text else "CASE CONTEXT: (none provided)\n\n")
    messages = (
        [{"role": "system", "content": DOCTOR_RESEARCH_SYSTEM_PROMPT}]
        + list(history or [])
        + [{"role": "user", "content": case_block + f"DOCTOR'S QUESTION:\n{message}"}]
    )

    registry, ordered_sources = {}, []
    MAX_ROUNDS = 4
    answer = ""

    for round_no in range(MAX_ROUNDS):
        try:
            response = client.chat.completions.create(
                model=gapgpt_model,
                messages=messages,
                tools=DOCTOR_RESEARCH_TOOLS,
            )
        except Exception as e:
            logger.exception("🔬 doctor_research_chat | LLM call failed (round %d): %s",
                             round_no, e)
            return ("I couldn't reach the language model just now. Please try again in a moment.",
                    ordered_sources)

        choice = response.choices[0]
        msg = choice.message
        logger.info("🔬 round %d | finish_reason=%s | tool_calls=%s",
                    round_no, choice.finish_reason,
                    len(msg.tool_calls) if msg.tool_calls else 0)

        if not msg.tool_calls:
            answer = msg.content or ""
            break

        # Append the assistant's tool-request message, then each tool result.
        messages.append(msg)
        for tc in msg.tool_calls:
            fname = tc.function.name
            try:
                fargs = json.loads(tc.function.arguments or "{}")
            except Exception:
                fargs = {}
            logger.info("🔬   ↳ tool call: %s(%s)", fname, fargs)
            tool_text, tool_sources = run_tool(fname, fargs)
            _dr_register_sources(tool_sources, registry, ordered_sources)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_text})
    else:
        # Loop exhausted without a final text answer — force one last completion.
        try:
            final = client.chat.completions.create(model=gapgpt_model, messages=messages)
            answer = final.choices[0].message.content or ""
        except Exception as e:
            logger.exception("🔬 doctor_research_chat | final completion failed: %s", e)
            answer = ("I gathered research material but couldn't compose a final answer. "
                      "Please try again.")

    logger.info("🔬 doctor_research_chat done | answer_len=%d | sources=%d",
                len(answer or ""), len(ordered_sources))
    return answer, ordered_sources