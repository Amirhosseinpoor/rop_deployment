import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# --- تنظیمات اولیه ---
URL = "https://www.darooyab.ir/%d8%af%d8%a7%d8%b1%d9%88%d8%ae%d8%a7%d9%86%d9%87"
TEHRAN_PROVINCE_VALUE = "40"
all_pharmacies_data = []

# --- راه‌اندازی WebDriver ---
driver = webdriver.Chrome()
driver.maximize_window()
wait = WebDriverWait(driver, 15)

try:
    driver.get(URL)
    print(f"✅ وارد صفحه شد: {URL}")

    # انتخاب استان تهران
    province_dropdown = wait.until(EC.presence_of_element_located((By.ID, "ProvinceId")))
    province_select = Select(province_dropdown)
    province_select.select_by_value(TEHRAN_PROVINCE_VALUE)
    print("✅ استان 'تهران' انتخاب شد")

    # مکث برای ارسال AJAX
    time.sleep(2)

    # انتظار برای پر شدن لیست شهرستان‌ها
    wait.until(EC.presence_of_element_located((By.XPATH, "//select[@id='CityId']/option[contains(text(), 'تهران')]")))

    # استخراج شهرها
    city_select = Select(driver.find_element(By.ID, "CityId"))
    cities = [(opt.get_attribute("value"), opt.text.strip()) for opt in city_select.options if opt.get_attribute("value") != "0"]
    print(f"✅ {len(cities)} شهر در استان تهران پیدا شد")

    for city_value, city_name in cities:
        try:
            print(f"\n🔄 در حال پردازش شهر: {city_name} (value={city_value})")

            old_html = driver.find_element(By.ID, "lstPharmacyList").get_attribute("innerHTML")

            city_select = Select(driver.find_element(By.ID, "CityId"))
            city_select.select_by_value(city_value)

            WebDriverWait(driver, 15).until(
                lambda d: d.find_element(By.ID, "lstPharmacyList").get_attribute("innerHTML") != old_html
            )

            wait.until(EC.presence_of_element_located((By.XPATH, "//div[@id='lstPharmacyList']//table/tbody/tr")))

            pharmacy_rows = driver.find_elements(By.XPATH, "//div[@id='lstPharmacyList']//table/tbody/tr")
            print(f"🟢 {len(pharmacy_rows)} داروخانه در {city_name} یافت شد")

            for row in pharmacy_rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) >= 3:
                    try:
                        name_element = cells[0].find_element(By.TAG_NAME, "a")
                        name = name_element.text.strip()
                        link = name_element.get_attribute("href").strip()

                        pharmacy_info = {
                            "name": name,
                            "province": "تهران",
                            "city": city_name,
                            "url": link
                        }
                        all_pharmacies_data.append(pharmacy_info)

                    except NoSuchElementException:
                        print("⚠️ ردیفی با ساختار غیرمنتظره. رد شد.")
                        continue

        except (TimeoutException, NoSuchElementException) as e:
            print(f"❌ خطا در پردازش شهر {city_name}: {e}")
            continue

    print("\n✅ استخراج اولیه داروخانه‌ها کامل شد. شروع استخراج جزئیات...\n")

    # --- استخراج جزئیات برای هر داروخانه ---
    enhanced_data = []

    for pharmacy in all_pharmacies_data:
        try:
            driver.get(pharmacy["url"])
            print(f"🧭 {pharmacy['city']} > {pharmacy['name']}")

            try:
                address = wait.until(EC.presence_of_element_located((By.ID, "h2Address"))).text.strip()
            except:
                address = ""

            try:
                phone = driver.find_element(By.ID, "h3Phone").text.strip()
            except:
                phone = ""

            drugs = []
            try:
                drug_rows = driver.find_elements(By.XPATH, "//table[@class='table table-bordered alternate_color']/tbody/tr")
                for row in drug_rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 3:
                        drug_name = cells[0].text.strip()
                        generic_name = cells[1].text.strip()
                        available_days = cells[2].text.strip()
                        drugs.append({
                            "brand_name": drug_name,
                            "generic_name": generic_name,
                            "available_days": available_days
                        })
            except:
                pass

            pharmacy["address"] = address
            pharmacy["phone"] = phone
            pharmacy["drugs"] = drugs
            enhanced_data.append(pharmacy)

        except Exception as e:
            print(f"❌ خطا در بارگذاری جزئیات داروخانه {pharmacy['name']}: {e}")
            continue

finally:
    driver.quit()
    print("\n🧹 مرورگر بسته شد.")

# --- ذخیره به صورت CSV ---
records = []
for pharmacy in enhanced_data:
    if pharmacy.get("drugs"):
        for drug in pharmacy["drugs"]:
            records.append({
                "pharmacy_name": pharmacy["name"],
                "province": pharmacy["province"],
                "city": pharmacy["city"],
                "address": pharmacy.get("address", ""),
                "phone": pharmacy.get("phone", ""),
                "drug_brand": drug["brand_name"],
                "drug_generic": drug["generic_name"],
                "available_days": drug["available_days"],
                "pharmacy_url": pharmacy["url"]
            })
    else:
        records.append({
            "pharmacy_name": pharmacy["name"],
            "province": pharmacy["province"],
            "city": pharmacy["city"],
            "address": pharmacy.get("address", ""),
            "phone": pharmacy.get("phone", ""),
            "drug_brand": "",
            "drug_generic": "",
            "available_days": "",
            "pharmacy_url": pharmacy["url"]
        })

df = pd.DataFrame(records)
csv_path = "tehran_pharmacies_with_drugs.csv"
df.to_csv(csv_path, index=False, encoding='utf-8-sig')

