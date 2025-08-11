import json
import requests
import re
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0"
}


def get_doctor_links(search_url, params=None, max_doctors=5):
    try:
        response = requests.get(search_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"خطا در دریافت لیست پزشکان: {e}")

    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.select('article a[href^="/doctor/"]')
    return [f"https://doctoreto.com{a['href']}" for a in links[:max_doctors]]


def parse_doctor_profile(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"خطا در بارگذاری پروفایل: {url} - {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    doctor = {"profile_url": url}

    name_tag = soup.select_one('div.sc-212a8fa3-4.qvYGb')
    doctor['name'] = name_tag.text.strip() if name_tag else "N/A"

    medical_id_tag = soup.select_one("div.sc-212a8fa3-12.jPJuqw > span")
    if medical_id_tag:
        match = re.search(r'\d+', medical_id_tag.text)
        doctor['medical_id'] = match.group() if match else "N/A"
    else:
        doctor['medical_id'] = "N/A"

    specialty_tag = soup.select_one('div.sc-910b97e5-2.flOxFb')
    doctor['specialty'] = specialty_tag.text.strip() if specialty_tag else "N/A"

    locations = []
    add_tag = soup.select_one('div.sc-4e6b0e39-4.ckjzZD')
    add = add_tag.text.strip() if add_tag else "N/A"

    add2_tag = soup.select_one('div.sc-dee9c446-2.jjnVxK')
    add2 = add2_tag.text.strip() if add2_tag else ''

    locations.append(add)
    locations.append(add2)

    doctor['locations'] = locations
    services_tags = soup.select('div.sc-a595fbf3-1.hDEPPr > span.sc-18058bc8-0.elbXID.sc-a595fbf3-2.hbWmHw')
    doctor['services'] = ', '.join([tag.text.strip() for tag in services_tags]) if services_tags else 'N/A'

    insurance_tag = soup.select('div.sc-395236ed-1.ldUByb > div.sc-395236ed-2.hZJELU > div.sc-395236ed-5.bcsEfX')
    doctor['accepted_insurances'] = ', '.join([tag.text.strip() for tag in insurance_tag]) if insurance_tag else 'N/A'

    score_tag = soup.select_one('div.sc-28976a9b-7.caezWg')
    from_user = soup.select_one('div.sc-28976a9b-5.kOQRfG > span')

    if score_tag and from_user:
        doctor['score_summary'] = f"امتیاز {score_tag.text.strip()}) از ۵ {from_user.text.strip()}) "
    else:
        doctor['score_summary'] = 'امتیاز ثبت نشده'

    return doctor


def scrape_doctors(city='tehran', speciality='cardiologist', region=None, insurance=None):
    base_url = "https://doctoreto.com"
    search_url = f"{base_url}/doctors/speciality/{speciality}/city/{city}/"
    if region:
        search_url += f"region/{region}/"
    params = {'insurance': insurance} if insurance else None

    try:
        doctor_urls = get_doctor_links(search_url, params=params)
    except RuntimeError as err:
        return json.dumps({"error": str(err)}, ensure_ascii=False, indent=4)

    doctors_data = []
    for url in doctor_urls:
        profile_data = parse_doctor_profile(url)
        if profile_data:
            doctors_data.append(profile_data)

    return json.dumps(doctors_data, ensure_ascii=False, indent=4)





def slugify(text):
    return re.sub(r'[^a-zA-Z0-9]+', '-', text.lower()).strip('-')


def convert_persian_to_english(text):
    persian_digits = '۰۱۲۳۴۵۶۷۸۹'
    english_digits = '0123456789'
    return text.translate(str.maketrans(persian_digits, english_digits))


def scrape_drugs(medicine_name):
    slug = slugify(medicine_name)
    name = slug
    url = f"https://mokamelkhoone.com/products/now-foods-{slug}/"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract price
        price_tag = soup.select_one("p.price bdi")
        if not price_tag:
            raise Exception("Price element not found")

        raw_price = price_tag.text.strip()
        price_only = re.sub(r"[^\d۰-۹٬,]", "", raw_price)
        price = convert_persian_to_english(price_only)

        # Extract product info
        info_tag = soup.select_one(".product-content-c")
        if not info_tag:
            raise Exception("Product info not found")
        info_text = info_tag.text.strip()

        # Extract title
        title_tag = soup.select_one("h1.elementor-heading-title")
        product_title = title_tag.text.strip() if title_tag else ""
        final_response = {
            "name": name,
            "url": url,
            "price": price,
            "info": info_text,
            "title": product_title
        }

        return json.dumps(final_response)

    except Exception as e:
        return json.dumps({
            "url": url,
            "error": str(e)
        })


