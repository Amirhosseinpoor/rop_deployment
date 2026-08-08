import re

files = [
    '/home/amir/Desktop/rop/templates/doctors_marketplace/market_index2.html',
    '/home/amir/Desktop/rop/templates/doctors_marketplace/doctor_detail2.html',
    '/home/amir/Desktop/rop/templates/doctors_marketplace/chat.html',
    '/home/amir/Desktop/rop/doctors_marketplace/templates/doctors_marketplace/studio/index2.html',
    '/home/amir/Desktop/rop/doctors_marketplace/templates/doctors_marketplace/studio/new2.html',
    '/home/amir/Desktop/rop/doctors_marketplace/templates/doctors_marketplace/studio/kb2.html',
]

for filepath in files:
    with open(filepath, 'r') as f:
        content = f.read()

    # 1. Replace the HTML for market_index2, doctor_detail2, chat
    content = re.sub(
        r'<span class="brand-mark">.*?</span>\s*<span class="brand-name">Mediverse<span>AI</span></span>',
        r'<img src="{% static \'mediversai_logo_final.png\' %}" alt="MediverseAI Logo" style="height: 38px; opacity: 0.92;">',
        content,
        flags=re.DOTALL
    )

    # 2. Replace the HTML for index2, new2, kb2 (which don't have brand-name but have Doctor Studio etc next to it)
    content = re.sub(
        r'<span class="brand-mark">.*?</span>',
        r'<img src="{% static \'mediversai_logo_final.png\' %}" alt="MediverseAI Logo" style="height: 38px; opacity: 0.92;">',
        content,
        flags=re.DOTALL
    )

    # 3. Remove .brand-mark CSS completely
    content = re.sub(
        r'\s*\.brand-mark\s*\{[^}]+\}\s*\.brand-mark img\s*\{[^}]+\}',
        '',
        content,
        flags=re.DOTALL
    )

    # 4. Remove .brand-name CSS
    content = re.sub(
        r'\s*\.brand-name\s*\{[^}]+\}\s*\.brand-name span\s*\{[^}]+\}',
        '',
        content,
        flags=re.DOTALL
    )

    with open(filepath, 'w') as f:
        f.write(content)

print("Updated logos successfully.")
