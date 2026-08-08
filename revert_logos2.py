import os

files_with_name = [
    '/home/amir/Desktop/rop/templates/doctors_marketplace/market_index2.html',
    '/home/amir/Desktop/rop/templates/doctors_marketplace/doctor_detail2.html',
    '/home/amir/Desktop/rop/templates/doctors_marketplace/chat.html',
]

files_without_name = [
    '/home/amir/Desktop/rop/doctors_marketplace/templates/doctors_marketplace/studio/index2.html',
    '/home/amir/Desktop/rop/doctors_marketplace/templates/doctors_marketplace/studio/new2.html',
    '/home/amir/Desktop/rop/doctors_marketplace/templates/doctors_marketplace/studio/kb2.html',
]

for filepath in files_with_name + files_without_name:
    with open(filepath, 'r') as f:
        content = f.read()
    
    target = '<img src="{% static \'mediversai_logo_final.png\' %}" alt="MediverseAI Logo" style="height: 38px; opacity: 0.92;">'
    
    if filepath in files_with_name:
        replacement = """<span class="brand-mark">
                    <img src="{% static 'icons/mediversai_logo_final-03.png' %}" alt="MediverseAI Logo">
                </span>
                <span class="brand-name">Mediverse<span>AI</span></span>"""
    else:
        replacement = """<span class="brand-mark"><img src="{% static 'icons/mediversai_logo_final-03.png' %}" alt="MediverseAI Logo"></span>"""
        
    content = content.replace(target, replacement)

    with open(filepath, 'w') as f:
        f.write(content)

print("Reverted logos successfully.")
