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

    # Fix the escaped quotes
    content = content.replace(r"{% static \'mediversai_logo_final.png\' %}", "{% static 'mediversai_logo_final.png' %}")

    with open(filepath, 'w') as f:
        f.write(content)

print("Fixed quotes successfully.")
