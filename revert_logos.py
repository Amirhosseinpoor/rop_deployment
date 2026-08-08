import re
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

css_to_add = """
        .brand-mark {
            width: 34px; height: 34px;
            border-radius: 10px;
            background: var(--spark-grad);
            display: grid; place-items: center;
            box-shadow: 0 6px 16px rgba(109, 90, 230, .35);
            flex-shrink: 0; padding: 2px;
        }
        .brand-mark img { width: 100%; height: 100%; object-fit: contain; border-radius: 8px; background: #fff; }
        .brand-name { font-weight: 700; font-size: 15px; letter-spacing: -.2px; }
        .brand-name span { color: var(--accent); }
"""

for filepath in files_with_name + files_without_name:
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Add CSS before </style>
    if '.brand-mark {' not in content:
        content = content.replace('</style>', css_to_add + '\n    </style>')
    
    target = r'<img src="{% static \'mediversai_logo_final.png\' %}" alt="MediverseAI Logo" style="height: 38px; opacity: 0.92;">'
    
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
