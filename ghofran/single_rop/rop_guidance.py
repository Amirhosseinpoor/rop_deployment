# ---------------- ROP Clinical Guidance (static) ----------------
# Keys are (zone, plus, stage) with normalized strings like: ("zone 1","plus","stage 3")
ROP_GUIDANCE: dict[tuple[str, str, str], dict] = {
    # ===== Zone 1 – PLUS =====
    ("zone 1", "plus", "stage 3"): {
        "title": "Zone 1 · Plus · Stage 3 → Treatment",
        "text": (
            "In Retinopathy of Prematurity (ROP), Zone 1 involvement with Plus disease and Stage 3 indicates "
            "aggressive posterior ROP, characterized by extraretinal fibrovascular proliferation (EFP) along with "
            "vascular dilation and tortuosity. This meets Type 1 ROP criteria per ETROP guidelines, posing a high risk "
            "of rapid progression to retinal detachment and potential blindness if untreated. Decision: Treatment within "
            "less than 1 week, typically involving peripheral retinal ablation via laser photocoagulation or intravitreal "
            "anti-VEGF injections such as bevacizumab; urgent referral to a pediatric ophthalmologist is essential, with "
            "post-treatment monitoring every 5-7 days to assess regression and complications like macular dragging."
        )
    },
    ("zone 1", "plus", "stage 2"): {
        "title": "Zone 1 · Plus · Stage 2 → Treatment",
        "text": (
            "ROP in Zone 1 at Stage 2 with Plus disease features a ridge of elevated tissue at the vascular-avascular "
            "junction, combined with aggressive vascular changes signaling increased retinal ischemia. According to ICROP "
            "classification, the presence of Plus in this posterior zone elevates the risk of neovascularization and "
            "vitreous hemorrhage, necessitating immediate intervention to prevent irreversible visual impairment. "
            "Decision: Treatment within less than 1 week using laser therapy to destroy avascular retina or anti-VEGF "
            "agents; coordinate with a retinal specialist for binocular indirect ophthalmoscopy follow-up, ensuring parental "
            "education on signs of progression like strabismus or leukocoria."
        )
    },
    ("zone 1", "plus", "stage 1"): {
        "title": "Zone 1 · Plus · Stage 1 → Treatment",
        "text": (
            "Early Stage 1 ROP in Zone 1 with Plus disease shows a demarcation line separating vascularized from "
            "avascular retina, but the Plus component (arterial tortuosity and venous dilation) indicates severe hypoxia "
            "and imminent progression. ETROP studies highlight that Zone 1 Plus, even at low stages, correlates with poor "
            "outcomes without prompt action, including risks of macular involvement. Decision: Treatment within less than "
            "1 week via cryotherapy or modern modalities like anti-VEGF; immediate specialist consultation is critical, with "
            "serial examinations to monitor for threshold disease and potential side effects such as endophthalmitis."
        )
    },
    ("zone 1", "plus", "stage 0"): {
        "title": "Zone 1 · Plus · Stage 0 → Treatment",
        "text": (
            "Immature vascularization at Stage 0 in Zone 1 with Plus disease reflects incomplete retinal vessel development "
            "accompanied by abnormal vascular caliber, suggesting high-risk pre-threshold ROP per ICROP. This configuration "
            "increases susceptibility to aggressive posterior ROP (AP-ROP), with potential for cicatricial changes leading to "
            "tractional detachment. Decision: Treatment within less than 1 week, often starting with anti-VEGF to promote "
            "regression; refer urgently to neonatology-integrated ophthalmology services, emphasizing fluorescein angiography "
            "if needed for precise avascular zone mapping and long-term visual acuity surveillance."
        )
    },
    ("zone 1", "plus", "normal"): {
        "title": "Zone 1 · Plus · Normal → Treatment",
        "text": (
            "Normal staging in Zone 1 with Plus disease implies subtle vascular anomalies without overt ROP features, yet the "
            "Plus signs of tortuosity and dilation signal underlying ischemia in the posterior retina. Guidelines from AAP "
            "recommend early intervention in such cases to avert progression to fibrovascular membranes. Decision: Treatment "
            "within less than 1 week using targeted laser or pharmacologic approaches; specialist evaluation is imperative, "
            "including wide-field imaging to document baseline and track peripheral vascular maturation over subsequent weeks."
        )
    },

    # ===== Zone 1 – NO PLUS =====
    ("zone 1", "no plus", "stage 3"): {
        "title": "Zone 1 · No Plus · Stage 3 → Treatment",
        "text": (
            "Stage 3 ROP in Zone 1 without Plus involves EFP extending into the vitreous, posing a substantial threat of "
            "macular detachment despite absent aggressive vascular signs. ETROP criteria classify this as Type 1 ROP due to "
            "the central location, with risks including retinal folds and vision loss. Decision: Treatment within less than "
            "1 week via laser ablation of peripheral avascular areas; prompt referral to a vitreoretinal surgeon, with follow-up "
            "imaging to confirm EFP regression and monitor for secondary complications like glaucoma."
        )
    },
    ("zone 1", "no plus", "stage 2"): {
        "title": "Zone 1 · No Plus · Stage 2 → Follow-up ≤1 week",
        "text": (
            "Stage 2 ROP in Zone 1 without Plus presents as a ridge formation at the vascular terminus, indicating moderate "
            "ischemia in a high-risk posterior area per ICROP. While not yet threshold, close surveillance is needed to detect "
            "emerging Plus or proliferation, as delays can lead to tractional forces. Decision: Follow-up within <=1 week using "
            "fundus examination; educate caregivers on prematurity-related risks, and prepare for potential escalation to "
            "treatment if changes occur, incorporating oxygen saturation monitoring in NICU settings."
        )
    },
    ("zone 1", "no plus", "stage 1"): {
        "title": "Zone 1 · No Plus · Stage 1 → Follow-up ≤1 week",
        "text": (
            "Early Stage 1 in Zone 1 without Plus shows a flat demarcation line, signifying initial vascular arrest with "
            "potential for spontaneous resolution but heightened risk due to zonation. AAP screening protocols emphasize "
            "weekly checks to track progression metrics like clock hours involved. Decision: Follow-up within <=1 week with "
            "detailed ophthalmoscopy; document postmenstrual age and birth weight for risk stratification, advising on "
            "environmental factors like supplemental oxygen that may influence ROP evolution."
        )
    },
    ("zone 1", "no plus", "stage 0"): {
        "title": "Zone 1 · No Plus · Stage 0 → Follow-up 1–2 weeks",
        "text": (
            "Stage 0 (immature, No ROP) in Zone 1 without Plus denotes incomplete vascularization without demarcation, "
            "common in very preterm infants but requiring vigilance for AP-ROP onset. Studies indicate that posterior "
            "immaturity correlates with higher incidence of severe ROP if unmonitored. Decision: Follow-up in 1-2 weeks to "
            "assess vessel extension; use digital retinal imaging for serial comparison, and integrate with multidisciplinary "
            "care including neonatologists for optimizing growth factors like IGF-1 levels."
        )
    },
    ("zone 1", "no plus", "normal"): {
        "title": "Zone 1 · No Plus · Normal → Follow-up 1–2 weeks",
        "text": (
            "Normal mature vascularization in Zone 1 without Plus suggests stable retinal development, yet in premature "
            "contexts, subtle microvascular changes may persist. ICROP advises continued screening until full vascular maturity "
            "to rule out late-onset anomalies. Decision: Follow-up in 1-2 weeks for confirmatory examination; provide parental "
            "guidance on long-term visual assessments, potentially including electroretinography if family history of retinal "
            "disorders exists."
        )
    },

    # ===== Zone 2 – PLUS =====
    ("zone 2", "plus", "stage 3"): {
        "title": "Zone 2 · Plus · Stage 3 → Treatment",
        "text": (
            "Stage 3 ROP in Zone 2 with Plus features EFP with aggressive vascular signs, classifying as Type 1 ROP and "
            "risking peripheral detachment or plus-related complications like hemorrhage. ETROP trials support early treatment "
            "to improve structural outcomes. Decision: Treatment within less than 1 week using laser or anti-VEGF; urgent "
            "specialist involvement, with post-intervention fluorescein angiography to evaluate perfusion and prevent recurrence."
        )
    },
    ("zone 2", "plus", "stage 2"): {
        "title": "Zone 2 · Plus · Stage 2 → Treatment",
        "text": (
            "Stage 2 in Zone 2 with Plus shows ridge elevation alongside vascular tortuosity, indicating threshold disease per "
            "guidelines and high progression potential to Stage 3 or beyond. This setup increases odds of cicatricial ROP leading "
            "to strabismus. Decision: Treatment within less than 1 week via ablative therapy; coordinate with pediatric retina "
            "experts, emphasizing anesthetic considerations in neonates and monitoring for systemic anti-VEGF effects."
        )
    },
    ("zone 2", "plus", "stage 1"): {
        "title": "Zone 2 · Plus · Stage 1 → Follow-up 1–2 weeks",
        "text": (
            "Stage 1 ROP in Zone 2 with Plus displays a demarcation line with mild aggressive features, warranting observation "
            "as it borders pre-threshold but may regress. ICROP notes variable outcomes, influenced by gestational age. "
            "Decision: Follow-up in 1-2 weeks for progression assessment; use wide-field cameras for documentation, and discuss "
            "nutritional interventions like omega-3 supplementation to support vascular health."
        )
    },
    ("zone 2", "plus", "stage 0"): {
        "title": "Zone 2 · Plus · Stage 0 → Follow-up 1–2 weeks",
        "text": (
            "Immature Stage 0 in Zone 2 with Plus suggests early vascular stress without structural ROP, but Plus elevates "
            "monitoring needs to catch rapid changes. Research links this to hyperoxia exposure in NICUs. Decision: Follow-up "
            "in 1-2 weeks with detailed funduscopy; integrate blood gas monitoring, and prepare educational materials for families "
            "on ROP risk factors."
        )
    },
    ("zone 2", "plus", "normal"): {
        "title": "Zone 2 · Plus · Normal → Follow-up 1–2 weeks",
        "text": (
            "Normal vascular state in Zone 2 with Plus implies residual aggressive signs post-regression, requiring checks for "
            "subtle recurrences. AAP guidelines stress extended screening in such variants. Decision: Follow-up in 1-2 weeks to "
            "confirm stability; employ optical coherence tomography (OCT) if available for macular evaluation, and advise on "
            "follow-up vision therapy if needed."
        )
    },

    # ===== Zone 2 – NO PLUS =====
    ("zone 2", "no plus", "stage 3"): {
        "title": "Zone 2 · No Plus · Stage 3 → Follow-up ≤1 week",
        "text": (
            "Stage 3 in Zone 2 without Plus involves EFP but lacks aggressive vascularity, classified as Type 2 ROP with "
            "moderate detachment risk. ETROP recommends tight intervals to detect Plus onset. Decision: Follow-up within <=1 "
            "week using imaging; monitor clock-hour extent, and collaborate with endocrinologists for growth hormone influences "
            "on ROP."
        )
    },
    ("zone 2", "no plus", "stage 2"): {
        "title": "Zone 2 · No Plus · Stage 2 → Follow-up 1–2 weeks",
        "text": (
            "Stage 2 ROP in Zone 2 without Plus features ridge without proliferation, often regressing spontaneously but needing "
            "surveillance for threshold crossing. Studies show better prognosis with early detection. Decision: Follow-up in 1-2 "
            "weeks for vascular reassessment; document with photos, and consider anemia corrections as contributing factors."
        )
    },
    ("zone 2", "no plus", "stage 1"): {
        "title": "Zone 2 · No Plus · Stage 1 → Follow-up 2 weeks",
        "text": (
            "Stage 1 in Zone 2 without Plus shows simple demarcation, low-risk for progression in mid-zone per ICROP. Routine "
            "monitoring suffices for most cases. Decision: Follow-up in 2 weeks to track line migration; educate on sepsis risks "
            "exacerbating ROP, and plan for discharge criteria once Zone 3 is reached."
        )
    },
    ("zone 2", "no plus", "stage 0"): {
        "title": "Zone 2 · No Plus · Stage 0 → Follow-up 2–3 weeks",
        "text": (
            "Immature Stage 0 in Zone 2 without Plus indicates ongoing vascularization without disease, suitable for longer "
            "intervals as per AAP. Focus on natural maturation. Decision: Follow-up in 2-3 weeks; use telemedicine if feasible "
            "for remote areas, and monitor weight gain as a proxy for retinal health."
        )
    },
    ("zone 2", "no plus", "normal"): {
        "title": "Zone 2 · No Plus · Normal → Follow-up 2–3 weeks",
        "text": (
            "Normal maturity in Zone 2 without Plus suggests nearing screening endpoint, but confirm no peripheral issues. "
            "Guidelines allow extension based on PMA. Decision: Follow-up in 2-3 weeks for final check; discuss long-term "
            "amblyopia risks, and integrate with developmental pediatrics."
        )
    },

    # ===== Zone 3 – PLUS / NO PLUS =====
    ("zone 3", "plus", "stage 2"): {
        "title": "Zone 3 · Plus · Stage 2 → Follow-up 2–3 weeks",
        "text": (
            "Stage 2 in Zone 3 with Plus features ridge in peripheral retina with mild aggression, rarely progressing but "
            "monitored for completeness. ICROP notes high regression rates here. Decision: Follow-up in 2-3 weeks; use fundus "
            "photography, and advise on light exposure controls in incubators."
        )
    },
    ("zone 3", "plus", "stage 1"): {
        "title": "Zone 3 · Plus · Stage 1 → Follow-up 2–3 weeks",
        "text": (
            "Stage 1 ROP in Zone 3 with Plus shows peripheral demarcation with vascular signs, low urgency due to anterior "
            "location. Often self-resolves per studies. Decision: Follow-up in 2-3 weeks for confirmation; monitor for rare "
            "traction, and educate on genetic predispositions if relevant."
        )
    },
    ("zone 3", "no plus", "stage 2"): {
        "title": "Zone 3 · No Plus · Stage 2 → Follow-up 2–3 weeks",
        "text": (
            "Stage 2 in Zone 3 without Plus involves peripheral ridge, minimal risk of vision-threatening issues. ETROP supports "
            "extended follow-up. Decision: Follow-up in 2-3 weeks; prepare for screening termination if regressed, and note "
            "transfusion history as a modifier."
        )
    },
    ("zone 3", "no plus", "stage 1"): {
        "title": "Zone 3 · No Plus · Stage 1 → Follow-up 2–3 weeks",
        "text": (
            "Stage 1 in Zone 3 without Plus is the mildest form, with demarcation far from macula. High spontaneous resolution "
            "likelihood. Decision: Follow-up in 2-3 weeks; use simple ophthalmoscopy, and discuss vitamin E supplementation trials."
        )
    },
    ("zone 3", "no plus", "stage 0"): {
        "title": "Zone 3 · No Plus · Stage 0 → Follow-up 2–3 weeks",
        "text": (
            "Immature Stage 0 in Zone 3 without Plus nears full vascularization, signaling end of at-risk period. AAP criteria "
            "for discharge apply. Decision: Follow-up in 2-3 weeks; confirm with imaging, and plan for routine pediatric eye "
            "exams post-discharge."
        )
    },
    ("zone 3", "no plus", "normal"): {
        "title": "Zone 3 · No Plus · Normal → Follow-up 2–3 weeks",
        "text": (
            "Fully normal in Zone 3 without Plus indicates mature retina, ready for screening cessation. Ensure no missed "
            "peripherals. Decision: Follow-up in 2-3 weeks as final verification; provide certificates of completion, and advise "
            "on school-age vision screening."
        )
    },

    # ===== Stages 4–5 (detachment) WITHOUT PLUS =====
    ("zone 1", "no plus", "stage 4"): {
        "title": "Zone 1 · No Plus · Stage 4 → Treatment",
        "text": (
            "Stage 4 ROP in Zone 1 without Plus involves partial retinal detachment, often tractional from prior proliferation, "
            "threatening central vision. Urgent surgical need per guidelines. Decision: Treatment within less than 1 week, "
            "potentially vitrectomy; mobilize surgical team, with pre-op OCT for extent mapping."
        )
    },
    ("zone 2", "no plus", "stage 4"): {
        "title": "Zone 2 · No Plus · Stage 4 → Treatment",
        "text": (
            "Stage 4 in Zone 2 without Plus features localized detachment, requiring intervention to prevent total involvement. "
            "Prognosis varies with macular status. Decision: Treatment within less than 1 week via scleral buckling or laser; "
            "specialist OR preparation, monitoring for amblyopia post-op."
        )
    },
    ("zone 3", "no plus", "stage 4"): {
        "title": "Zone 3 · No Plus · Stage 4 → Treatment",
        "text": (
            "Peripheral Stage 4 in Zone 3 without Plus shows anterior detachment, less vision-threatening but still needs "
            "addressal. Rare but actionable. Decision: Treatment within less than 1 week with targeted surgery; assess with "
            "ultrasound, and focus on rehabilitation potential."
        )
    },
    ("zone 1", "no plus", "stage 5"): {
        "title": "Zone 1 · No Plus · Stage 5 → Treatment",
        "text": (
            "Stage 5 ROP in Zone 1 without Plus is total funnel detachment, often with poor visual prognosis but salvage "
            "attempts warranted. Involves complex vitreoretinal surgery. Decision: Treatment within less than 1 week; urgent "
            "enucleation consideration if intractable, with psychological support for families."
        )
    },
    ("zone 2", "no plus", "stage 5"): {
        "title": "Zone 2 · No Plus · Stage 5 → Treatment",
        "text": (
            "Total Stage 5 detachment in Zone 2 without Plus signifies end-stage ROP, focusing on pain management and cosmesis. "
            "Surgical options limited. Decision: Treatment within less than 1 week via advanced procedures; involve low-vision "
            "specialists early for adaptive strategies."
        )
    },
    ("zone 3", "no plus", "stage 5"): {
        "title": "Zone 3 · No Plus · Stage 5 → Treatment",
        "text": (
            "Stage 5 in Zone 3 without Plus is peripheral total detachment, potentially sparing central vision if isolated. "
            "Requires confirmation imaging. Decision: Treatment within less than 1 week with localized interventions; emphasize "
            "multidisciplinary follow-up for associated systemic prematurity issues."
        )
    },

    # ===== Stages 4–5 WITH PLUS =====
    ("zone 1", "plus", "stage 4"): {
        "title": "Zone 1 · Plus · Stage 4 → Treatment",
        "text": (
            "Stage 4 ROP in Zone 1 with Plus disease indicates partial retinal detachment with aggressive vascular tortuosity "
            "and dilation, posing an imminent threat to central vision due to posterior involvement. The presence of Plus "
            "exacerbates ischemia and neovascularization, necessitating urgent surgical intervention beyond standard ablation. "
            "Decision: Treatment within less than 1 week, likely involving vitrectomy or scleral buckling; immediate referral "
            "to a vitreoretinal surgeon is critical, with pre-operative wide-field imaging to map detachment extent and post-op "
            "monitoring for reattachment success."
        )
    },
    ("zone 1", "plus", "stage 5"): {
        "title": "Zone 1 · Plus · Stage 5 → Treatment",
        "text": (
            "Stage 5 ROP in Zone 1 with Plus disease represents total retinal detachment with severe vascular abnormalities, "
            "often leading to a closed-funnel configuration and profound visual loss. The Plus component reflects advanced "
            "ischemia driving cicatricial changes, requiring aggressive salvage attempts despite poor prognosis. Decision: "
            "Treatment within less than 1 week via complex vitreoretinal surgery or enucleation if intractable; urgent specialist "
            "consultation is essential, with psychological support for families and planning for prosthetic fitting if needed."
        )
    },
    ("zone 2", "plus", "stage 4"): {
        "title": "Zone 2 · Plus · Stage 4 → Treatment",
        "text": (
            "Stage 4 ROP in Zone 2 with Plus disease features localized detachment accompanied by Plus-related vascular stress, "
            "increasing the risk of progression to total detachment in a mid-peripheral zone. The aggressive vascular signs "
            "suggest ongoing hypoxia, necessitating rapid surgical correction to preserve remaining vision. Decision: Treatment "
            "within less than 1 week, potentially with scleral buckling or vitrectomy; coordinate with a surgical team, using "
            "intraoperative OCT to guide repair and monitor for post-operative inflammation or recurrence."
        )
    },
    ("zone 2", "plus", "stage 5"): {
        "title": "Zone 2 · Plus · Stage 5 → Treatment",
        "text": (
            "Total Stage 5 detachment in Zone 2 with Plus disease indicates end-stage ROP with extensive cicatricial traction and "
            "vascular abnormalities, severely compromising retinal function. The Plus component reflects persistent ischemia, making "
            "surgical intervention a last resort to alleviate pain or improve cosmesis. Decision: Treatment within less than 1 week "
            "via advanced vitreoretinal procedures or enucleation; involve low-vision specialists early, with detailed pre-op ultrasound "
            "to assess detachment severity and family counseling on outcomes."
        )
    },
    ("zone 3", "plus", "stage 4"): {
        "title": "Zone 3 · Plus · Stage 4 → Treatment",
        "text": (
            "Stage 4 ROP in Zone 3 with Plus disease shows anterior partial detachment with abnormal vascular features, a rare but "
            "serious finding in the peripheral retina that requires intervention to prevent further progression. The Plus signs indicate "
            "ongoing neovascular drive, potentially affecting adjacent zones if untreated. Decision: Treatment within less than 1 week, "
            "likely with localized surgery or laser-assisted repair; urgent evaluation with B-scan ultrasound is needed, followed by "
            "rehabilitation planning for residual vision."
        )
    },
    ("zone 3", "plus", "stage 5"): {
        "title": "Zone 3 · Plus · Stage 5 → Treatment",
        "text": (
            "Stage 5 ROP in Zone 3 with Plus disease presents as total peripheral detachment with aggressive vascular changes, though "
            "less likely to impact central vision due to its anterior location. The Plus component suggests persistent hypoxia, "
            "necessitating intervention despite low visual salvage potential. Decision: Treatment within less than 1 week via targeted "
            "surgical approaches or pain management procedures; specialist assessment with imaging is urgent, with focus on "
            "multidisciplinary care for associated prematurity complications."
        )
    },
}
