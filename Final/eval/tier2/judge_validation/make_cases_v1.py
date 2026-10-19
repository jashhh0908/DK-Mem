import json, sys
from collections import Counter

C = []


def add(cat, role, exp, lang, attr, a, b, note, contested=False):
    C.append(dict(case_id=f"jv_{len(C)+1:03d}", category=cat, role=role, expected=exp, language=lang, attribute_pattern=attr,
                  contested=contested, text_a=a, text_b=b, note=note))


M, K = "merge", "keep_both"
# ---- controls: same fact restated
add("same_fact", "control", M, "en", "same_fact", "I'm allergic to peanuts", "I have a peanut allergy", "same fact, reworded")
add("same_fact", "control", M, "en", "same_fact", "My gym membership renews every January", "The gym membership renews in January every year", "same fact, reworded")
add("same_fact", "control", M, "hi", "same_fact", "Mujhe chai bina cheeni pasand hai", "Main chai bina sugar ke peeta hoon", "same preference, code-mixed rewording")
add("same_fact", "control", M, "hi", "same_fact", "Meri car ki service har teen mahine hoti hai", "Car ki servicing har 3 mahine mein karwata hoon", "same fact about the user's car")
add("same_fact", "control", M, "de", "same_fact", "Ich trinke morgens immer grünen Tee", "Morgens trinke ich grünen Tee", "same habit, reworded")
add("same_fact", "control", M, "mixed", "same_fact", "Mein Fahrrad ist ein rotes Gravelbike", "My bike is a red gravel bike", "same fact in two languages")
# ---- controls: same entity, different compatible facts
add("same_entity_compatible", "control", M, "en", "different_attribute", "My sister Hannah works as a nurse", "Hannah lives in Leeds", "same named sister, two facts")
add("same_entity_compatible", "control", M, "en", "different_attribute", "Our dog Biscuit is three years old", "Biscuit is scared of thunderstorms", "same pet, two facts")
add("same_entity_compatible", "control", M, "en", "different_attribute", "Oliver from my book club is moving in May", "Oliver from my book club has twin daughters", "same person, same identifying context")
add("same_entity_compatible", "control", M, "hi", "different_attribute", "Kavya ka birthday November mein hai", "Kavya Gwalior mein rehti hai", "same name, same spelling, two facts")
add("same_entity_compatible", "control", M, "hi", "different_attribute", "Mera dost Ishaan doctor hai", "Ishaan ki shaadi February mein hai", "same named friend")
add("same_entity_compatible", "control", M, "hi", "different_attribute", "Hamare driver Ramesh Monday ko chhutti lete hain", "Ramesh Gorakhpur ke rehne wale hain", "same named person")
add("same_entity_compatible", "control", M, "de", "different_attribute", "Mein Kollege Moritz spielt Handball", "Moritz hat im Oktober Geburtstag", "same named colleague")
add("same_entity_compatible", "control", M, "de", "different_attribute", "Meine Nachbarin Katrin arbeitet bei der Post", "Katrin hat zwei Katzen", "same named neighbour")
add("same_entity_compatible", "control", M, "de", "different_attribute", "Unser Auto ist ein alter blauer Golf", "Unser Auto muss im März zum TÜV", "the user's one car")
add("same_entity_compatible", "control", M, "mixed", "different_attribute", "Meri roommate Tanvi Bangalore se hai", "Tanvi works at a design studio", "same named person, two languages")
# ---- controls: same entity, conflicting or changed facts
add("same_entity_conflict", "control", M, "en", "same_attribute", "Grace lives in Bristol", "Grace now lives in Cardiff", "same person, changed city")
add("same_entity_conflict", "control", M, "en", "same_attribute", "My yoga class is on Wednesdays", "My yoga class is on Thursdays now", "same class, changed day")
add("same_entity_conflict", "control", M, "en", "same_attribute", "Leo's birthday is on 4 March", "Leo's birthday is on 14 March", "same person, contradictory dates (no update cue)")
add("same_entity_conflict", "control", M, "hi", "same_attribute", "Farhan Prayagraj mein rehta hai", "Farhan ab Dehradun mein rehta hai", "same person, changed city")
add("same_entity_conflict", "control", M, "hi", "same_attribute", "Meher ki flight Friday ko hai", "Meher ki flight Saturday ko hai", "same flight, contradictory day")
add("same_entity_conflict", "control", M, "hi", "same_attribute", "Meri bike ka colour kaala hai", "Meri bike ka colour laal hai", "the user's bike, contradictory colour")
add("same_entity_conflict", "control", M, "de", "same_attribute", "Tobias wohnt in Leipzig", "Tobias wohnt inzwischen in Dresden", "same person, changed city")
add("same_entity_conflict", "control", M, "de", "same_attribute", "Der Elternabend ist am Dienstag", "Der Elternabend wurde auf Donnerstag verschoben", "same event, moved")
add("same_entity_conflict", "control", M, "de", "same_attribute", "Svenja arbeitet bei Siemens", "Svenja arbeitet jetzt bei Bosch", "same person, changed employer")
add("same_entity_conflict", "control", M, "mixed", "same_attribute", "Devika ka office Andheri mein hai", "Devika's office is in Powai now", "same person, changed office, two languages")
# ---- controls: different entities, same attribute, similar wording
add("different_entity_similar", "control", K, "en", "same_attribute", "Sophie is flying to Rome on Friday", "Emily is flying to Lisbon on Friday", "two named people")
add("different_entity_similar", "control", K, "en", "same_attribute", "My colleague Mark lives in Leeds", "My colleague Paul lives in York", "two named colleagues")
add("different_entity_similar", "control", K, "en", "same_attribute", "The kitchen tap is leaking", "The bathroom tap is leaking", "two taps")
add("different_entity_similar", "control", K, "en", "same_attribute", "Anna from work moved to Berlin", "Anna from my choir moved to Potsdam", "same first name, told apart by context (English control of the name+context probe)")
add("different_entity_similar", "control", K, "hi", "same_attribute", "Anika Hubli mein rehti hai", "Ritika Nashik mein rehti hai", "two named people")
add("different_entity_similar", "control", K, "hi", "same_attribute", "Sameer ka interview Tuesday ko hai", "Sanjay ka interview Thursday ko hai", "two named people")
add("different_entity_similar", "control", K, "hi", "same_attribute", "Mere bete ki school trip Shimla ja rahi hai", "Meri beti ki school trip Manali ja rahi hai", "son vs daughter")
add("different_entity_similar", "control", K, "de", "same_attribute", "Herr Brandt fährt nach Rostock", "Herr Albers fährt nach Kiel", "two named people")
add("different_entity_similar", "control", K, "de", "same_attribute", "Mein Bruder wohnt in Ulm", "Meine Schwester wohnt in Passau", "brother vs sister")
add("different_entity_similar", "control", K, "de", "same_attribute", "Die Geburtstagsfeier von Nina ist am Samstag", "Die Geburtstagsfeier von Paula ist am Sonntag", "two people's parties")
add("different_entity_similar", "control", K, "mixed", "same_attribute", "Kunal ka flat Gurgaon mein hai", "Varun's flat is in Noida", "two named people, two languages")
add("different_entity_similar", "control", K, "en", "same_attribute", "My mother's brother lives in Mathura", "My father's brother lives in Ujjain", "two uncles (English control of the kinship probe)")
# ---- controls: different entities, different attributes
add("different_entity_different_attribute", "control", K, "en", "different_attribute", "My cousin Ella is a vet", "My friend Ruby lives in Dublin", "two named people")
add("different_entity_different_attribute", "control", K, "en", "different_attribute", "Jack's birthday is in July", "Tom works night shifts", "two named people")
add("different_entity_different_attribute", "control", K, "en", "different_attribute", "The blue suitcase has a broken wheel", "The green suitcase is in the attic", "two suitcases")
add("different_entity_different_attribute", "control", K, "hi", "different_attribute", "Ananya ka birthday April mein hai", "Tara Chandigarh mein kaam karti hai", "two named people")
add("different_entity_different_attribute", "control", K, "hi", "different_attribute", "Mere boss bahut strict hain", "Mere landlord Kanpur se hain", "boss vs landlord")
add("different_entity_different_attribute", "control", K, "de", "different_attribute", "Lukas hat im Juni Geburtstag", "Felix arbeitet in Augsburg", "two named people")
add("different_entity_different_attribute", "control", K, "de", "different_attribute", "Unser altes Sofa ist grau", "Unser neues Regal kommt am Montag", "two pieces of furniture")
add("different_entity_different_attribute", "control", K, "mixed", "different_attribute", "Rhea ki shaadi December mein hai", "Kiara lives in Goa", "two named people, two languages")
# ---- controls: different entities, same value
add("different_entity_same_value", "control", K, "en", "same_fact", "Mia lives in Glasgow", "Zoe lives in Glasgow", "same fact, two people")
add("different_entity_same_value", "control", K, "hi", "same_fact", "Aditya ka birthday September mein hai", "Nikhil ka birthday September mein hai", "same fact, two people")
add("different_entity_same_value", "control", K, "de", "same_fact", "Mein Bruder wohnt in Ulm", "Meine Schwester wohnt in Ulm", "same fact, two people")
add("different_entity_same_value", "control", K, "mixed", "same_fact", "Kabir works at Infosys", "Rohit bhi Infosys mein kaam karta hai", "same employer, two people")
# ---- controls: only the user is shared
add("user_only", "control", K, "en", "na", "I went for a long run this morning", "I'm planning to repaint the hallway", "two unrelated things the user did or plans")
add("user_only", "control", K, "en", "na", "I finished reading a mystery novel", "I ordered a new phone case", "two unrelated things the user did")
add("user_only", "control", K, "hi", "na", "Aaj office mein bahut meetings thi", "Kal raat maine biryani banayi", "two unrelated things the user did")
add("user_only", "control", K, "hi", "na", "Is weekend main ghar ki safai karunga", "Maine naya backpack khareeda", "two unrelated things the user did or plans")
add("user_only", "control", K, "de", "na", "Heute bin ich mit dem Rad zur Arbeit gefahren", "Ich lerne gerade Spanisch", "two unrelated things the user does")
add("user_only", "control", K, "de", "na", "Gestern habe ich lange geschlafen", "Am Abend koche ich Linsensuppe", "two unrelated things the user did or plans")
# ---- controls: identity not supported (ambiguous) -> keep both
add("ambiguous", "control", K, "en", "different_attribute", "A friend of mine just moved to Leeds", "A friend of mine got a new job", "'a friend' does not identify one person")
add("ambiguous", "control", K, "en", "different_attribute", "My colleague is on holiday this week", "My colleague is moving to Manchester", "'my colleague' does not identify one of many colleagues")
add("ambiguous", "control", K, "en", "different_attribute", "My cousin is getting married in June", "My cousin started university", "'my cousin' does not identify one of several cousins")
add("ambiguous", "control", K, "hi", "different_attribute", "Mera dost aaj ghar aaya", "Mera dost Mysore shift ho gaya", "'mera dost' does not identify one friend")
add("ambiguous", "control", K, "hi", "different_attribute", "Ek colleague ki shaadi hai", "Ek colleague ne job chhod di", "'ek colleague' is indefinite")
add("ambiguous", "control", K, "hi", "different_attribute", "Mere cousin ka result aaya", "Mere cousin Canada ja rahe hain", "'mere cousin' does not identify one cousin")
add("ambiguous", "control", K, "de", "different_attribute", "Ein Freund von mir zieht nach Graz", "Ein Freund von mir hat ein Café eröffnet", "'ein Freund' is indefinite")
add("ambiguous", "control", K, "de", "different_attribute", "Meine Kollegin ist krank", "Meine Kollegin fliegt nach Malta", "'meine Kollegin' does not identify one of many colleagues")
# ---- probes (reported, never gated): the language distinctions Tier 2 studies
add("probe_kinship", "probe", K, "hi", "same_attribute", "Mere mama Mathura mein rehte hain", "Mere chacha Ujjain mein rehte hain", "mother's brother vs father's brother")
add("probe_kinship", "probe", K, "hi", "different_attribute", "Meri mausi ka birthday May mein hai", "Meri bua Agra mein rehti hain", "mother's sister vs father's sister")
add("probe_kinship", "probe", K, "hi", "same_attribute", "Meri bhabhi Monday ko aa rahi hain", "Meri saali Wednesday ko aa rahi hain", "brother's wife vs wife's sister")
add("probe_kinship", "probe", K, "hi", "same_attribute", "Mere tauji ki dukaan Meerut mein hai", "Mere chachaji ki dukaan Aligarh mein hai", "father's elder vs younger brother")
add("probe_register_hi", "probe", M, "hi", "different_attribute", "Didi, tum Ranchi kab ja rahi ho?", "Didi, aap Friday ko free ho?", "same addressee; a register shift is not a different person")
add("probe_register_hi", "probe", M, "hi", "same_attribute", "Chachi, tum Jhansi ja rahi ho na?", "Chachi, aap Raipur ja rahi hain na?", "same addressee, conflicting destination")
add("probe_register_de", "probe", K, "de", "same_attribute", "Jan, kommst du am Montag?", "Jan, kommen Sie am Mittwoch?", "benchmark convention: du vs Sie marks two different addressees", True)
add("probe_register_de", "probe", K, "de", "different_attribute", "Tim, du ziehst nach Kassel", "Tim, Sie haben im Mai Geburtstag", "benchmark convention: du vs Sie marks two different addressees", True)
add("probe_name_script", "probe", M, "hi", "different_attribute", "Ishita ka birthday March mein hai", "इशिता Varanasi mein rehti hai", "same name in two scripts, two facts")
add("probe_name_script", "probe", M, "hi", "same_attribute", "Raghav Patna mein rehta hai", "राघव ab Ranchi mein rehta hai", "same name in two scripts, changed city")
add("probe_name_script", "probe", M, "hi", "same_fact", "Simran yoga sikhati hai", "सिमरन yoga sikhati hai", "same name in two scripts, same fact")
add("probe_name_context", "probe", K, "hi", "same_attribute", "Meri office wali Shreya Goa gayi hai", "Meri college wali Shreya Kochi gayi hai", "same first name, told apart by context")
add("probe_name_context", "probe", K, "hi", "same_attribute", "Gym wala Yash Monday ko aata hai", "Office wala Yash Friday ko aata hai", "same first name, told apart by context")

ids = [c["case_id"] for c in C]
assert len(set(ids)) == len(ids)
assert len({(c["text_a"], c["text_b"]) for c in C}) == len(C)
with open(sys.argv[1], "w", encoding="utf-8", newline="\n") as f:
    for c in C:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
print(len(C), dict(Counter((c["role"], c["expected"]) for c in C)))
print(dict(Counter(c["category"] for c in C)))
