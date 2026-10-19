"""Write cases_v2.jsonl: the labelled validation set for merge_judge_entity_t2_v2 (EVALUATOR side; never sent to Kaggle with labels).

Independent of cases_v1.jsonl: new names, sentences and values (checked by tests/test_tier2_entity.py against v1, both benchmark
arms, the distractor bank and both prompts' examples; only closed classes such as months, weekdays and kinship terms recur).
Same categories and counts as v1, so the v1 acceptance criteria apply unchanged. Usage: python make_cases_v2.py <out.jsonl>
"""
import json
import sys
from collections import Counter

C = []


def add(cat, role, exp, lang, attr, a, b, note, contested=False):
    C.append(dict(case_id=f"jv2_{len(C)+1:03d}", category=cat, role=role, expected=exp, language=lang, attribute_pattern=attr,
                  contested=contested, text_a=a, text_b=b, note=note))


M, K = "merge", "keep_both"
# ---- controls: same fact restated
add("same_fact", "control", M, "en", "same_fact", "I can't eat gluten", "I have to avoid gluten in my food", "same fact, reworded")
add("same_fact", "control", M, "en", "same_fact", "My car insurance is due every October", "The car insurance renews each October", "same fact, reworded")
add("same_fact", "control", M, "hi", "same_fact", "Mujhe subah jaldi uthna pasand hai", "Main subah jaldi uthna pasand karta hoon", "same preference, reworded")
add("same_fact", "control", M, "hi", "same_fact", "Hamara wifi bill har mahine 900 rupaye aata hai", "Wifi ka bill monthly 900 rupaye hai", "same fact about the user's wifi bill")
add("same_fact", "control", M, "de", "same_fact", "Ich gehe jeden Donnerstag schwimmen", "Donnerstags gehe ich immer schwimmen", "same habit, reworded")
add("same_fact", "control", M, "mixed", "same_fact", "Meine Wohnung hat drei Zimmer", "My flat has three rooms", "same fact in two languages")
# ---- controls: same entity, different compatible facts
add("same_entity_compatible", "control", M, "en", "different_attribute", "My brother-in-law Owen is a pilot", "Owen lives in Exeter", "same named person, two facts")
add("same_entity_compatible", "control", M, "en", "different_attribute", "Our cat Pepper is eleven years old", "Pepper only eats wet food", "same pet, two facts")
add("same_entity_compatible", "control", M, "en", "different_attribute", "Ruth from my running club is a dentist", "Ruth from my running club has a son at university", "same person, same identifying context")
add("same_entity_compatible", "control", M, "hi", "different_attribute", "Mitali ka birthday September mein hai", "Mitali Udaipur mein kaam karti hai", "same name, same spelling, two facts")
add("same_entity_compatible", "control", M, "hi", "different_attribute", "Mera dost Kartik engineer hai", "Kartik ki shaadi November mein hai", "same named friend")
add("same_entity_compatible", "control", M, "hi", "different_attribute", "Hamare plumber Suresh har Tuesday aate hain", "Suresh Ajmer ke rehne wale hain", "same named person")
add("same_entity_compatible", "control", M, "de", "different_attribute", "Mein Kollege Florian fährt Motorrad", "Florian hat im April Geburtstag", "same named colleague")
add("same_entity_compatible", "control", M, "de", "different_attribute", "Meine Nachbarin Greta ist Lehrerin", "Greta hat einen Hund", "same named neighbour")
add("same_entity_compatible", "control", M, "de", "different_attribute", "Unser Wohnmobil ist ein weißer Transporter", "Unser Wohnmobil muss im Mai zur Inspektion", "the user's one camper van")
add("same_entity_compatible", "control", M, "mixed", "different_attribute", "Meri cousin Juhi Siliguri se hai", "Juhi works as a lawyer", "same named person, two languages")
# ---- controls: same entity, conflicting or changed facts
add("same_entity_conflict", "control", M, "en", "same_attribute", "Nora lives in Bath", "Nora now lives in Norwich", "same person, changed city")
add("same_entity_conflict", "control", M, "en", "same_attribute", "My piano lesson is on Mondays", "My piano lesson is on Fridays now", "same lesson, changed day")
add("same_entity_conflict", "control", M, "en", "same_attribute", "Ethan's birthday is on 2 July", "Ethan's birthday is on 22 July", "same person, contradictory dates (no update cue)")
add("same_entity_conflict", "control", M, "hi", "same_attribute", "Pranav Kota mein rehta hai", "Pranav ab Bikaner mein rehta hai", "same person, changed city")
add("same_entity_conflict", "control", M, "hi", "same_attribute", "Rohini ki train Wednesday ko hai", "Rohini ki train Thursday ko hai", "same train, contradictory day")
add("same_entity_conflict", "control", M, "hi", "same_attribute", "Mera scooter safed rang ka hai", "Mera scooter neele rang ka hai", "the user's scooter, contradictory colour")
add("same_entity_conflict", "control", M, "de", "same_attribute", "Matthias wohnt in Mainz", "Matthias wohnt inzwischen in Trier", "same person, changed city")
add("same_entity_conflict", "control", M, "de", "same_attribute", "Das Sommerfest ist am Samstag", "Das Sommerfest wurde auf Sonntag verlegt", "same event, moved")
add("same_entity_conflict", "control", M, "de", "same_attribute", "Ulrike arbeitet bei Allianz", "Ulrike arbeitet jetzt bei Lufthansa", "same person, changed employer")
add("same_entity_conflict", "control", M, "mixed", "same_attribute", "Nandini ka clinic Thrissur mein hai", "Nandini's clinic is in Vellore now", "same person, changed clinic, two languages")
# ---- controls: different entities, same attribute, similar wording
add("different_entity_similar", "control", K, "en", "same_attribute", "Lily is flying to Madrid on Tuesday", "Daisy is flying to Prague on Tuesday", "two named people")
add("different_entity_similar", "control", K, "en", "same_attribute", "My colleague Callum lives in Durham", "My colleague Rory lives in Aberdeen", "two named colleagues")
add("different_entity_similar", "control", K, "en", "same_attribute", "The front door lock is broken", "The back door lock is broken", "two locks")
add("different_entity_similar", "control", K, "en", "same_attribute", "Isla from the office moved to Dundee", "Isla from my pottery class moved to Perth", "same first name, told apart by context (English control of the name+context probe)")
add("different_entity_similar", "control", K, "hi", "same_attribute", "Bhavna Belgaum mein rehti hai", "Shalini Jalandhar mein rehti hai", "two named people")
add("different_entity_similar", "control", K, "hi", "same_attribute", "Harsh ka exam Monday ko hai", "Mohit ka exam Friday ko hai", "two named people")
add("different_entity_similar", "control", K, "hi", "same_attribute", "Meri badi beti ka hostel Pilani mein hai", "Meri chhoti beti ka hostel Manipal mein hai", "elder vs younger daughter")
add("different_entity_similar", "control", K, "de", "same_attribute", "Frau Lorenz fährt nach Erfurt", "Frau Becker fährt nach Bamberg", "two named people")
add("different_entity_similar", "control", K, "de", "same_attribute", "Mein Onkel wohnt in Lübeck", "Meine Tante wohnt in Regensburg", "uncle vs aunt")
add("different_entity_similar", "control", K, "de", "same_attribute", "Die Hochzeit von Leonie ist im Juni", "Die Hochzeit von Carola ist im Juli", "two people's weddings")
add("different_entity_similar", "control", K, "mixed", "same_attribute", "Dhruv ka office Baroda mein hai", "Tushar's office is in Rajkot", "two named people, two languages")
add("different_entity_similar", "control", K, "en", "same_attribute", "My mother's sister lives in Chester", "My father's sister lives in Hull", "two aunts (English control of the kinship probe)")
# ---- controls: different entities, different attributes
add("different_entity_different_attribute", "control", K, "en", "different_attribute", "My nephew Jude plays the violin", "My friend Amber lives in Brighton", "two named people")
add("different_entity_different_attribute", "control", K, "en", "different_attribute", "Finn's wedding is in March", "Seth works at a bakery", "two named people")
add("different_entity_different_attribute", "control", K, "en", "different_attribute", "The red umbrella has a broken handle", "The black umbrella is in the car", "two umbrellas")
add("different_entity_different_attribute", "control", K, "hi", "different_attribute", "Reema ka birthday February mein hai", "Sonal Vadodara mein padhati hai", "two named people")
add("different_entity_different_attribute", "control", K, "hi", "different_attribute", "Mere manager bahut busy rehte hain", "Hamare makaan-maalik Ludhiana se hain", "manager vs landlord")
add("different_entity_different_attribute", "control", K, "de", "different_attribute", "Henrik hat im März Geburtstag", "Konstantin arbeitet in Würzburg", "two named people")
add("different_entity_different_attribute", "control", K, "de", "different_attribute", "Unser alter Tisch ist aus Eiche", "Unser neuer Schrank kommt am Dienstag", "two pieces of furniture")
add("different_entity_different_attribute", "control", K, "mixed", "different_attribute", "Lavanya ki engagement January mein hai", "Mira lives in Pondicherry", "two named people, two languages")
# ---- controls: different entities, same value
add("different_entity_same_value", "control", K, "en", "same_fact", "Molly lives in Cambridge", "Freya lives in Cambridge", "same fact, two people")
add("different_entity_same_value", "control", K, "hi", "same_fact", "Gaurav ka birthday December mein hai", "Aakash ka birthday December mein hai", "same fact, two people")
add("different_entity_same_value", "control", K, "de", "same_fact", "Mein Cousin wohnt in Freiburg", "Meine Cousine wohnt in Freiburg", "same fact, two people")
add("different_entity_same_value", "control", K, "mixed", "same_fact", "Siddharth works at Wipro", "Dennis bhi Wipro mein kaam karta hai", "same employer, two people")
# ---- controls: only the user is shared
add("user_only", "control", K, "en", "na", "I cleaned out the garage yesterday", "I'm thinking of learning to sew", "two unrelated things the user did or plans")
add("user_only", "control", K, "en", "na", "I watched a documentary about whales", "I bought new running shoes", "two unrelated things the user did")
add("user_only", "control", K, "hi", "na", "Aaj maine bahut der tak padhai ki", "Kal main bank gaya tha", "two unrelated things the user did")
add("user_only", "control", K, "hi", "na", "Is mahine main paise bachane ki koshish kar raha hoon", "Maine naya chashma banwaya", "two unrelated things the user does or did")
add("user_only", "control", K, "de", "na", "Heute habe ich den Keller aufgeräumt", "Ich höre gerade einen Krimi als Hörbuch", "two unrelated things the user does")
add("user_only", "control", K, "de", "na", "Letzte Woche war ich beim Friseur", "Morgen backe ich einen Apfelkuchen", "two unrelated things the user did or plans")
# ---- controls: identity not supported (ambiguous) -> keep both
add("ambiguous", "control", K, "en", "different_attribute", "Someone from my team is leaving next month", "Someone from my team got engaged", "'someone from my team' does not identify one person")
add("ambiguous", "control", K, "en", "different_attribute", "My neighbour is redoing the kitchen", "My neighbour is selling a car", "'my neighbour' does not identify one of several neighbours")
add("ambiguous", "control", K, "en", "different_attribute", "My classmate is moving abroad", "My classmate has a new baby", "'my classmate' does not identify one of many classmates")
add("ambiguous", "control", K, "hi", "different_attribute", "Meri ek saheli ne naya ghar liya", "Meri ek saheli ki naukri chali gayi", "'meri ek saheli' is indefinite")
add("ambiguous", "control", K, "hi", "different_attribute", "Mere ek rishtedaar aaj aaye the", "Mere ek rishtedaar hospital mein hain", "'mere ek rishtedaar' is indefinite")
add("ambiguous", "control", K, "hi", "different_attribute", "Mere padosi ne gaadi bechi", "Mere padosi ki beti ki shaadi hai", "'mere padosi' does not identify one neighbour")
add("ambiguous", "control", K, "de", "different_attribute", "Eine Bekannte von mir macht ein Praktikum", "Eine Bekannte von mir hat geheiratet", "'eine Bekannte' is indefinite")
add("ambiguous", "control", K, "de", "different_attribute", "Ein Kollege von mir ist im Urlaub", "Ein Kollege von mir sucht eine neue Stelle", "'ein Kollege' is indefinite")
# ---- probes (reported, never gated)
add("probe_kinship", "probe", K, "hi", "same_attribute", "Mere mausaji Alwar mein rehte hain", "Mere fufaji Sikar mein rehte hain", "mother's sister's husband vs father's sister's husband")
add("probe_kinship", "probe", K, "hi", "different_attribute", "Meri nanad ka birthday June mein hai", "Meri devrani Satara mein rehti hai", "husband's sister vs husband's younger brother's wife")
add("probe_kinship", "probe", K, "hi", "same_attribute", "Mera bhatija Saturday ko aa raha hai", "Mera bhanja Sunday ko aa raha hai", "brother's son vs sister's son")
add("probe_kinship", "probe", K, "hi", "same_attribute", "Meri taiji ka ghar Rewa mein hai", "Meri bhabhiji ka ghar Satna mein hai", "father's elder brother's wife vs brother's wife")
add("probe_register_hi", "probe", M, "hi", "different_attribute", "Bhaiya, tum Thursday ko aa rahe ho?", "Bhaiya, aap Mandi kab jaoge?", "same addressee; a register shift is not a different person")
add("probe_register_hi", "probe", M, "hi", "same_attribute", "Mamaji, tum Solan ja rahe ho na?", "Mamaji, aap Kullu ja rahe hain na?", "same addressee, conflicting destination")
add("probe_register_de", "probe", K, "de", "same_attribute", "Lars, fährst du am Freitag?", "Lars, fahren Sie am Dienstag?", "benchmark convention: du vs Sie marks two different addressees", True)
add("probe_register_de", "probe", K, "de", "different_attribute", "Eva, du arbeitest in Gera", "Eva, Sie haben im Oktober Geburtstag", "benchmark convention: du vs Sie marks two different addressees", True)
add("probe_name_script", "probe", M, "hi", "different_attribute", "Tanmay ka birthday May mein hai", "तन्मय Hisar mein rehta hai", "same name in two scripts, two facts")
add("probe_name_script", "probe", M, "hi", "same_attribute", "Ritu Bareilly mein rehti hai", "रितु ab Jhunjhunu mein rehti hai", "same name in two scripts, changed city")
add("probe_name_script", "probe", M, "hi", "same_fact", "Manish guitar bajata hai", "मनीष guitar bajata hai", "same name in two scripts, same fact")
add("probe_name_context", "probe", K, "hi", "same_attribute", "Meri tuition wali Payal Kurnool gayi hai", "Meri society wali Payal Guntur gayi hai", "same first name, told apart by context")
add("probe_name_context", "probe", K, "hi", "same_attribute", "Cricket wala Naveen Tuesday ko aata hai", "Library wala Naveen Saturday ko aata hai", "same first name, told apart by context")

ids = [c["case_id"] for c in C]
assert len(set(ids)) == len(ids) and len({(c["text_a"], c["text_b"]) for c in C}) == len(C)
if __name__ == "__main__":
    with open(sys.argv[1], "w", encoding="utf-8", newline="\n") as f:
        for c in C:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(len(C), dict(Counter((c["role"], c["expected"]) for c in C)))
    print(dict(Counter(c["category"] for c in C)))
