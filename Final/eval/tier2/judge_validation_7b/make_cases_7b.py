"""Write cases_7b.jsonl: the labelled validation set for merge_judge_entity_t2_v2 on Qwen2.5-7B (EVALUATOR side; texts only reach Kaggle).

Third independent set: new names, sentences and values, disjoint from cases_v1, cases_v2, both benchmark arms, the distractor bank and
both identity prompts' examples (checked by tests/test_tier2_entity.py; only closed classes recur). Same categories and counts as v1/v2,
so the unchanged criteria apply. Usage: python make_cases_7b.py <out.jsonl>
"""
import json
import sys
from collections import Counter

C = []


def add(cat, role, exp, lang, attr, a, b, note, contested=False):
    C.append(dict(case_id=f"jv7_{len(C)+1:03d}", category=cat, role=role, expected=exp, language=lang, attribute_pattern=attr,
                  contested=contested, text_a=a, text_b=b, note=note))


M, K = "merge", "keep_both"
# ---- controls: same fact restated
add("same_fact", "control", M, "en", "same_fact", "I'm lactose intolerant", "Dairy makes me ill because I'm lactose intolerant", "same fact, reworded")
add("same_fact", "control", M, "en", "same_fact", "Our bins are collected every Wednesday", "The bin collection is on Wednesdays", "same fact, reworded")
add("same_fact", "control", M, "hi", "same_fact", "Mujhe raat ko jaldi sona achha lagta hai", "Main raat ko jaldi so jaata hoon, mujhe achha lagta hai", "same habit, reworded")
add("same_fact", "control", M, "hi", "same_fact", "Hamare ghar ka kiraya 15000 rupaye hai", "Ghar ka rent 15000 rupaye mahina hai", "same fact about the rent")
add("same_fact", "control", M, "de", "same_fact", "Ich fahre jeden Morgen mit der S-Bahn", "Morgens nehme ich immer die S-Bahn", "same habit, reworded")
add("same_fact", "control", M, "mixed", "same_fact", "Mein Balkon zeigt nach Süden", "My balcony faces south", "same fact in two languages")
# ---- controls: same entity, different compatible facts
add("same_entity_compatible", "control", M, "en", "different_attribute", "My stepbrother Gideon is an electrician", "Gideon lives in Truro", "same named person, two facts")
add("same_entity_compatible", "control", M, "en", "different_attribute", "Our tortoise Basil is forty years old", "Basil sleeps through the winter", "same pet, two facts")
add("same_entity_compatible", "control", M, "en", "different_attribute", "Martha from my choir is a midwife", "Martha from my choir has three grandchildren", "same person, same identifying context")
add("same_entity_compatible", "control", M, "hi", "different_attribute", "Kritika ka birthday August mein hai", "Kritika Mangalore mein padhati hai", "same name, same spelling, two facts")
add("same_entity_compatible", "control", M, "hi", "different_attribute", "Mera dost Omkar photographer hai", "Omkar ki engagement March mein hai", "same named friend")
add("same_entity_compatible", "control", M, "hi", "different_attribute", "Hamare electrician Hemant har Saturday aate hain", "Hemant Rohtak ke rehne wale hain", "same named person")
add("same_entity_compatible", "control", M, "de", "different_attribute", "Mein Kollege Jannik spielt Schach", "Jannik hat im Juli Geburtstag", "same named colleague")
add("same_entity_compatible", "control", M, "de", "different_attribute", "Meine Nachbarin Heike ist Ärztin", "Heike hat zwei Pferde", "same named neighbour")
add("same_entity_compatible", "control", M, "de", "different_attribute", "Unser Segelboot ist ein altes Holzboot", "Unser Segelboot liegt im Winter in der Halle", "the user's one sailing boat")
add("same_entity_compatible", "control", M, "mixed", "different_attribute", "Meri bhabhi Charu Nellore se hai", "Charu works as an architect", "same named person, two languages")
# ---- controls: same entity, conflicting or changed facts
add("same_entity_conflict", "control", M, "en", "same_attribute", "Hazel lives in Lancaster", "Hazel now lives in Carlisle", "same person, changed city")
add("same_entity_conflict", "control", M, "en", "same_attribute", "My swimming class is on Tuesdays", "My swimming class is on Saturdays now", "same class, changed day")
add("same_entity_conflict", "control", M, "en", "same_attribute", "Arthur's birthday is on 9 May", "Arthur's birthday is on 19 May", "same person, contradictory dates (no update cue)")
add("same_entity_conflict", "control", M, "hi", "same_attribute", "Vedant Hosur mein rehta hai", "Vedant ab Tirupati mein rehta hai", "same person, changed city")
add("same_entity_conflict", "control", M, "hi", "same_attribute", "Ishani ki flight Monday ko hai", "Ishani ki flight Tuesday ko hai", "same flight, contradictory day")
add("same_entity_conflict", "control", M, "hi", "same_attribute", "Meri cycle hare rang ki hai", "Meri cycle peele rang ki hai", "the user's bicycle, contradictory colour")
add("same_entity_conflict", "control", M, "de", "same_attribute", "Wolfgang wohnt in Koblenz", "Wolfgang wohnt inzwischen in Fulda", "same person, changed city")
add("same_entity_conflict", "control", M, "de", "same_attribute", "Der Flohmarkt ist am Freitag", "Der Flohmarkt wurde auf Samstag verschoben", "same event, moved")
add("same_entity_conflict", "control", M, "de", "same_attribute", "Bettina arbeitet bei Henkel", "Bettina arbeitet jetzt bei Infineon", "same person, changed employer")
add("same_entity_conflict", "control", M, "mixed", "same_attribute", "Swati ka studio Karnal mein hai", "Swati's studio is in Panipat now", "same person, changed studio, two languages")
# ---- controls: different entities, same attribute, similar wording
add("different_entity_similar", "control", K, "en", "same_attribute", "Poppy is driving to Bangor on Sunday", "Esme is driving to Stirling on Sunday", "two named people")
add("different_entity_similar", "control", K, "en", "same_attribute", "My colleague Toby lives in Hereford", "My colleague Rufus lives in Ipswich", "two named colleagues")
add("different_entity_similar", "control", K, "en", "same_attribute", "The upstairs radiator is cold", "The downstairs radiator is cold", "two radiators")
add("different_entity_similar", "control", K, "en", "same_attribute", "Wren from the library moved to Wells", "Wren from my climbing gym moved to Bangor", "same first name, told apart by context (English control of the name+context probe)")
add("different_entity_similar", "control", K, "hi", "same_attribute", "Pallavi Durg mein rehti hai", "Trisha Bokaro mein rehti hai", "two named people")
add("different_entity_similar", "control", K, "hi", "same_attribute", "Rajat ka interview Wednesday ko hai", "Abhay ka interview Saturday ko hai", "two named people")
add("different_entity_similar", "control", K, "hi", "same_attribute", "Mere bade bhai ki dukaan Jamnagar mein hai", "Mere chhote bhai ki dukaan Bhuj mein hai", "elder vs younger brother")
add("different_entity_similar", "control", K, "de", "same_attribute", "Herr Vogel fährt nach Kempten", "Herr Kraus fährt nach Siegen", "two named people")
add("different_entity_similar", "control", K, "de", "same_attribute", "Mein Opa wohnt in Celle", "Meine Oma wohnt in Wetzlar", "grandfather vs grandmother")
add("different_entity_similar", "control", K, "de", "same_attribute", "Die Taufe von Marlene ist im August", "Die Taufe von Johanna ist im September", "two people's christenings")
add("different_entity_similar", "control", K, "mixed", "same_attribute", "Chirag ka godown Kakinada mein hai", "Ketan's warehouse is in Haldwani", "two named people, two languages")
add("different_entity_similar", "control", K, "en", "same_attribute", "My mother's father lives in Lancaster", "My father's father lives in Truro", "two grandfathers (English control of the kinship probe)")
# ---- controls: different entities, different attributes
add("different_entity_different_attribute", "control", K, "en", "different_attribute", "My godson Theo plays rugby", "My friend Imogen lives in Bangor", "two named people")
add("different_entity_different_attribute", "control", K, "en", "different_attribute", "Harriet's retirement party is in April", "Stanley works at a garage", "two named people")
add("different_entity_different_attribute", "control", K, "en", "different_attribute", "The silver kettle has a loose lid", "The white kettle is in the cupboard", "two kettles")
add("different_entity_different_attribute", "control", K, "hi", "different_attribute", "Varsha ka birthday July mein hai", "Nikita Muzaffarpur mein naukri karti hai", "two named people")
add("different_entity_different_attribute", "control", K, "hi", "different_attribute", "Mere senior bahut madad karte hain", "Hamare driver sahab Haldwani se hain", "senior vs driver")
add("different_entity_different_attribute", "control", K, "de", "different_attribute", "Torsten hat im Mai Geburtstag", "Rüdiger arbeitet in Paderborn", "two named people")
add("different_entity_different_attribute", "control", K, "de", "different_attribute", "Unsere alte Lampe ist aus Messing", "Unser neuer Teppich kommt am Mittwoch", "two pieces of furnishing")
add("different_entity_different_attribute", "control", K, "mixed", "different_attribute", "Aditi ki housewarming November mein hai", "Yuvraj lives in Jamnagar", "two named people, two languages")
# ---- controls: different entities, same value
add("different_entity_same_value", "control", K, "en", "same_fact", "Florence lives in Hereford", "Beatrice lives in Hereford", "same fact, two people")
add("different_entity_same_value", "control", K, "hi", "same_fact", "Ketan ka birthday March mein hai", "Chirag ka birthday March mein hai", "same fact, two people")
add("different_entity_same_value", "control", K, "de", "same_fact", "Mein Schwager wohnt in Göttingen", "Meine Schwägerin wohnt in Göttingen", "same fact, two people")
add("different_entity_same_value", "control", K, "mixed", "same_fact", "Trisha works at Mahindra", "Kai bhi Mahindra mein kaam karta hai", "same employer, two people")
# ---- controls: only the user is shared
add("user_only", "control", K, "en", "na", "I repaired the garden fence on Saturday", "I'm thinking of joining a book club", "two unrelated things the user did or plans")
add("user_only", "control", K, "en", "na", "I listened to a podcast about volcanoes", "I ordered a new desk lamp", "two unrelated things the user did")
add("user_only", "control", K, "hi", "na", "Aaj maine poora din coding ki", "Kal main post office gaya tha", "two unrelated things the user did")
add("user_only", "control", K, "hi", "na", "Is saal main marathon daudna chahta hoon", "Maine naya mattress khareeda", "two unrelated things the user plans or did")
add("user_only", "control", K, "de", "na", "Heute habe ich die Fenster geputzt", "Ich lese gerade einen Roman über Seefahrer", "two unrelated things the user does")
add("user_only", "control", K, "de", "na", "Am Wochenende war ich wandern", "Nächste Woche streiche ich das Bad", "two unrelated things the user did or plans")
# ---- controls: identity not supported (ambiguous) -> keep both
add("ambiguous", "control", K, "en", "different_attribute", "Somebody at my gym is training for a triathlon", "Somebody at my gym just had twins", "'somebody at my gym' does not identify one person")
add("ambiguous", "control", K, "en", "different_attribute", "A relative of mine is visiting from abroad", "A relative of mine is in hospital", "'a relative' is indefinite")
add("ambiguous", "control", K, "en", "different_attribute", "One of my students won a prize", "One of my students is moving schools", "'one of my students' is indefinite")
add("ambiguous", "control", K, "hi", "different_attribute", "Mere ek dost ne car li", "Mere ek dost ki tabiyat kharab hai", "'mere ek dost' is indefinite")
add("ambiguous", "control", K, "hi", "different_attribute", "Office mein kisi ka promotion hua", "Office mein kisi ne resign kiya", "'kisi' is indefinite")
add("ambiguous", "control", K, "hi", "different_attribute", "Meri ek cousin ki shaadi hai", "Meri ek cousin abroad padh rahi hai", "'meri ek cousin' is indefinite")
add("ambiguous", "control", K, "de", "different_attribute", "Ein Nachbar von mir renoviert das Dach", "Ein Nachbar von mir hat einen Unfall gehabt", "'ein Nachbar' is indefinite")
add("ambiguous", "control", K, "de", "different_attribute", "Eine Freundin von mir zieht ins Ausland", "Eine Freundin von mir hat ein Kind bekommen", "'eine Freundin' is indefinite")
# ---- probes (reported, never gated)
add("probe_kinship", "probe", K, "hi", "same_attribute", "Mere nanaji Haridwar mein rehte hain", "Mere dadaji Roorkee mein rehte hain", "mother's father vs father's father")
add("probe_kinship", "probe", K, "hi", "different_attribute", "Meri jethani ka birthday April mein hai", "Meri bahu Hapur mein rehti hai", "husband's elder brother's wife vs son's wife")
add("probe_kinship", "probe", K, "hi", "same_attribute", "Mera saala Wednesday ko aa raha hai", "Mera jija Friday ko aa raha hai", "wife's brother vs sister's husband")
add("probe_kinship", "probe", K, "hi", "same_attribute", "Meri nanadji ka ghar Etawah mein hai", "Meri saas ka ghar Mainpuri mein hai", "husband's sister vs mother-in-law")
add("probe_register_hi", "probe", M, "hi", "different_attribute", "Didi, tum Sunday ko free ho?", "Didi, aap Hapur kab jaogi?", "same addressee; a register shift is not a different person")
add("probe_register_hi", "probe", M, "hi", "same_attribute", "Mausi, tum Etawah ja rahi ho na?", "Mausi, aap Mainpuri ja rahi hain na?", "same addressee, conflicting destination")
add("probe_register_de", "probe", K, "de", "same_attribute", "Nils, kommst du am Montag?", "Nils, kommen Sie am Donnerstag?", "benchmark convention: du vs Sie marks two different addressees", True)
add("probe_register_de", "probe", K, "de", "different_attribute", "Ute, du wohnst in Siegen", "Ute, Sie haben im Juni Geburtstag", "benchmark convention: du vs Sie marks two different addressees", True)
add("probe_name_script", "probe", M, "hi", "different_attribute", "Tanish ka birthday June mein hai", "तनिश Roorkee mein rehta hai", "same name in two scripts, two facts")
add("probe_name_script", "probe", M, "hi", "same_attribute", "Meera Hapur mein rehti hai", "मीरा ab Etawah mein rehti hai", "same name in two scripts, changed city")
add("probe_name_script", "probe", M, "hi", "same_fact", "Rohan tabla bajata hai", "रोहन tabla bajata hai", "same name in two scripts, same fact")
add("probe_name_context", "probe", K, "hi", "same_attribute", "Meri yoga wali Sapna Haridwar gayi hai", "Meri bank wali Sapna Roorkee gayi hai", "same first name, told apart by context")
add("probe_name_context", "probe", K, "hi", "same_attribute", "Badminton wala Ankit Monday ko aata hai", "Tuition wala Ankit Thursday ko aata hai", "same first name, told apart by context")

ids = [c["case_id"] for c in C]
assert len(set(ids)) == len(ids) and len({(c["text_a"], c["text_b"]) for c in C}) == len(C)
if __name__ == "__main__":
    with open(sys.argv[1], "w", encoding="utf-8", newline="\n") as f:
        for c in C:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(len(C), dict(Counter((c["role"], c["expected"]) for c in C)))
