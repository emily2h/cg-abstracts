import scispacy
from scispacy.abbreviation import AbbreviationDetector
import spacy

nlp = spacy.load("en_ner_bionlp13cg_md")

abbreviation_pipe = AbbreviationDetector(nlp)

nlp.add_pipe(abbreviation_pipe)

text = "Conducted an in vitro analysis of the cell."


doc = nlp(text)

#print("sentences?:",list(doc.sents))

print("entities?",doc.ents)

print("Abbreviation", "\t", "Definition")
for abrv in doc._.abbreviations:
	print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
        
