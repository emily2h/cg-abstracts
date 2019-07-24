import scispacy
from scispacy.abbreviation import AbbreviationDetector
import spacy

nlp = spacy.load("en_ner_bionlp13cg_md")

abbreviation_pipe = AbbreviationDetector(nlp)

nlp.add_pipe(abbreviation_pipe)

text = """The mutant genotype is the main determinant of the metabolic phenotype in phenylalanine hydroxylase deficiency. || Alleles Amino Acid Substitution Animals COS Cells Gene Expression Regulation, Enzymologic Genotype Heterozygote Homozygote Humans Mutation Phenotype Phenylalanine Hydroxylase Phenylketonurias || Phenylketonuria and mild hyperphenylalaninemias are allelic disorders caused by mutations in the phenylalanine hydroxylase (PAH) gene. Following identification of the disease-causing mutation in 11 PAH-deficient patients, we tested the activity of the mutant gene products in an eukaryotic expression system. Two mutations markedly reduced PAH activity (A259V and L333F), one mutation mildly altered the enzyme activity (E390G), while the majority of mutant genotypes reduced the in vitro expression of PAH activity to 15-30% of controls. Comparing the predicted residual activity derived from expression studies to the clinical phenotypes of our PAH-deficient patients, we found that homozygosity for the L333F and E390G mutations resulted in severe and mild PAH deficiencies, respectively, both in vivo and in vitro, while compound heterozygosity (L333F/E390G) resulted in an intermediate dietary tolerance. Similarly, in vitro expression studies largely predicted dietary tolerance in compound heterozygotes for the A259V/IVS12nt1 (typical PKU), A259V/A403V, G218V/I65T, and G218V/R158Q mutations (mild variants). Taken together, these results support the view that expression studies are useful in predicting residual enzyme activity and that the mutant genotype at the PAH locus is the major determinant of metabolic phenotype in hyperphenylalaninemias.
"""


doc = nlp(text)

#print("sentences?:",list(doc.sents))

print("entities?",doc.ents)

print("Abbreviation", "\t", "Definition")
for abrv in doc._.abbreviations:
	print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
        
