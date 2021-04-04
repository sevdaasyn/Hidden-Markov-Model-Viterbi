## HMM(Hidden Markov Model) and Viterbi

NER(Named-entity recognition) tagger using HMMs, which is subtask of information extraction that seeks to
locate and classify named entity mentioned in unstructured text into pre-defined categories such as 
person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc. 

The observations are the words in a given sentence. 
The hidden states are the NER tags of the words, which will be predicted by the model.

Viterbi algorithm will assign the most probable NER tag sequence for a given a word sequence.

### Dataset
The dataset is a CoNLL 2003 NER dataset that is composed of sentences with words
per line 5 possible labels : \LOC, ORG, MISC, PER, O", that represent the NER
classes. The \LOC, ORG, MISC, PER, O" labels are abbreviation of \LOCATION,
ORGANIZAT_ION, MISCELLANEOUS, PERSON, OTHER" respectively.
