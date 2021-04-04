import numpy as np
import string

DATA_PATH = "dataset/train.txt"
TEST_PATH = "dataset/test.txt"


np.set_printoptions(precision=4)

class MY_HMM:


	def dataset(self, path):
		#word_tag_list is a two dimensional array that holds words and their corresponding tags
		word_tag_list = [] # [['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'],...]]
		#sentence_tags holds tag sequences for every sentence
		sentence_tags = [] # ['<s> B-ORG O B-MISC O O O B-MISC O O </s>', '<s> B-PER I-PER </s>', ...]
		sent = "<s> "

		file = open(path, "r")
		i=0
		for line in file:
			i+=1
			if(i==1 or i==2):
				continue # to pass unintended lines

			if(line != '\n' ):
				items = line.split()
				word_tag_list.append([items[0], items[3]])
				sent+=items[3] + " "
			else:
				sent += "</s>" 
				sentence_tags.append(sent)
				sent = "<s> "

		return sentence_tags, word_tag_list


	def HMM(self):
		sentence_tags, word_tag_list = self.dataset(DATA_PATH)
		#trans_probs ans emis_probs are nested dictionaries
		#trans_probs holds tags and count of its fallowing tags
		#emis_probs hols holds tags ans words that used with this tag
		trans_probs = {} #{"B-LOC" : {'O': 5943, 'I-LOC': 1041, '</s>': 110, 'B-ORG': 10, 'B-MISC': 27, 'B-PER': 1, 'B-LOC': 8}, "O" :{...}}
		emis_probs = {} #{"B-LOC" : {'brussels': 31, 'germany': 142, 'britain': 134, 'france': 123,...}, "O" :{...}}
		#tag_list is like set, includes tag types and word boundaries
		tag_list = []
		tag_list.append("<s>")

		for sent in sentence_tags:
			tags = sent.split()
			for i in range( len(tags)-1 ):
				if(tags[i] not in trans_probs.keys()):
					trans_probs[tags[i]] = {tags[i+1] : 1}
				else:
					if(tags[i+1] not in trans_probs[tags[i]].keys()):
						trans_probs[tags[i]][tags[i+1]] = 1
					else:
						trans_probs[tags[i]][tags[i+1]] += 1


		for word_tag in  word_tag_list:
			word_tag[0] = word_tag[0].lower()
			if(word_tag[1] not in emis_probs):
				emis_probs[word_tag[1]] = {word_tag[0] : 1}
			else:
				if(word_tag[0] not in emis_probs[word_tag[1]].keys()):
					emis_probs[word_tag[1]][word_tag[0]] = 1
				else:
					emis_probs[word_tag[1]][word_tag[0]] += 1
			if (word_tag[1] not in tag_list):
				tag_list.append(word_tag[1])

		tag_list.append("</s>")

		return tag_list, trans_probs, emis_probs



mm = MY_HMM()

def viterbi():
	tag_list, trans_probs, emis_probs = mm.HMM()

	number_of_tag = len(emis_probs)
	# predicted_tags is 2d array, it includes predicted tag sequences for every sentence
	predicted_tags = [] #[['<s>', 'O', 'O', 'B-LOC', 'O', 'I-LOC', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', '</s>'], ['<s>', 'B',...], ...]

	file = open(TEST_PATH, "r")
	sent = ""
	for line in file:

		if(line != '\n' and line.split()[0] != "-DOCSTART-"):
			sent += line.split()[0] + " "

		elif(sent != "" and line=='\n'):

			sent = "<s> " + sent.lower() + "</s>" 
			words = sent.split()


			'''
			matrix is for holding probabilities for every word and every tag
			it has w+2 columns and tag+2 rows
			first column corresponds to start(<s>) and similarly last column corresponds to end(</s>)
			each row represents a tag by order in the tag_list
			an example final matrix for a sentence that has two words is stated below 
			[[1.0000e+00 0.0000e+00 0.0000e+00 0.0000e+00]
 			 [0.0000e+00 2.5875e-05 2.6759e-09 0.0000e+00]
 			 [0.0000e+00 7.6156e-04 4.5245e-06 0.0000e+00]
 			 [0.0000e+00 9.7315e-06 3.7079e-09 0.0000e+00]
 			 [0.0000e+00 1.3622e-05 3.5029e-09 0.0000e+00]
 			 [0.0000e+00 1.4699e-08 1.9467e-09 0.0000e+00]
 			 [0.0000e+00 1.4758e-05 3.4569e-09 0.0000e+00]
 			 [0.0000e+00 3.5922e-08 2.7368e-09 0.0000e+00]
 			 [0.0000e+00 5.7293e-08 2.0834e-09 0.0000e+00]
 			 [0.0000e+00 5.7195e-08 1.8448e-09 0.0000e+00]
 			 [0.0000e+00 0.0000e+00 0.0000e+00 3.7858e-07]]
 			'''
			matrix = np.zeros( (number_of_tag+2, len(words) ) )
			matrix[0][0] = 1


			for w in range( 1, len(words)-1 ):
				for i in range( 1, len(tag_list) -1):
					emision_probability = 0
					transition_probability = 0
					final_probability = 0

					token = words[w]
					curr_tag = tag_list[i]

					
					if(token in emis_probs[curr_tag].keys()):
						emision_probability = (emis_probs[curr_tag][token] +1) / (sum(emis_probs[curr_tag].values()) + number_of_tag)
					else:
						emision_probability = 1 / (sum(emis_probs[curr_tag].values()) + number_of_tag)


					# for first token, do not consider previous probabilities
					if(token == words[1]):	
						if("<s>" in trans_probs.keys()):
							if(curr_tag in trans_probs["<s>"].keys()):
								transition_probability = (trans_probs["<s>"][curr_tag] +1) / (sum(trans_probs["<s>"].values()) + number_of_tag)
							else:
								transition_probability = (0 +1) / (sum(trans_probs["<s>"].values()) + number_of_tag)

						matrix[i][w] = transition_probability * emision_probability

					# when calculating second and after tokens probabilities
					# scan all tags for previous token and find maximum probability
					else:
						for t in range(1, len(tag_list)-1):
							query_tag = tag_list[t]
							if(query_tag in trans_probs.keys()):
								if(curr_tag in trans_probs[query_tag].keys()):
									transition_probability = (trans_probs[query_tag][curr_tag] +1 )/ (sum(trans_probs[query_tag].values()) + number_of_tag)
								else:
									transition_probability = (0 +1 )/ (sum(trans_probs[query_tag].values()) + number_of_tag)
							
							final_probability = transition_probability * emision_probability * matrix[t][w-1]


							if(final_probability > matrix[i][w]):
								matrix[i][w] = final_probability



					# if tokens are finished, fill end of sentence probability 
					if(w==len(words)-2  and i==len(tag_list)-2 ):
						for t in range(1, len(tag_list)-1):
							query_tag = tag_list[t]
							if(query_tag in trans_probs.keys()):
								if("</s>" in trans_probs[query_tag].keys()):
									transition_probability = (trans_probs[query_tag]["</s>"] +1) / (sum(trans_probs[query_tag].values()) + number_of_tag)
								else:
									transition_probability = (0 +1) / (sum(trans_probs[query_tag].values()) + number_of_tag)
							final_probability = matrix[t][w] * transition_probability

							if(final_probability > matrix[i+1][w+1]):
								matrix[i+1][w+1] = final_probability
							
					
				sent = ""

			# after obtaining final matrix, backtrace and extract predicted tags
			result = np.amax(matrix, axis=0)
			predicted_tags_word = []
			for maxval in result:
				r , c = np.where(matrix == maxval)
				predicted_tags_word.append( tag_list[r[0]] )
				#print("r,c = ", r[0], ",", c[0])

			predicted_tags.append(predicted_tags_word)
	return predicted_tags
			


def accuracy():
	predicted_sequence = viterbi()
	# gold_sequence is 2d array, it includes gold tag sequences for every sentence
	gold_sequence = [] #[['<s>', 'O', 'O', 'B-LOC', ...], ['<s>', 'B-PER',...], ...]
	
	file = open(TEST_PATH, "r")
	
	seq = ["<s>"]
	i=0
	for line in file:
		i+=1
		if(line != '\n' and line.split()[0] != "-DOCSTART-"):
			gold_tag = line.split()[3]
			seq.append( gold_tag )

		elif(line == '\n' and seq != ["<s>"]):
			seq.append("</s>")
			gold_sequence.append(seq)
			seq = ["<s>"]

	true_tag_count = 0
	tag_count = 0


	for i in range(len(gold_sequence)):
		if(len(predicted_sequence[i]) ==len(gold_sequence[i]) ):
			for k in range(len(gold_sequence[i])):
				if( predicted_sequence[i][k] == gold_sequence[i][k]):
					true_tag_count+=1
				tag_count+=1

	return true_tag_count/tag_count


final_accuracy = accuracy()
print("\nAccuracy = " ,final_accuracy)
print()