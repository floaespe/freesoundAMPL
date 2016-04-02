import utils
import random
import collections
from sklearn import svm, tree

# Set parameters and load dataset
DATASET = 'soundscapesDescriptor'
NUMBER_OF_DIMENSIONS_OF_FEATURE_VECTOR = 7 # Maximum number of dimensions for the feature vector. Only the N most common tags will be used. Use a big number to "ommit" this parameter
CLASSIFIER_TYPE = 'tree' # Use 'svm' or 'tree'
PERCENTAGE_OF_TRAINING_DATA = 0.5 # Percentage of sounds that will be used for training (others are for testing)
MAX_INPUT_TAGS_FOR_TESTING = 5 # Use a big number to "omit" this parameter and use as many tags as originally are in the sound
dataset = utils.load_from_json(DATASET + '.json')
N = len(dataset[dataset.keys()[0]]) # Number of sounds per class
CLASS_NAMES = dataset.keys()

# 3) Define vector space
# **********************

# Get all tags in the dataset (the vocabulary)
all_tags = list()
for class_name in CLASS_NAMES:
	class_tags = utils.get_all_tags_from_class(class_name, dataset)
	all_tags += class_tags

# Filter out tags with less frequency (get only top N tags)
most_common_tags = [tag for tag, count in collections.Counter(all_tags).most_common(NUMBER_OF_DIMENSIONS_OF_FEATURE_VECTOR)]
filtered_tags = [tag for tag in most_common_tags if tag in all_tags]

# Build our prototype feature vector (unique list of tags), and print first 10 tags
prototype_feature_vector = list(set(filtered_tags))
print 'Created prototype feature vector with %i dimensions (originally %i dimensions)' % (len(prototype_feature_vector), len(set(all_tags)))
print 'Prototype vector tags (sorted by occurrence in filtered_tags):', ', '.join([tag for tag in filtered_tags[:10]]),
print '...\n' if len(filtered_tags) > 10 else '\n'

# 4) Project documents in the vector space
# ****************************************

# Example of getting feature vector from tags list...
random_sound_tags = random.choice(dataset[random.choice(dataset.keys())])['tags']
random_sound_feature_vector = utils.get_feature_vector_from_tags(random_sound_tags, prototype_feature_vector)
print 'Sound tags:', ', '.join([tag for tag in random_sound_tags])
print random_sound_feature_vector
print ''

# 5) Define train and testing set
# *******************************

n_training_sounds_per_class = int(N*PERCENTAGE_OF_TRAINING_DATA)
training_set = dict()
testing_set = dict()

# Get 'n_training_sounds_per_class' sounds per class 
for class_name, sounds in dataset.items():
	sounds_from_class = sounds[:] # Copy the list so when we later shuffle it does not affect the original data 
	random.shuffle(sounds_from_class)
	training_set[class_name] = sounds_from_class[:n_training_sounds_per_class] # First sounds for training
	testing_set[class_name] = sounds_from_class[n_training_sounds_per_class:] # Following sounds for testing

print 'Created training and testing sets with the following number of sounds:\n\tTrain\tTest'
for class_name in CLASS_NAMES:
	training_sounds = training_set[class_name]
	testing_sounds = testing_set[class_name]
	print '\t%i\t%i\t%s' % (len(training_sounds), len(testing_sounds), class_name)

# 6) Train classifier
# *******************

# Prepare data for fitting classifier (as sklearn classifiers require)
classes_vector = list()
feature_vectors = list()
for class_name, sounds in training_set.items():
	for sound in sounds:
		classes_vector.append(CLASS_NAMES.index(class_name))  # Use index of class name in CLASS_NAMES as numerical value (classifier internally represents class label as number)
		feature_vectors.append(utils.get_feature_vector_from_tags(sound['tags'], prototype_feature_vector))

# Create and fit classifier
print 'Training classifier (%s) with %i sounds...' % (CLASSIFIER_TYPE, len(feature_vectors)),
if CLASSIFIER_TYPE == 'svm':
	classifier = svm.LinearSVC()
	classifier.fit(feature_vectors, classes_vector)
elif CLASSIFIER_TYPE == 'tree':
	classifier = tree.DecisionTreeClassifier()
	classifier.fit(feature_vectors, classes_vector)
	# Print classifier decision rules
	# WARNING: do not run this if tree is too big, might freeze
	utils.export_tree_as_graph(classifier, prototype_feature_vector, class_names=CLASS_NAMES, filename='%s_tree.png' % DATASET)
else:
	raise Exception('Bad classifier type!!!')
print 'done!'

# 7) Evaluate
# ***********

# Test with testing set
print '\nEvaluating with %i instances...' % sum([len(sounds) for sounds in testing_set.values()]),
predicted_data = list()
for class_name, sounds in testing_set.items():
	for sound in sounds:
		input_tags = random.sample(sound['tags'], min(MAX_INPUT_TAGS_FOR_TESTING, len(sound['tags'])))  #sound['tags'] 
		feature_vector = utils.get_feature_vector_from_tags(input_tags, prototype_feature_vector)
		predicted_class_name = unicode(CLASS_NAMES[classifier.predict([feature_vector])[0]])  # Convert back to unicode
		predicted_data.append((sound['id'], class_name, predicted_class_name))
print 'done!'

# Compute overall accuracy
good_predictions = len([1 for sid, cname, pname in predicted_data if cname == pname])
wrong_predictions = len([1 for sid, cname, pname in predicted_data if cname != pname])
print '%i correct predictions' % good_predictions
print '%i wrong predictions' % wrong_predictions
print 'Overall accuracy %.2f%%' % (100 * float(good_predictions)/(good_predictions + wrong_predictions))

# Compute confussion matrix (further analysis)
matrix = list()
for class_name in CLASS_NAMES:
	predicted_classes = list()
	for sid, cname, pname in predicted_data:
		if cname == class_name:
			predicted_classes.append(pname)
	matrix.append([predicted_classes.count(target_class) for target_class in CLASS_NAMES])
print 'Confussion matrix:'
utils.print_confussion_matrix(matrix, labels=CLASS_NAMES, L=15)
