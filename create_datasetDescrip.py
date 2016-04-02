from freesound import freesound
import random
import utils
from utils.run_entity_linking import spotlight

# Setup Freesound client
# Get the API key from http://www.freesound.org/apiv2/apply/ (you'll need Freesound user account)
API_KEY='15cc0ccd293636ef67e12db888b94834bbddc7f2'
c = freesound.FreesoundClient()
c.set_token(API_KEY,"token")

# 1) Define a number of audio categories and find audio examples from Freesound for each category
# ***********************************************************************************************

# Configure dataset parameters and audio categories
DATASET_NAME = 'soundscapesDescriptor' # Dataset will be saved in a .json file with this name
N = 100 # Number of sounds per class
DATASET_CLASSES = {  
	# Must be dictionary with structure like {'class name': 'query terms', 'class name 2': 'query terms 2',... }
	'Geophony': 'city', 	
	'Biophony': 'forest', 		
	'Anthropohony': 'human voice', 										
}
N_SOUNDS_PER_USER = 3  # Do not get more than 3 sounds per user

# Get sound examples from Freesound
dataset = dict()
for name, target_query in DATASET_CLASSES.items():
	print 'Getting sounds for class %s...' % name,
	
	# Get first page of results
	PAGE_SIZE = 150 # Page size for fs requests
	N_PAGES = int((N * 5) / PAGE_SIZE)  # Number of pages to retrieve
	descriptors = "lowlevel.mfcc.mean,lowlevel.mfcc.var,lowlevel.spectral_flux.mean,lowlevel.spectral_flux.var,lowlevel.dissonance.mean,lowlevel.dissonance.var"
	fields = "id,tags,description,username"
	results_pager = c.text_search(
		query=target_query,
		fields=fields,
		descriptors=descriptors,
		page_size=PAGE_SIZE,
		group_by_pack=1,
		)
	all_results = results_pager.results

	# TIP ON AUDIO FEATURES: you can get also audio features extracted in freesound by passing a 'descriptors' 
	# parameter in the text_search function and including 'analysis' in the fields list 
	# (see http://www.freesound.org/docs/api/resources_apiv2.html#response-sound-list):
	#
	# fields = "id,tags,description,username,analysis"
	# descriptors = "lowlevel.spectral_centroid,lowlevel.barkbands.mean"
	#
	# results_page = c.text_search(query=target_query, fields=fields, descriptors=descriptors, ...)
	# ...
	
	# Get extra pages
	for i in range(0, N_PAGES):
		if results_pager.count > (i+1) * PAGE_SIZE:
			results_pager = results_pager.next_page()
			all_results += results_pager.results
	
	# Get only N sounds max per user
	user_sounds_count = dict()
	filtered_results = list()
	random.shuffle(all_results)  # Shuffle list of sounds (randomise order)
	for result in all_results:
		if result["username"] in user_sounds_count:
			user_sounds_count[result["username"]] += 1
		else:
			user_sounds_count[result["username"]] = 1
		if user_sounds_count[result["username"]] <= N_SOUNDS_PER_USER:
			filtered_results.append(result)

	# Randomly select N sounds from al results obtained
	if len(filtered_results) >= N:
		selected_sounds = random.sample(filtered_results, N)
		dataset[name] = selected_sounds
		print 'selected %i sounds out of %i!' % (len(selected_sounds), len(filtered_results))
	else:
		print 'not enough sounds were found for current class (%i sounds found).' % len(filtered_results)

	# TIP ON KEYWORD EXTRACTION: we could extract some keywords from the textual descriptions using functions
	# provided in ELVIS (see https://github.com/sergiooramas/elvis and run_entity_linking.py file in utils folder)
	# For each selected sound in our dataset we could do something like:
	#
	# from utils.run_entity_linking import spotlight
	#
	# sound_textual_description = "One of the English summer storms of 2014 recorded on a condenser mic. The neighbor's dog barks at it at some point. \r\n\r\nNaturalistic, no processing done to it whatsoever."
	# results = spotlight(sound_textual_description.split('\n'))
	# keywords = list()
	# for element in results:
	# 	 for entity in element['entities']:
	#		 keywords.append(entity['label'])

# Save dataset to file so we can work with it later on
utils.save_to_json('%s.json' % DATASET_NAME, dataset)

# 2) Know your dataset
# ********************

# Generate html files with sound examples and show most common tags per class
for class_name, sounds in dataset.items():
	print class_name
	utils.generate_html_file_with_sound_examples([sound['id'] for sound in sounds][:15], 'html/%s_%s.html' % (DATASET_NAME, class_name))
	class_tags = utils.get_all_tags_from_class(class_name, dataset)
	utils.print_most_common_tags(class_tags)
