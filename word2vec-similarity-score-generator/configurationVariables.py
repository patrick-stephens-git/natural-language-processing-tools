## Configuration Variables:
input_training_text_filename = 'training-text.txt' 
input_stopwords_filename = 'stopwords.txt'

# Note: Context Word Window Size could be 50, 100, 200, etc.
context_word_window_size = 5 # Context Word is within N words before AND N words after the primary word being reviewed in the training text
number_negative_samples = 3 # For each primary word we include N examples that are not within it's context