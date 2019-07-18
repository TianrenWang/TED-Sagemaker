import numpy as np
import tensorflow_datasets as tfds
from sklearn.utils import shuffle

# data_path used to be "data/p53-50000.txt"

def text_processor(data_path):
    data = open(data_path, "r")
    line = data.readline()
    abstracts = []

    debugCounter = -99999999

    while line and debugCounter < 100:
        abstracts.append(str.encode(line))
        line = data.readline()
        debugCounter += 1

    data.close()

    input_vocab_size = 2 ** 13

    # Create a BPE vocabulary using the abstracts
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        abstracts, target_vocab_size=input_vocab_size)

    def encode(abstract):
        """Turns an abstract in English into BPE (Byte Pair Encoding).
        Adds start and end token to the abstract.

        Keyword arguments:
        abstract -- the abstract (type: bytes)
        """

        encoded_abstract = [tokenizer.vocab_size] + tokenizer.encode(
            abstract) + [tokenizer.vocab_size + 1]

        return encoded_abstract

    MAX_PROMPT_LENGTH = 128
    MAX_RESPONSE_LENGTH = 512

    # Create a list of encoded abstracts
    encoded_prompts = []
    encoded_responses = []

    for abstract in abstracts:

        # Separate the first sentence from rest
        periodIndex = abstract.find(b'.')
        firstSentence = abstract[:periodIndex + 1]
        rest = abstract[periodIndex + 2:]

        # Encode the responses and prompts
        encoded_prompt = encode(firstSentence)
        encoded_response = encode(rest)
        prompt_length = len(encoded_prompt)
        response_length = len(encoded_response)

        if response_length <= MAX_RESPONSE_LENGTH and prompt_length <= MAX_PROMPT_LENGTH:
            difference = MAX_PROMPT_LENGTH - prompt_length
            encoded_prompts.append(np.pad(encoded_prompt, (0, difference), 'constant'))
            difference = MAX_RESPONSE_LENGTH - response_length
            encoded_responses.append(np.pad(encoded_response, (0, difference), 'constant'))

    prompts = np.array(encoded_prompts)
    responses = np.array(encoded_responses)
    prompts, responses = shuffle(prompts, responses)

    return prompts, responses, tokenizer