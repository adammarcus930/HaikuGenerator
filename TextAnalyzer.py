import numpy as np
import pandas as pd
import re
from num2words import num2words
import copy
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

model = BertForMaskedLM.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class TextFormat:
    def __init__(self):
        self = self

    # Split text into words
    def getwords(self, text):
        words = pd.Series(re.findall(r"[\w']+", text))
        return words

    # Convert an array of words to a single string
    def arraytotext(self, arr):
        text = " ".join(arr)
        return text

    # Clean text by removing numbers and transforming it text
    def numtotext(self, text):
        words = self.getwords(text)
        for word in words:
            if word.isdigit():
                text = re.sub(rf"\b{word}\b", num2words(word), text)
        return text


class PhonemeTransform:
    def __init__(self, phodict):
        self.phodict = phodict

    # Find words that are not located in the cmudict phoenetic dictionary
    def getunknowns(self, words):
        unknown_words = set(
            [word for word in words if word.lower() not in self.phodict.keys()]
        )
        return unknown_words

    # Takes a list of unknown words and detrmines which ones can be can be split into two words
    def _findwordsplits(self, unknown_words):
        splitwords = {}
        for word in unknown_words:
            wordlength = len(word)
            for i in range(wordlength):
                split1 = word[0 : i + 1]
                split2 = word[i + 1 : wordlength]
                if split1 in self.phodict.keys():
                    if split2 in self.phodict.keys():
                        splitwords[word] = [split1, split2]
        return splitwords

    # Convert plain text to Phoenemes
    def _getphoneme(self, word):
        phoneme = "".join(
            self.phodict[word][0]
        )  # always use first pronuciations at index 0
        return phoneme

    # Get Count of Phonemes
    def syllablecount(self, phoneme):
        count = 0
        for letter in phoneme:
            if letter.isdigit():
                count = count + 1
        return count

    def addsplitwords(self, unknown_words):
        # Find words that can be split into to and converted
        splitwords = self._findwordsplits(unknown_words)
        # make a copy of this dictionary
        worddict = copy.deepcopy(splitwords)

        # handle words with single letters separately
        special_cases = {
            "a": "AH0",
            "i": "IY0",
            "o": "OW0",
            "u": "UW0",
            "y": "IY0",
            "s": "Z",
        }

        for i in worddict:

            firstword = worddict[i][0]
            secondword = worddict[i][1]

            # If the first word is a single letter then make it all caps
            if len(firstword) == 1:
                firstword = firstword.upper()

            # Otherwise use CMUDICT to convert
            else:
                firstword = self._getphoneme(firstword)

            # If the second word is a single letter then apply the special cases where applicable
            if len(secondword) == 1:

                # If letter is a special case replace it
                if secondword in special_cases.keys():
                    secondword = pd.Series(worddict[i]).replace(special_cases)[1]

                # Otherwise make it all caps
                else:
                    secondword = secondword.upper()

            # Otherwise use CMUDICT to convert
            else:
                secondword = self._getphoneme(secondword)

            # added nested list to match format of CMUDICT
            worddict[i] = [[firstword, secondword]]

            # Add worddict to the current dictionary and return the new result
        self.phodict = {**self.phodict, **worddict}
        # return self.phodict

    # CMUDICT uses numbers (0-2) to denote stress of the syllable. Although this is something that could be explored
    # later, it is adding unnecessary complexity and should be changed to a consistent format.
    def convertsyllables(self, text):
        text = re.sub("(1|2)", "0", text)
        return text

    # transform english text into phonemes
    def transform(self, text):
        words = TextFormat.getwords(self, text)
        try:
            phonemes = [*map(self._getphoneme, words)]
            phonemes = TextFormat.arraytotext(self, phonemes)
            return phonemes
        except:
            raise ValueError(
                "A word in the Haiku was not in the CMUDICT."
                " Make sure only valid haikus are used for this function."
            )

    # Create a phoneme dictionary
    def invertdictionary(self):

        engdict = {}
        # Loop through all of the words in the dictionary
        for word in self.phodict:
            # Create a list to hold all of the possible words associated with a phoneme
            p_list = []

            phoneme = "".join(
                self.phodict[word][0]
            )  # always use first phoneme for a word
            phoneme = self.convertsyllables(phoneme)

            # if the phoneme already exists add it to that list
            if phoneme in engdict.keys():
                p_list = engdict[phoneme]
                p_list.append(word)

            # Otherwise create a new list
            else:
                p_list.append(word)

            engdict[phoneme] = p_list
        return engdict


class PhonemeReverse(PhonemeTransform):
    def __init__(self, phodict, engdict):
        super().__init__(phodict)
        self.engdict = engdict

    # Used for getenlgish funtion to convert phoneme to a list of possible english words
    def _getwordarray(self, phoneme):
        try:
            words = self.engdict[phoneme]
        # if there are no words in the dictionary then create a flag to handle later
        except:
            syllables = super(PhonemeReverse, self).syllablecount(phoneme)
            words = ["", syllables]
        return words

    # A function that takes in haikus and predicts what words should be used in the list. The algorithm works as such:
    # 1) Create a sentence using a "naive" prediction which simply uses the first word in the array
    # 2) Iterate through the phonemes in the sentence, if it only has one option than use it, else move to step 3
    # 3) Use BERT a bi-directional NLP package to create a similarity matrix of most likely words in the sentence
    # 4) Join array of similar words with possible words associated with the phoneme
    # 5) If the array is empty default to the first word, else pick the highest ranked word
    # 6) Add the word to the list and move to the next phoneme
    # 7) If run is > 1 then rewrite the base to the newly predicted text
    # 8) Keep rewriting the base until the runs are done and return the final version

    def getenglish(self, text, runs):
        englishlist = []
        # use getwords to get phonemes
        phonemes = TextFormat.getwords(self, text)
        # get 2D array of lists of possibilities for each word
        wordsarray = phonemes.apply(self._getwordarray)
        # Get a baseline to use predictions off of by taking the first word for each list
        baseline = "[CLS] "
        baseline = baseline + " ".join(
            wordsarray.apply(lambda x: x[0])
        )  # Use first word
        baseline = baseline + " [SEP]"
        # Run algorithim twice for better results
        for i in range(runs):
            # if its past the first run then use the first run as the new baseline
            if i > 0:
                baseline = (
                    "[CLS] " + TextFormat.arraytotext(self, englishlist) + " [SEP]"
                )
                englishlist = []
            for index, word in enumerate(wordsarray):
                if len(word) > 1:
                    # Replace word with mask
                    masked = " " + word[0] + " "  # ensure it is a full word
                    sentence = re.sub(masked, " [MASK] ", baseline)

                    tokenized_text = tokenizer.tokenize(sentence)
                    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

                    # Create the segments tensors.
                    segments_ids = [0] * len(indexed_tokens)

                    # Convert inputs to PyTorch tensors
                    tokens_tensor = torch.tensor([indexed_tokens])
                    segments_tensors = torch.tensor([segments_ids])

                    # Predict all tokens
                    with torch.no_grad():
                        predictions = model(tokens_tensor, segments_tensors)

                    predicted_words = pd.DataFrame(predictions[0, index])
                    predicted_words["Word"] = [
                        tokenizer.convert_ids_to_tokens([x])[0]
                        for x in range(len(predicted_words))
                    ]
                    # If its an actual word then merge the predicted words with the word array and grab the
                    # highest ranked word
                    if type(word[1]) != int:
                        # Create a word dataframe to merge with
                        wordlist = pd.DataFrame(word, columns=["Word"])
                        try:
                            best_word = (
                                pd.merge(predicted_words, wordlist, on="Word")
                                .sort_values(0, ascending=False, ignore_index=True)
                                .loc[0, "Word"]
                            )
                        except:
                            best_word = word[0]
                    # If the word is a number then that means there was are no english words associated with the phoneme
                    # So instead use the highest ranked word with the same syllable count
                    else:
                        # The second value is the syllable count created by the get word array function
                        syllables = word[1]
                        # Sort Predicted words
                        predicted_words.sort_values(
                            0, ascending=False, inplace=True, ignore_index=True
                        )
                        # Loop through and find the first word with the same syllable count
                        for pred_word in predicted_words["Word"]:
                            # Try to get syllable count of the word
                            try:
                                pred_syllable = super(
                                    PhonemeReverse, self
                                ).syllablecount(self._getphoneme(pred_word))
                                if pred_syllable == syllables:
                                    best_word = word
                                    break
                            # If the word doesnt exist then skip that word
                            except:
                                continue
                        # If the word still wasnt found then grab the best word
                        # Noted: This will create an error in the syllable co
                        if type(word[1]) == int:
                            best_word = predicted_words.loc[0, "Word"]

                else:
                    # Only one word
                    best_word = word[0]

                englishlist.append(best_word)
        return englishlist

