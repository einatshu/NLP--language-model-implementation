from math import log
import random
import re


class Ngram_Language_Model:
    """The class implements a Markov Language Model that learns amodel from a given text.
        It supoprts language generation and the evaluation of a given string.
        The class can be applied on both word level and caracter level.
    """
    n = 3
    chars = False
    models = {}

    def __init__(self, n=3, chars=False):
        """Initializing a language model object.
        Arges:
            n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
            chars (bool): True iff the model consists of ngrams of characters rather then word tokens.
                          Defaults to False
        """
        self.n = n
        self.chars = chars

    def split_text_to_ngrams(self, string, ngram):
        """split text string into a list of ngrams
            Args:
                param1: The text to split.
                param2: ngram size.

            Returns:
                list with all strings as ngram.
        """
        words = string
        start_string = ''
        if not self.chars:
            words = string.split()
            start_string = ' '
        grouped_words = [start_string.join(words[i: i + ngram]) for i in range(0, (len(words) - ngram+1))]
        return grouped_words

    def build_model(self, text):  # should be called build_model
        """populates a dictionary counting all ngrams in the specified text.
            Args:
                text (str): the text to construct the model from.
        """
        self.models.clear()
        # built model for each ngram length from n and down
        for i in range(1, self.n + 1):
            grouped_words = Ngram_Language_Model.split_text_to_ngrams(self, text, i)
            self.models[i] = {}
            for word in set(grouped_words):
                self.models[i][word] = grouped_words.count(word)

    def get_model(self):
        """Returns the model as a dictionary of the form {ngram:count}
        """
        return self.models[self.n]

    def get_model_probability(self, ngram):
        """return the model probability for the given ngram, if the ngram not exist in the model,
            return a smoothed  probability
            Args:
                param1: The ngram string

            Returns:
                the probability (between 0 to 1)
        """
        ngram_len = len(ngram)
        if not self.chars:
            ngram_len = len(ngram.split())
        if ngram_len == 1:
            # return probability from unigram model
            if ngram in self.models[1]:
                probability = (self.models[1][ngram]) / (sum(self.models[1].values())-(self.n-1))
            else:
                probability = 1 / (sum(self.models[1].values())-(self.n-1) + len(self.models[1]))
        else:
            context = ngram[:ngram_len - 1]
            if not self.chars:
                context = ' '.join(ngram.split()[:ngram_len - 1]) + " "
            if ngram in self.models[ngram_len]:
                probability = (self.models[ngram_len][ngram]) / (
                    sum(dict(filter(lambda item: item[0].startswith(context), self.models[ngram_len].items())).values()))
            else:
                probability = 1 / (sum(dict(filter(lambda item: item[0].startswith(context), self.models[ngram_len].items())).values()) + len(self.models[1]))
        return probability

    def get_all_candidates(self, context):
        """return all possible ngram that starts with the given context (n-1 string)
           Args:
               param1: the context (n-1 string)

           Returns:
               dict with all the possible ngram as keys and there occurrences in the model as values
        """
        context_len = len(context)
        if not self.chars:
            context_len = len(context.split())
            context = context + " "
        # find on ngram model context_len+1 all possible candidates
        return dict(filter(lambda item: item[0].startswith(context), self.models[context_len+1].items()))

    def get_random_candidate(self, context):
        """returns one ngram from candidates that was chosen randomly given it's occurrences in the model
            if candidates is empty return NONE
             Args:
                 param1: dict with all the possible ngram as keys and there occurrences in the model as values

             Returns:
                 returns one ngram string from candidates that was chosen randomly given it's occurrences in the model
                if candidates is empty return NONE
        """
        candidates = Ngram_Language_Model.get_all_candidates(self, context)
        if not candidates:
            return None
        random_candidate_ngram = random.choices(list(candidates.keys()), weights=list(candidates.values()), k=1)[0]
        random_candidate = random_candidate_ngram[-1]
        if not self.chars:
            random_candidate = random_candidate_ngram.split()[-1]
        return random_candidate

    def generate(self, context=None, n=20):
        """Returns a string of the specified length, generated by applying the language model
        to the specified seed context. If no context is specified the context should be sampled
        from the models' contexts distribution. Generation should stop before the n'th word if the
        contexts are exhausted.

            Args:
                context (str): a seed context to start the generated string from. Defaults to None
                n (int): the length of the string to be generated.

            Return:
                String. The generated text.
        """
        generated_string = ""
        if context is None:
            context = ""
            for i in range(1, self.n):
                random_candidate = Ngram_Language_Model.get_random_candidate(self, context)
                if random_candidate is None:
                    return context
                if not self.chars:
                    context += " " + random_candidate
                else:
                    context += random_candidate

        if context.endswith(" ") and not self.chars:
            context = context[-1:]
        generated_string += context
        generated_string_length = n-len(context.split())
        if self.chars:
            context = context[-self.n+1:]
            generated_string_length = n-len(generated_string)
        for i in range(0, generated_string_length):
            random_candidate = Ngram_Language_Model.get_random_candidate(self, context)
            if random_candidate is None:
                break
            context_len = len(context)
            if not self.chars:
                context_len = len(context.split())
                random_candidate = " " + random_candidate
            generated_string += random_candidate
            if context_len < self.n-1:
                context = context + random_candidate
            else:
                if not self.chars:
                    context = ' '.join(context.split()[context_len-self.n:]) + random_candidate
                else:
                    context = context[context_len-self.n:] + random_candidate

        return generated_string

    def evaluate(self, text):
        """Returns the log-likelihod of the specified text to be generated by the model.
           Laplace smoothing should be applied if necessary.

           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        # get probability for the seed
        prob = 0
        for i in range(1, self.n):
            ngram = Ngram_Language_Model.split_text_to_ngrams(self, text, i)[0]
            prob += log(Ngram_Language_Model.get_model_probability(self, ngram))

        ngrams = Ngram_Language_Model.split_text_to_ngrams(self, text, self.n)
        text_len = len(text)-1
        if not self.chars:
            text_len = len(text.split())-1
        for i in range(1, text_len):
            ngram = ngrams[i-1]
            prob += log(Ngram_Language_Model.get_model_probability(self, ngram))
        return prob

    def smooth(self, ngram):
        """Returns the smoothed (Laplace) probability of the specified ngram.
            Args:
                ngram (str): the ngram to have it's probability smoothed

            Returns:
                float. The smoothed probability.
        """
        smoothed_probability = 0
        ngram_len = len(ngram)
        if not self.chars:
            ngram_len = len(ngram.split())
        if ngram_len == 1:
            # return probability from unigram model
            if ngram in self.models[1]:
                smoothed_probability = (self.models[1][ngram]+1) / (sum(self.models[1].values()) - (self.n - 1) + len(self.models[1]))
            else:
                smoothed_probability = 1 / (sum(self.models[1].values()) - (self.n - 1) + len(self.models[1]))
        else:
            context = ngram[:ngram_len - 1]
            if not self.chars:
                context = ' '.join(ngram.split()[:ngram_len - 1])
            if ngram in self.models[ngram_len]:
                smoothed_probability = (self.models[ngram_len][ngram]+1) / (
                    sum(dict(
                        filter(lambda item: item[0].startswith(context + " "),
                               self.models[ngram_len].items())).values()) + len(self.models[1]))
            else:
                smoothed_probability = 1 / (sum(dict(
                    filter(lambda item: item[0].startswith(context + " "),
                           self.models[ngram_len].items())).values()) + len(
                    self.models[1]))
        return smoothed_probability


def normalize_text(text):
    text = text.lower()
    pattern = r'''(?x)          # set flag to allow verbose regexps
        (?:[A-Z]\.)+          # abbreviations, e.g. U.S.A.
        | \w+(?:-\w+)*        # words with optional internal hyphens
        | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
        | \.\.\.              # ellipsis
        | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
        '''
    regexp = re.compile(pattern)
    tokens = regexp.findall(text)
    return ' '.join(tokens)
