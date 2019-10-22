#!/usr/bin/python3
import time

import speech_recognition as sr

import nltk
import re

from gtts import gTTS
import os

"""
Class to score sentiment of text.
Use domain-independent method of dictionary lookup of sentiment words,
handling negations and multiword expressions. Based on SentiWordNet 3.0.
"""


class SentimentAnalysis(object):
    """Class to get sentiment score based on analyzer."""

    def __init__(self, filename='SentiWordNet_3.0.0.txt', weighting='average'):
        """Initialize with filename and choice of weighting."""
        if weighting not in ('geometric', 'harmonic', 'average'):
            raise ValueError('Allowed weighting options are geometric, harmonic, average')
        # parse file and build sentiwordnet dicts
        self.swn_pos = {'a': {}, 'v': {}, 'r': {}, 'n': {}}
        self.swn_all = {}
        # print(filename)
        self.build_swn(filename, weighting)

    def average(self, score_list):
        """Get arithmetic average of scores."""
        if(score_list):
            return sum(score_list) / float(len(score_list))
        else:
            return 0

    def geometric_weighted(self, score_list):
        """"Get geometric weighted sum of scores."""
        weighted_sum = 0
        num = 1
        for el in score_list:
            weighted_sum += (el * (1 / float(2**num)))
            num += 1
        return weighted_sum

    # another possible weighting instead of average
    def harmonic_weighted(self, score_list):
        """Get harmonic weighted sum of scores."""
        weighted_sum = 0
        num = 2
        for el in score_list:
            weighted_sum += (el * (1 / float(num)))
            num += 1
        return weighted_sum

    def build_swn(self, filename, weighting):
        """Build class's lookup based on SentiWordNet 3.0."""
        records = [line.split('\t') for line in open(filename)]
        for rec in records:
            # has many words in 1 entry
            words = rec[4].split()
            pos = rec[0]
            for word_num in words:
                word = word_num.split('#')[0]
                sense_num = int(word_num.split('#')[1])

                # build a dictionary key'ed by sense number
                if word not in self.swn_pos[pos]:
                    self.swn_pos[pos][word] = {}
                self.swn_pos[pos][word][sense_num] = float(
                    rec[2]) - float(rec[3])
                if word not in self.swn_all:
                    self.swn_all[word] = {}
                self.swn_all[word][sense_num] = float(rec[2]) - float(rec[3])

        # convert innermost dicts to ordered lists of scores
        for pos in self.swn_pos.keys():
            for word in self.swn_pos[pos].keys():
                newlist = [self.swn_pos[pos][word][k] for k in sorted(
                    self.swn_pos[pos][word].keys())]
                if weighting == 'average':
                    self.swn_pos[pos][word] = self.average(newlist)
                if weighting == 'geometric':
                    self.swn_pos[pos][word] = self.geometric_weighted(newlist)
                if weighting == 'harmonic':
                    self.swn_pos[pos][word] = self.harmonic_weighted(newlist)

        for word in self.swn_all.keys():
            newlist = [self.swn_all[word][k] for k in sorted(
                self.swn_all[word].keys())]
            if weighting == 'average':
                self.swn_all[word] = self.average(newlist)
            if weighting == 'geometric':
                self.swn_all[word] = self.geometric_weighted(newlist)
            if weighting == 'harmonic':
                self.swn_all[word] = self.harmonic_weighted(newlist)

    def pos_short(self, pos):
        """Convert NLTK POS tags to SWN's POS tags."""
        if pos in set(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']):
            return 'v'
        elif pos in set(['JJ', 'JJR', 'JJS']):
            return 'a'
        elif pos in set(['RB', 'RBR', 'RBS']):
            return 'r'
        elif pos in set(['NNS', 'NN', 'NNP', 'NNPS']):
            return 'n'
        else:
            return 'a'

    def score_word(self, word, pos):
        """Get sentiment score of word based on SWN and part of speech."""
        try:
            return self.swn_pos[pos][word]
        except KeyError:
            try:
                return self.swn_all[word]
            except KeyError:
                return 0

    def score(self, sentence):
        """Sentiment score a sentence."""
        # init sentiwordnet lookup/scoring tools
        impt = set(['NNS', 'NN', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS','RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN','VBP', 'VBZ', 'unknown'])
        non_base = set(['VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NNS', 'NNPS'])
        negations = set(['not', 'n\'t', 'less', 'no', 'never','nothing', 'nowhere', 'hardly', 'barely','scarcely', 'nobody', 'none'])
        stopwords = nltk.corpus.stopwords.words('english')
        wnl = nltk.WordNetLemmatizer()

        scores = []
        tokens = nltk.tokenize.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)

        index = 0
        for el in tagged:

            pos = el[1]
            try:
                word = re.match('(\w+)', el[0]).group(0).lower()
                start = index - 5
                if start < 0:
                    start = 0
                neighborhood = tokens[start:index]

                # look for trailing multiword expressions
                word_minus_one = tokens[index-1:index+1]
                word_minus_two = tokens[index-2:index+1]

                # if multiword expression, fold to one expression
                if(self.is_multiword(word_minus_two)):
                    if len(scores) > 1:
                        scores.pop()
                        scores.pop()
                    if len(neighborhood) > 1:
                        neighborhood.pop()
                        neighborhood.pop()
                    word = '_'.join(word_minus_two)
                    pos = 'unknown'

                elif(self.is_multiword(word_minus_one)):
                    if len(scores) > 0:
                        scores.pop()
                    if len(neighborhood) > 0:
                        neighborhood.pop()
                    word = '_'.join(word_minus_one)
                    pos = 'unknown'

                # perform lookup
                if (pos in impt) and (word not in stopwords):
                    if pos in non_base:
                        word = wnl.lemmatize(word, self.pos_short(pos))
                    score = self.score_word(word, self.pos_short(pos))
                    if len(negations.intersection(set(neighborhood))) > 0:
                        score = -score
                    scores.append(score)

            except AttributeError:
                pass

            index += 1

        if len(scores) > 0:
            return sum(scores) / float(len(scores))
        else:
            return 0

    def is_multiword(self, words):
        """Test if a group of words is a multiword expression."""
        joined = '_'.join(words)
        return joined in self.swn_all


    def voice_interpreter(self):

        # enter the name of usb microphone that you found
        # using lsusb
        # the following name is only used as an example
        mic_name = "USB Device 0x46d:0x825: Audio (hw:1, 0)"
        # Sample rate is how often values are recorded
        sample_rate = 48000
        # Chunk is like a buffer. It stores 2048 samples (bytes of data)
        # here.
        # it is advisable to use powers of 2 such as 1024 or 2048
        chunk_size = 2048

        # Initialize the recognizer
        r = sr.Recognizer()

        # generate a list of all audio cards/microphones
        mic_list = sr.Microphone.list_microphone_names()

        # the following loop aims to set the device ID of the mic that
        # we specifically want to use to avoid ambiguity.
        for i, microphone_name in enumerate(mic_list):
            if microphone_name == mic_name:
                device_id = i

        # use the microphone as source for input. Here, we also specify
        # which device ID to specifically look for in case the microphone
        # is not working, an error will pop up saying "device_id undefined"
        with sr.Microphone(sample_rate=sample_rate,
                           chunk_size=chunk_size) as source:
            # wait for a second to let the recognizer adjust the
            # energy threshold based on the surrounding noise level
            r.adjust_for_ambient_noise(source)
            print("*listening")
            # listens for the user's input
            audio = r.listen(source)

            try:
                statement = r.recognize_google(audio)
                print("you said: " + statement)
                if "stop depression test" in statement.lower():
                    print("Stopping the test!")
                    s.speak_gtts(10)
                    exit()
                return statement

            # error occurs when google could not understand what was said
            except sr.UnknownValueError:
                print("Could not understand audio")
                return ""
                # exit()

            except sr.RequestError as e:
                print("Could not request results from Speech Recognition service; {0}".format(e))
                exit()



    def analyze_tone(self, read_data):
        # The ToneAnalyzer class from WDC
        from watson_developer_cloud import ToneAnalyzerV3

        # -------------------------------------------------------------------------
        # Instantiate TA Object with my Credentials
        # -------------------------------------------------------------------------
        tone_analyzer = ToneAnalyzerV3(
            iam_apikey="wN3kBNH9MJcyo7LDKBlrq2mmSbupCTnhC_hugTWgoa55",
            version='2018-02-16',
            url='https://gateway-lon.watsonplatform.net/tone-analyzer/api'
        )

        # -------------------------------------------------------------------------

        # Pass a single review to TA (one by one):

        json_output = tone_analyzer.tone(read_data, content_type='text/plain')

        result = json_output.result
        for i in result['document_tone']['tones']:
            if i is None:
                break
            print(i['tone_name']+": "+str(i['score']*100)+"%")
            # print(i['score'])

        # ------------------------------------------------------------------------

    def init_audio(self):
        tts = gTTS(text="Hello dear, how are u doing? Please describe your day.", lang='en')
        tts.save("audio1.mp3")
        tts = gTTS(text="It seems unpleasant. Tell me more about it", lang='en')
        tts.save("audio2.mp3")
        tts = gTTS(text="It seems you are not happy. Try sharing your problem with your loved ones.", lang='en')
        tts.save("audio3.mp3")
        tts = gTTS(text="Hope you are feeling better now.", lang='en')
        tts.save("audio4.mp3")
        tts = gTTS(text="You seem a bit off. Tell me more about it.", lang='en')
        tts.save("audio5.mp3")
        tts = gTTS(text="You are a bit upset. don't worry you would be better", lang='en')
        tts.save("audio6.mp3")
        tts = gTTS(text="Hope you are feeling better now.", lang='en')
        tts.save("audio7.mp3")
        tts = gTTS(text="You seem in a pleasant mood!", lang='en')
        tts.save("audio8.mp3")
        tts = gTTS(text="You seem to be in a good mood. Enjoy your day!", lang='en')
        tts.save("audio9.mp3")
        tts = gTTS(text="Stopping the test!", lang='en')
        tts.save("audio10.mp3")

    def speak_gtts(self, mp3_sequence):
        if mp3_sequence == 1:
            os.system('mpg321 audio1.mp3 -quiet')
        elif mp3_sequence == 2:
            os.system('mpg321 audio2.mp3 -quiet')
        elif mp3_sequence == 3:
            os.system('mpg321 audio3.mp3 -quiet')
        elif mp3_sequence == 4 or mp3_sequence == 7:
            os.system('mpg321 audio4.mp3 -quiet')
        elif mp3_sequence == 5:
            os.system('mpg321 audio5.mp3 -quiet')
        elif mp3_sequence == 6:
            os.system('mpg321 audio6.mp3 -quiet')
        elif mp3_sequence == 7:
            os.system('mpg321 audio7.mp3 -quiet')
        elif mp3_sequence == 8:
            os.system('mpg321 audio8.mp3 -quiet')
        elif mp3_sequence == 9:
            os.system('mpg321 audio9.mp3 -quiet')
        elif mp3_sequence == 10:
            os.system('mpg321 audio10.mp3 -quiet')



    def chatbot_reply(self,first_ans):
        print("sentiment score: "+str(first_ans))
        sum = first_ans
        if -1 <= first_ans < -0.20:
            print("It seems unpleasant. Tell me more about it.")
            s.speak_gtts(2)
            read_statement = s.voice_interpreter()
            second_neg_ans = s.score(read_statement)
            print("sentiment score: "+str(second_neg_ans))
            if read_statement == "":
                return
            s.analyze_tone(read_statement)
            sum += second_neg_ans

            if -1.0 <= second_neg_ans < 0.0:
                print("It seems you are not happy. Try sharing your problem with your loved ones.")
                s.speak_gtts(3)

            else:
                print("Hope you are feeling better now.")
                s.speak_gtts(4)

        elif -0.20 <= first_ans < 0.10:
            print("You seem a bit off... Tell me more about it.")
            s.speak_gtts(5)
            read_statement = s.voice_interpreter()
            second_ans = s.score(read_statement)
            print("sentiment score: "+str(second_ans))
            if read_statement == "":
                return
            s.analyze_tone(read_statement)
            sum += second_ans

            if -1.0 <= second_ans < 0.0:
                print("You are a bit upset, don't worry you would be better.")
                s.speak_gtts(6)

            else:
                print("Hope you are feeling better now.")
                s.speak_gtts(7)

        elif 0.10 <= first_ans < 0.40:
            print("You seem in a pleasant mood!")
            s.speak_gtts(8)

        elif 0.40 <= first_ans <= 1.0:
            print("You seem to be in a good mood. Enjoy your day!!")
            s.speak_gtts(9)

        return sum


if __name__ == "__main__":
    s = SentimentAnalysis(filename='SentiWordNet_3.0.0.txt', weighting='average')
    start_trigger = "Start Depression Test"
    stop_trigger = "Stop Depression Test"
    print("Always listening is on\nTrigger words:\n"
          "\"Start Depression Test\"-> To start the test\n"
          "\"Stop Depression Test\"-> To stop the test\n Wait untill Initialisation:")
    s.init_audio()
    while 1:
        voice_to_text = s.voice_interpreter().lower()
        if (start_trigger.lower()) in voice_to_text:
            # call("open /Applications/Siri.app")
            print("Hello dear, how are u doing? Please describe your day.")
            s.speak_gtts(1)
            read_statement = s.voice_interpreter()
            if read_statement == "":
                continue
            first_ans = s.score(read_statement)
            s.analyze_tone(read_statement)
            total_sum = s.chatbot_reply(first_ans)
            print("Overall depression scale : " + str(total_sum))
        elif (stop_trigger.lower()) in voice_to_text:
            print("Stopping the test!")
            s.speak_gtts(10)
            exit()
        time.sleep(0.001)
