#Bliss Translator Project by Miheer Patankar
from __future__ import division

import os
import threading
import pickle
import pyaudio
from six.moves import queue

from kivy.uix.scatter import Scatter
from kivy.uix.label import Label
from kivy.app import App
from kivy.graphics.svg import Svg
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout
from kivy.lang import Builder
from kivy.clock import Clock


from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types


# obtain a authentication from https://cloud.google.com/docs/authentication/getting-started
# place file path to json below
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Path/to/json.json"

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
Clock.max_iteration=1000000


#class microphone stream modificication from https://cloud.google.com/speech-to-text/docs/streaming-recognize
class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):

        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)



#function response loop  modificication from https://cloud.google.com/speech-to-text/docs/streaming-recognize
def response_loop(responses):

    if responses:
        try:
            response=next(responses)
            if not response.results:
                return

            result = response.results[0]
            if not result.alternatives:
                return

            transcript = result.alternatives[0].transcript
            if result.is_final:

                # Exit recognition if any of the transcribed phrases could be
                # one of our keywords. This is disabled from this version
                #if re.search(r'\b(exit|quit)\b', transcript, re.I):
                #    print('Exiting..')
                #    return
                return transcript
        except StopIteration:
            return

# dictionary that stores compositon of words
# comp_dict(cow)=['bovine', 'female']
# multiple word keys are have _ between
with open('./dict/comp_dict.pkl', 'rb') as f:
    comp_dict = pickle.load(f)

# dictionary gives additional information about word (definition, context)
with open('./dict/def_dict.pkl', 'rb') as f:
    def_dict = pickle.load(f)

#stores SVG of word
with open('./dict/svg_dict.pkl', 'rb') as f:
    svg_dict = pickle.load(f)

#

file_root='./svg/'

#kivy builder for svg widget
Builder.load_string("""
<SvgWidget>:
    do_rotation: False
<FloatLayout>:
    canvas.before:
        Color:
            rgb: (1, 1, 1)
        Rectangle:
            pos: self.pos
            size: self.size
""")

# Kivy SVG widget class
class SvgWidget(Scatter):

    def __init__(self, filename, **kwargs):
        super(SvgWidget, self).__init__(**kwargs)
        with self.canvas:
            svg = Svg(filename)
        self.size = svg.width, svg.height

# Main App for Kivy GUI
class SvgApp(App):

    def build(self):
        self.root = FloatLayout()
        self.n=0
        self.maintree = []# data structure for main word images
        self.maintreelabel = []# data structure for main word text
        self.leaftree = dict()# data structure for composition (sub) images
        self.sayings = []
        self.leaftreelabel=dict() # data structure for composition (sub) text

        # thread starts speech to text recognition in background
        t1 = threading.Thread(target=self.speechtotext)
        t1.start()

        # search for new words to display every .25 seconds. Can be increased/ decreased
        Clock.schedule_interval(self.update, .25)

    def update(self, *args, **kwargs):

        try:
            listobj=self.sayings.pop(0)

            self.screenupdate(listobj)
        except IndexError:
            pass

    # main speech to text function, runs as thread in background, see: https://cloud.google.com/speech-to-text/docs/streaming-recognize
    def speechtotext(self):
        language_code = 'en-US'  # a BCP-47 language tag

        client = speech.SpeechClient()
        config = types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code=language_code)
        streaming_config = types.StreamingRecognitionConfig(
            config=config,
            interim_results=True)

        with MicrophoneStream(RATE, CHUNK) as stream:
            audio_generator = stream.generator()
            requests = (types.StreamingRecognizeRequest(audio_content=content)
                        for content in audio_generator)

            responses = client.streaming_recognize(streaming_config, requests)
            print(responses)
            while True:
                output = response_loop(responses)
                if output is not None:
                    words = output.strip().split()
                    for val in words:
                        print(val)
                        self.sayings.append(val)


    # main function handler for GUI interface
    def screenupdate(self, word, *args, **kwargs):

        for wid in self.maintree:#moves all main images to the left
            wid.center=(wid.center_x-700, wid.center_y)

        for wid in self.maintreelabel:#moves all main labels to the left
            wid.center = (wid.center_x - 700, wid.center_y)


        for k,v in self.leaftree.items():#deletes all composition images
            for val in v:
                self.root.remove_widget(val)
            self.leaftree[k].clear()
        for k,v in self.leaftreelabel.items():#deletes all composition labels
            for val in v:
                self.root.remove_widget(val)
            self.leaftreelabel[k].clear()

        if word in svg_dict.keys():
            #if word is in dict, adds image of word and label to screen
            filename = file_root+svg_dict[word]
            svg = SvgWidget(filename, size_hint=(None, None))
            svg.scale=1
            svg.center=Window.center
            self.maintree.append(svg)
            self.root.add_widget(self.maintree[-1])

            mname=Label(text=word, color=(0, 0, 0, 1), size_hint=(None, None))
            mname.center = (svg.center_x, svg.center_y - (svg.scale * 328) / 2)
            self.maintreelabel.append(mname)
            self.root.add_widget(self.maintreelabel[-1])

            #if word has defined composition, adds compositional elements to screen
            if word in comp_dict.keys():

                self.leaftree[word]=[]
                self.leaftreelabel[word] = []
                leftpos=(self.maintree[-1].center[0]-((len(comp_dict[word])-1)/2*400*self.maintree[-1].scale),self.maintree[-1].center[1]- (328.00 * 1.75* svg.scale) / 2)
                n=0

                for val in comp_dict[word]:

                    if val in svg_dict.keys():
                        str=file_root+svg_dict[val]
                    else:#if image can't be found- default to circle image. (Should be removed with better error handling)
                        str='./svg/blank.svg'

                    #adds image of word and label of compositions to screen
                    lsvg = SvgWidget(str, size_hint=(None, None))
                    lsvg.scale=self.maintree[-1].scale*.75
                    lsvg.center=(leftpos[0]+n*400, leftpos[1])

                    lname=Label(text=val, color=(0, 0, 0, 1), size_hint=(None, None))
                    lname.center=(lsvg.center_x, lsvg.center_y-(lsvg.scale*328)/2)

                    self.leaftree[word].append(lsvg)
                    self.leaftreelabel[word].append(lname)
                    self.root.add_widget(self.leaftree[word][-1])
                    self.root.add_widget(self.leaftreelabel[word][-1])
                    n=n+1

        else: #if word does not have associated svg- just place word in canvas
            mname = Label(text=word, color=(0, 0, 0, 1), size_hint=(None, None))
            mname.center=Window.center
            self.maintree.append(mname)
            self.root.add_widget(self.maintree[-1])


if __name__ == '__main__':

    SvgApp().run()
