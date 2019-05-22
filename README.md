#Bliss Translator 1.0.0

A live updating translator of speech to [BlissSymbolics][bliss_website]. 

Features not added in this release:

1.    Verb tenses: all verbs are unconjugated in their Bliss dictionary, so there needs to be a module that converts conjugations to their root form. 
2.    Non- English support: Blissymbolics have characters associated with non English languages, which makes it easy to add support for other languages
3.    Gendered Words: Blissymbolics is gendered, but my project doesn’t account for this
4.    Synonyms: Not all words have a direct Bliss Symbol, but they do have a likely synonym. I also did this in NLTK Wordnet
5.    Contextual understanding: my project does not have context dependent symbol selection

See additional information on the project [here][wu]

Required Dependencies: 
[Kivy][kivy_link], 
[Google Cloud Speech][cloud_link], 
[Pyaudio][pyaudio_link], 
[Six][six_link]


----

[bliss_website]:http://www.blissymbolics.org
[wu]: http://www.blissymbolics.org
[kivy_link]: https://kivy.org/#home
[cloud_link]: https://cloud.google.com/speech-to-text/
[pyaudio_link]: https://people.csail.mit.edu/hubert/pyaudio/
[six_link]: https://pypi.org/project/six/