# M6(GPT)3: Generating Multitrack Modifiable Multi-Minute MIDI Music from Text using Genetic algorithms, Probabilistic methods and GPT Models in any Progression and Time signature 

This work introduces the M6(GPT)3 composer system, capable of generating complete, multi-minute musical compositions with complex structures in any time signature, in the MIDI domain from input descriptions in natural language. The system utilizes an autoregressive transformer language model to map natural language prompts to composition parameters in JSON format. The defined structure includes time signature, scales, chord progressions, and valence-arousal values, from which accompaniment, melody, bass, motif, and percussion tracks are created. We propose a genetic algorithm for the generation of melodic elements. The algorithm incorporates mutations with musical significance and a fitness function based on normal distribution and predefined musical feature values. The values adaptively evolve, influenced by emotional parameters and distinct playing styles. The system for generating percussion in any time signature utilises probabilistic methods, including Markov chains. Through both human and objective evaluations, we demonstrate that our music generation approach outperforms baselines on specific, musically meaningful metrics, offering a viable alternative to purely neural network-based systems.

## Example generations
To present the functionality of the system, I generated couple of songs using descriptions from [Meta's MusicGen](https://audiocraft.metademolab.com/musicgen.html) and [Google's MusicLM](https://google-research.github.io/seanet/musiclm/examples/) sites. Presented wav files are automatically synthesized from MIDI using General MIDI soundfont, so they have quite basic instrument sounding. They can however be used to synthesize with finer samples.

Mind, that system generates full songs with various sections, for example verse, chorus, bridge, so things like overall mood / scales change as the song progresses.

<table>

<tr>
<td>

Prompt: The main soundtrack of an arcade game. It is fast-paced and upbeat, with a catchy electric guitar riff. The music is repetitive and easy to remember, but with unexpected sounds, like cymbal crashes or drum rolls.

</td>
<td>

<audio controls>
  <source src="assets/audio/Arcade-Rush-20231208-135906.wav" type="audio/wav">
Audio can not be played here.
</audio>

</td>
</tr>

<tr>
<td>

Prompt: We can hear a choir, singing a Gregorian chant, and a drum machine, creating a rhythmic beat. The slow, stately sounds of strings provide a calming backdrop for the fast, complex sounds of futuristic electronic music.

</td>
<td>

<audio controls>
  <source src="assets/audio/Ancient-Future-20231208-121749.wav" type="audio/wav">
Audio can not be played here.
</audio>

</td>
</tr>

<tr>
<td>

Prompt: Smooth jazz, with a saxophone solo, piano chords, and snare full drums

</td>
<td>

<audio controls>
  <source src="assets/audio/Velvet-Evening-20231208-114938.wav" type="audio/wav">
Audio can not be played here.
</audio>

</td>
</tr>

<tr>
<td>

Prompt: 80s electronic track with melodic synthesizers, catchy beat and groovy bass

</td>
<td>

<audio controls>
  <source src="assets/audio/Retro-Synthwave-20231209-175338.wav" type="audio/wav">
Audio can not be played here.
</audio>

</td>
</tr>

<tr>
<td>

Prompt: Progressive rock drum and bass solo

</td>
<td>

<audio controls>
  <source src="assets/audio/Progressive-Odyssey-20231208-143.wav" type="audio/wav">
Audio can not be played here.
</audio>

</td>
</tr>

<tr>
<td>

Prompt: drum and bass beat with intense percussions

</td>
<td>

<audio controls>
  <source src="assets/audio/Intense-Rhythm-20231208-115146.wav" type="audio/wav">
Audio can not be played here.
</audio>

</td>
</tr>

<tr>
<td>

Prompt: A grand orchestral arrangement with thunderous percussion, epic brass fanfares, and soaring strings, creating a cinematic atmosphere fit for a heroic battle.

</td>
<td>

<audio controls>
  <source src="assets/audio/Heroic-Skies-20231208-130450.wav" type="audio/wav">
Audio can not be played here.
</audio>

</td>
</tr>

<tr>
<td>

Prompt: Funky piece with a strong, danceable beat and a prominent bassline. A catchy melody from a keyboard adds a layer of richness and complexity to the song.

</td>
<td>

<audio controls>
  <source src="assets/audio/Funk-Odyssey-20231209-134348.wav" type="audio/wav">
Audio can not be played here.
</audio>

</td>
</tr>
<tr>
<td>

Prompt: Epic soundtrack using orchestral instruments. The piece builds tension, creates a sense of urgency. An a cappella chorus sing in unison, it creates a sense of power and strength.

</td>
<td>

<audio controls>
  <source src="assets/audio/Epic-Orchestral-Surge-20231208-1.wav" type="audio/wav">
Audio can not be played here.
</audio>

</td>
</tr>

<tr>
<td>

Prompt: Violins and synths that inspire awe at the finiteness of life and the universe.

</td>
<td>

<audio controls>
  <source src="assets/audio/Celestial-Reverie-20231208-11391.wav" type="audio/wav">
Audio can not be played here.
</audio>

</td>
</tr>

</table>

### Generating from MIDICaps descriptions

In the second test, we compared the music generated by our system with pieces from the recently released [MIDICaps dataset](https://github.com/AMAAI-Lab/MidiCaps), which currently is the only openly available large-scale MIDI dataset with text captions. We randomly selected pieces from the dataset and generated corresponding songs using our system based on the same descriptions. These descriptions precisely specify structures of the songs, such as chords and time signatures. Sounds were generated with the usage of Musescore samples.

<table>

<tr>
<td>

Prompt: A melodic electronic soundtrack featuring a synth lead, piano, drums, synth pad, and synth strings. The song evokes a cinematic and relaxing atmosphere, suitable for meditative moments or epic scenes. Set in the key of C major with a moderate tempo, the composition maintains a 4/4 time signature throughout its duration. The harmonic structure revolves around the chords G and C, which repeat consistently.

</td>
<td>

<audio controls>
  <source src="assets/audio/MIDICaps_desc/Cinematic Serenity_20240727_202433.mp3" type="audio/mp3">
Audio can not be played here.
</audio>

</td>
</tr>

<tr>
<td>

Prompt: A classical soundtrack composition featuring piano and violin, this song evokes a cinematic atmosphere with its dark and epic tones. Set in the key of E major and moving at a moderate tempo, it's characterized by an uncommon 1/4 time signature. The emotional depth and relaxing qualities make it a fitting choice for film scores.
</td>
<td>

<audio controls>
  <source src="assets/audio/MIDICaps_desc/Cinematic Echoes_20240727_204036.mp3" type="audio/mp3">
Audio can not be played here.
</audio>

</td>
</tr>

<tr>
<td>

Prompt: A lengthy electronic composition set in D major, this piece unfolds at a slow tempo, creating an atmosphere that's equal parts melodic and dark. The Hammond organ takes the lead, supported by the steady pulse of the electric bass and drums. Throughout the piece, the chord progression of F#m, D, and E/B repeats, adding to its epic and relaxing qualities.
</td>
<td>

<audio controls>
  <source src="assets/audio/MIDICaps_desc/Epic Dusk_20240727_205343.mp3" type="audio/mp3">
Audio can not be played here.
</audio>

</td>
</tr>

<tr>
<td>

Prompt: A meditative pop song with electronic elements, featuring acoustic guitar and piano leads accompanied by synth strings, acoustic bass, and electric piano. The piece is in C major with a 4/4 time signature and a Moderato tempo of 100 bpm. The chord progression of Dm7, G7, C, Gm6, and A7 adds to the song's melodic and relaxing atmosphere, evoking a sense of love and happiness throughout its duration.

</td>
<td>

<audio controls>
  <source src="assets/audio/MIDICaps_desc/Serenity Pulse_20240727_210757.mp3" type="audio/mp3">
Audio can not be played here.
</audio>

</td>
</tr>

<tr>
<td>

Prompt: A melodic pop song with a touch of electronic elements, set in a 12/8 time signature at a moderate tempo. The piano leads the way, creating a cinematic atmosphere. The piece is in the key of F# major, with a chord progression of B, F#, and C# adding to its captivating nature.

</td>
<td>

<audio controls>
  <source src="assets/audio/MIDICaps_desc/Cinematic Echoes_20240727_214932.mp3" type="audio/mp3">
Audio can not be played here.
</audio>

</td>
</tr>

</table>

### Modifying compositions
The system is also capable of modifying previously generated compositions to the user's needs. Table below presents prompts that modify the composition, compositions themselves, as well as time signature, tempo and structure.

<table>
<tr>
<td>

Prompt: Write me a slow, dark song with complex chords.

</td>
<td>

<audio controls>
  <source src="assets/audio/iterative/1/Shadows-Linger-20231211-113318.wav" type="audio/wav">
Audio can not be played here.
</audio>

</td>
<td>

4/4, 60 BPM.

</td>
<td>

Intro, verse, chorus, verse, chorus, bridge, chorus.

</td>
</tr>

<tr>
<td>

Prompt: Increase the tempo of the song and add a more progressive time signature.

</td>
<td>

<audio controls>
  <source src="assets/audio/iterative/2/Whispers-of-the-Eclipse-20231211.wav" type="audio/wav">
Audio can not be played here.
</audio>

</td>
<td>

7/8, 90 BPM.

</td>
<td>

Intro, verse, chorus, verse, chorus, bridge, chorus.

</td>
</tr>

<tr>
<td>

Prompt: Keep the instruments from the previous version. Let's use an even more complex
11/8 meter.

</td>
<td>

<audio controls>
  <source src="assets/audio/iterative/3/Veil-of-Twilight-20231211-102124.wav" type="audio/wav">
Audio can not be played here.
</audio>

</td>
<td>

11/8, 90 BPM.

</td>
<td>

Intro, verse, chorus, verse, chorus, bridge, chorus.

</td>
</tr>

<tr>
<td>

Prompt: Add an ambient section to the piece, with one chord playing through the
entire section with a mesmerizing motif.

</td>
<td>

<audio controls>
  <source src="assets/audio/iterative/4/Veil-of-Twilight-20231211-001750.mp3" type="audio/mp3">
Audio can not be played here.
</audio>

</td>
<td>

11/8, 90 BPM.

</td>
<td>

Intro, verse, chorus, verse, chorus, bridge, ambient, chorus.

</td>
</tr>

<tr>
<td>

Prompt: Alright, but let's make the ambient section shorter. After the ambient section, I would like a guitar solo section that would sound like a collaboration between Pink Floyd and Peter Gabriel.

</td>
<td>

<audio controls>
  <source src="assets/audio/iterative/5/Veil-of-Twilight-20231211-003343.mp3" type="audio/mp3">
Audio can not be played here.
</audio>

</td>
<td>

11/8, 90 BPM.

</td>
<td>

Intro, verse, chorus, verse, chorus, bridge, ambient, guitar solo, chorus.

</td>
</tr>

<tr>
<td>

Prompt: Add a Bach-inspired passage before the solo begins. Let's also use
a different instrumentation for the solo to better fit in with the rest. You can also change the guitar, because it doesn't
fit very well. Let's change the time signature to 13/8. Remember that the duration of chords is given in
bars.

</td>
<td>

<audio controls>
  <source src="assets/audio/iterative/6/Veil-of-Twilight-20231210-234356.wav" type="audio/wav">
Audio can not be played here.
</audio>

</td>
<td>

13/8, 90 BPM.

</td>
<td>

Intro, verse, chorus, verse, chorus, bridge, ambient, guitar solo, chorus.

</td>
</tr>
<tr>
<td>

Prompt: Let's change the intro to one inspired by Chopin's Funeral March. Let's also change the
the chorus to a more powerful one, keeping its dark atmosphere.

</td>
<td>

<audio controls>
  <source src="assets/audio/iterative/7/Veil-of-Twilight-20231210-232103.wav" type="audio/wav">
Audio can not be played here.
</audio>

</td>
<td>

13/8, 90 BPM.

</td>
<td>

Intro, verse, chorus, verse, chorus, bridge, ambient, guitar solo, chorus.

</td>
</tr>


</table>

### Real life usages
#### Jakub Poćwiardowski - Na skraju nieświadomości

M6(GPT)3 output:
<audio controls>
  <source src="assets/audio/Synthetic Dreams_20240323_203846.mp3" type="audio/mp3">
Audio can not be played here.
</audio>

Final work:

<iframe width="560" height="315" src="https://www.youtube.com/embed/4qFC7g5LPgk" frameborder="0" allowfullscreen></iframe>

M6(GPT)3 was used as a template creator for this track, providing MIDI for percussion, chords, bass, and a distinctive violin motif in 7/8 time. The remaining parts, such as guitars, were recorded, added, and produced by me in a DAW.
#### Jakub Poćwiardowski - Istnienia

<iframe width="560" height="315" src="https://www.youtube.com/embed/TYkjHCxog0Q" frameborder="0" allowfullscreen></iframe>

Following the previous track, *Istnienia* uses only the motif in 7/8. However, this time the generated MIDI triggers multiple layers of samples, including clock sounds and yangqin. The time signature later shifts to 6/8 with a tempo shift.

#### Jakub Poćwiardowski & Lili - The Run

<iframe width="560" height="315" src="https://www.youtube.com/embed/U0jGWyrcGjM?si=aYLsAGKMtfmizU9L&amp;start=149" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

This song involved a minor but very specific use of M6(GPT)3. I needed a backing template for a guitar solo in 7/4 (minute 2:29), and I wanted it to feature an interesting key change. After a long discussion with the model, we came up with a chord progression that I then used to improvise the solo myself.

In the final version, only the drums from the M6(GPT)3-generated MIDI remained, as my friend Lili created an orchestral arrangement based on the chords. Interestingly, due to mismatched drum component mappings, the drum fills ended up triggering different sounds than originally intended. However, I  found the result interesting and decided to keep it.


### System prompt

To generate song structure and parameters a LLM is used. We chose GPT family because of its promising music theory knowledge and ability to use it via openai API, making it accessible from any computer, contrary to locally hosted LLMs requiring high computing resources. 

To obtain the structure in an appropriate form, the model is provided with precise instructions as a system prompt, along with the JSON structure presented below:

```text
{
  "name": "<name of the song>",
  "sections": {
    "<section 1 name>": {
      "scale": "<scale_tonic> <scale_type>",
      "bpm": <bpm>,
      "time_signature": "<time signature>",
        "parts": {
          "bass": "<instrument> <style>  or None",
          "motif": "<instrument> <style>  or None",
          "percussion": "<instrument> <style>  or None",
          "melody": "<instrument> <style>  or None",
          "chords_base": [
          "<instrument> <style>",
          // add more if needed
          ],
        },
      "chords": [
        {"c": "<chord>", "dur": <measures>},
        {"c": "<chord>", "dur": <measures>},
        ...
      ],
      "repeats": <how many times sequence is repeated>,
    },
    "<section 2 name>": {
      "scale": "<scale_tonic> <scale_type>",
      "bpm": <bpm>,
      "time_signature": "<time signature>",
        "parts": {
          "bass": "<instrument> <style>  or None",
          ...
        },
      "chords": [
        {"c": "<chord>", "dur": <measures>},
        {"c": "<chord>", "dur": <measures>},
        ...
      ],
      "repeats": <how many times sequence is repeated>,
    },
    ...
  },
  "structure": [
    {"s": "<section 1 name>", "val": <-1 to 1>, "ar": <0 to 1>},
    {"s": "<section 2 name>", "val": <-1 to 1>, "ar": <0 to 1>},
    {"s": "<section 1 name>", "val": <-1 to 1>, "ar": <0 to 1>},
    ...
    {"s": "<section 4 name>", "val": <-1 to 1>, "ar": <0 to 1>}
  ],
  "com": "<your very short, max 3 sentence comment about your choices in the song and how they relate to user's input>"
}
```

### Citation
If you use this repository, please cite it using the following BibTeX entry:

```bibtex
@misc{pocwiardowski2024m6gpt3,
      title={M6(GPT)3: Generating Multitrack Modifiable Multi-Minute MIDI Music from Text using Genetic algorithms, Probabilistic methods and GPT Models in any Progression and Time signature}, 
      author={Jakub Poćwiardowski and Mateusz Modrzejewski and Marek S. Tatara},
      year={2024},
      eprint={2409.12638},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2409.12638}, 
}
```


Technologies used: Python, NumPy, Pandas, openai, MidiUtil, Mingus
