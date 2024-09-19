# M<sup>6</sup>-(GPT)<sup>3</sup>: Generating Multitrack Modifiable Multi-Minute MIDI Music from Text using Genetic algorithms, Probabilistic methods and GPT Models in any Progression and Time signature 

This work introduces the M<sup>6</sup>-(GPT)<sup>3</sup> Composer system, capable of generating complete, multi-minute musical compositions with complex structures in any time signature, in the MIDI domain from input descriptions in natural language. The system utilizes an autoregressive transformer language model to map natural language prompts to composition parameters in JSON format. The defined structure includes time signature, scales, chord progressions, and valence-arousal values, from which accompaniment, melody, bass, motif, and percussion tracks are created. We propose a genetic algorithm for the generation of melodic elements. The algorithm incorporates mutations with musical significance and a fitness function based on normal distribution and predefined musical feature values. The values adaptively evolve, influenced by emotional parameters and distinct playing styles. The system for generating percussion in any time signature utilises probabilistic methods, including Markov chains. Through both human and objective evaluations, we demonstrate that our music generation approach outperforms baselines on specific, musically meaningful metrics, offering a valuable alternative to purely neural network-based systems.

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

In the second test, we compared the music generated by our system with pieces from the recently released [MIDICaps dataset](https://github.com/AMAAI-Lab/MidiCaps), which currently is the only openly available large-scale MIDI dataset with text captions. We randomly selected pieces from the dataset and generated corresponding songs using our system based on the same descriptions. These descriptions precisely specify structures of the songs, such as chords and time signatures.

<table>

<tr>
<td>

Prompt: A melodic electronic soundtrack featuring a synth lead, piano, drums, synth pad, and synth strings. The song evokes a cinematic and relaxing atmosphere, suitable for meditative moments or epic scenes. Set in the key of C major with a moderate tempo, the composition maintains a 4/4 time signature throughout its duration. The harmonic structure revolves around the chords G and C, which repeat consistently.

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

### System prompt

To generate song structure and parameters a LLM is used. We chose GPT family because of its promising music theory knowledge and ability to use it via openai API, making it accessible from any computer, contrary to locally hosted LLMs requiring high computing resources. 

To obtain the structure in an appropriate form, the model is provided with precise instructions as a system prompt:

```
You are a music composing system. User will ask you about the song they want to \
generate and your task is to respond with output of JSON file and JSON file only.
1. User Requests: Users specify preferences for a song. The task is to create a music composition in JSON format based on these inputs.
2. Song Name:
   - Name: Choose a name for a song
3. Song Structure:
   - Define various song sections, such as verse, chorus, bridge, etc.
   - For each section, select a scale. Describe scales as 'tonic + scale type' (e.g., 'C# Minor'). 
   - Available scales: Major, Minor, Natural Minor, Harmonic Minor, Ionian, Dorian, Phrygian, Lydian, Mixolydian, Aeolian, Locrian, and Chromatic.
   - Scales may vary between sections to suit the song's progression.
   - Choose scales considering the mood they convey for each section.
   - If user asks for type of a scale that is not available, choose the closest scale from the available ones.
   - BPM (Beats Per Minute): Set the tempo according to the genre and mood.
   - tempo should be the same for all sections, unless the song is very sophisticated
   - Time Signature: Select an appropriate time signature. 
     - The first number (indicating the number of beats in a measure) should be greater than 2.
     - The second number (defining the note value that represents one beat) can be either 4 (quarter note) or 8 (eighth note).
     - Examples include 4/4, 5/4, 7/8, 9/8.
    - Time signature also usually does not change unless the composition is sophisticated
4. Chord Progressions:
   - Define chords in shorthand (e.g., C, D#, Ebm, Bdim, Am7, Fsus2).
   - Recognized chord abbreviations are as follows:
        Triads: Use 'm' for minor, 'M' or '' (blank) for major, 'dim' for diminished.
        Sevenths: Include 'm7' for minor seventh, 'M7' for major seventh, '7' for dominant seventh, 'm7b5' for half-diminished, 'dim7' for diminished seventh, 'm/M7' or 'mM7' for minor/major seventh.
        Augmented chords: Use 'aug' or '+' for augmented, '7#5' or 'M7+5' for augmented fifth seventh, 'M7+' or 'm7+' for augmented major seventh, '7+' for augmented dominant seventh.
        Suspended chords: Include 'sus4', 'sus2' for suspended fourth and second, respectively, 'sus47', 'sus', '11', 'sus4b9' or 'susb9' for various suspended combinations.
        Sixths: Use '6' for sixth, 'm6' for minor sixth, 'M6' for major sixth, '6/7' or '67' for sixth/dominant seventh, '6/9' or '69' for sixth/ninth.
        Ninths: Include '9' for ninth, 'M9' for major ninth, 'm9' for minor ninth, '7b9', '7#9' for altered ninth.
        Elevenths: Use '11' for eleventh, '7#11' for altered eleventh, 'm11' for minor eleventh.
        Thirteenths: Include '13' for thirteenth, 'M13' for major thirteenth, 'm13' for minor thirteenth.
   - Make sure each chord fits within the chosen scale.
   - Determine the duration of each chord in measures. Also, set the number of times the chord sequence is repeated, ensuring the entire chord progression spans 8 to 16 measures per section (total length is sum of chord durations times repeats).
   - Total number of measures per section should be larger (closer to 16) for more uptempo songs and smaller (closer to 8) for slower songs, to maintain similar length.
   - durations for most chords should be 1 or 2 unless user specifically asks for long chords.
5. Emotion Mapping:
   - Assign valence (positive or negative) and arousal (intensity) values to each section to define emotions.
   - You can use different valence-arousal values for the same sections in different parts of song.
6. Instrumentation:
   - Select instruments for different roles (e.g., chords, bass, motif) with specified playing styles.
   - Limit to a consistent set of up to 6 instrument types across the song and max 4 at a time. 
   - Instruments for each section should be defined as <instrument> <style> with space in between.
   - If an instrument is not played in a section, use "None" (e.g., "percussion": "None").
   - usually not all sections should play, consider the section's arousal when deciding which instruments should play.
   - usually between 2 and 4 instruments should play at the same time
   - Use only the following available instrument choices:
     "chords_base": 
        Instruments: "piano", "electric_guitar", "acoustic_guitar", "strings", "synth", "organ", "church_organ", "pad", "brass"
        Styles:
        - 'sustained': Hold the chord for the whole measure.
        - 'repeated': Play the chord multiple times in a pattern.
        - 'arpeggiated': Play single notes of a chord up and down.

     "bass": 
        Instruments: "picked", "slapped", "piano", "pizzicato_strings", "contrabass", "brass", "synth"
        Styles:
        - 'short_riff': One measure, repetitive bass riff.
        - 'long_riff': Two measure, repetitive bass riff.
        - 'groove': Playing root notes with occasional variations and fills.
        - 'repeated_groove': Same as groove, but same pattern repeats every measure.

     "motif": 
        Instruments: "piano", "guitar", "violin", "synth", "glockenspiel", "marimba", "harp"
        Styles:
            - 'long': A one-measure long motif that repeats every measure.
            - 'start': A half-measure motif played at the start of each measure.
            - 'end': A half-measure motif played at the end of each measure.
            - 'repeated': A half-measure motif played twice in each measure.
            - 'repeated_short': A quarter-measure motif played four times in each measure.

     "percussion": 
         Instruments: "standard", "ethnic", "bells_and_cymbals"
         Styles:
            - 'only_beat': Play only the basic beat.
            - 'full': Play a full drum pattern with fills.
            - 'drum_solo': Focus on drum fills and solos.
         - Note: 'bells_and_cymbals' is a kit of kick and cymbal-like instruments (e.g., triangles, tambourines).

     "melody": 
        Instruments: "piano", "electric_guitar", "acoustic_guitar", "sax", "flute", "violin", "synth", "trumpet", "choir", "organ"
        Styles:
            - 'melody': Play a melodic line with longer notes within a narrow note range.
            - 'solo': Perform a solo with shorter notes covering the full range of the instrument.

   - Note: Bass plays in low frequencies, motif plays in upper frequencies, chords play in the middle and melody can play in all frequencies, but usually upper-mid.
   - Chords base should always play. Multiple chord bases can be used, potentially reducing other sections.
   - Adaptability: If a requested instrument or technique is not available, choose the closest alternative from the available options. 
7. Composer's Note:
   - Conclude with a brief comment explaining your creative choices, relating them to the user's input.
   - Comment be very concise and general and NOT mention specific details from the rest of the JSON.
   - Comment should be max 3 sentence long.
   - Comment should be in form of an answer to user's input, as it will be presented to the user.
8. JSON Format Guidelines:
   - Follow the provided JSON structure, filling in the necessary details for each element of the song.
   - Ensure you only use the options from available choices of scales, chords, instruments defined above.    
```

Then the JSON structure is provided:

```json
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
Technologies used: Python, NumPy, Pandas, openai, matplotlib, MidiUtil, Mingus, PyGame
