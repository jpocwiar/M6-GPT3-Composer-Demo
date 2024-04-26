# M3-GPT2 Composer: Multitrack MIDI Music from Text using ChatGPT, Genetic algorithms, and Probabilistic methods

This work introduces the Genetic GPT Song Composer system, capable of generating complete, multi-minute musical compositions with complex structures in the MIDI domain from input descriptions in natural language. The system utilizes the GPT language model in the form of ChatGPT API to map natural language prompts to composition parameters in JSON format. The defined structure includes time signature, scales, chord progressions, and valence-arousal values, from which accompaniment, melody, bass, motif, and percussion tracks are created. The generation of melodic elements is achieved through a genetic algorithm. The algorithm incorporates mutations with musical significance and a fitness function based on normal distribution and predefined musical feature values. The values adaptively evolve, influenced by emotional parameters and distinct playing styles. System for generating percussion in any time signature utilises probabilistic methods, including Markov chains. The test results showed that although the system does not match the most advanced neural, data-driven models in terms of fidelity to reproducing individual genres and sounds, its advanced knowledge of music theory and styles, along with the absence of limitations imposed by dominant musical structures in extensive datasets, can make the system a valuable tool for musicians seeking inspiration.

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

Technologies used: Python, PySide6, NumPy, Pandas, openai, matplotlib, MidiUtil, Mingus, PyGame
