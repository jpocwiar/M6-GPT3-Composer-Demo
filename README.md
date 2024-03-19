# GGPT-Composer-Demo

This work introduces the Genetic GPT Song Composer system, capable of generating complete, multi-minute musical compositions with complex structures in the MIDI domain from input descriptions in natural language. The system utilizes the GPT language model in the form of ChatGPT API to map natural language prompts to composition parameters in JSON format. The defined structure includes time signature, scales, chord progressions, and valence-arousal values, from which accompaniment, melody, bass, motif, and percussion tracks are created. The generation of melodic elements is achieved through a genetic algorithm. The algorithm incorporates mutations with musical significance and a fitness function based on normal distribution and predefined musical feature values. The values adaptively evolve, influenced by emotional parameters and distinct playing styles. System for generating percussion in any time signature utilises probabilistic methods, including Markov chains. The test results showed that although the system does not match the most advanced machine learning models in terms of fidelity to reproducing individual genres and sounds, its advanced knowledge of music theory and styles, along with the absence of limitations imposed by dominant musical structures in extensive datasets, can make the system a valuable tool for musicians seeking inspiration.

#### Example generations
To present the functionality of the system, I generated couple of songs using descriptions from [Meta's MusicGen](https://audiocraft.metademolab.com/musicgen.html) and [Google's MusicLM](https://google-research.github.io/seanet/musiclm/examples/) sites. Presented wav files are synthesized from MIDI, so they have quite basic instrument sounding. They can however be used to synthesize with finer samples.

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Structure of the Song "Arcade Rush"</title>
</head>
<body>
    <table border="1">
        <caption>Structure of the Song "Arcade Rush" / Struktura utworu "Arcade Rush"</caption>
        <thead>
            <tr>
                <th>Element</th>
                <th>Details / Szczegóły</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Song Name / Nazwa utworu</td>
                <td>Arcade Rush</td>
            </tr>
            <tr>
                <td>Tempo / Tempo</td>
                <td>160 BPM (beats per minute) / 160 uderzeń na minutę</td>
            </tr>
            <tr>
                <td>Scales Used / Użyte skale</td>
                <td>E Mixolydian, E Dorian, E Minor / E Miksolidyjska, E Dorycka, E Molowa</td>
            </tr>
            <tr>
                <td>Instruments Used / Użyte instrumenty</td>
                <td>Guitar, harp, accent percussion, synthesizer, strings, standard drums / Gitara, harfa, perkusja akcentowa, syntezator, smyczki, perkusja standardowa</td>
            </tr>
            <tr>
                <td>Structure / Struktura</td>
                <td>Intro, main theme, break, bridge, main theme, outro / Intro, główny motyw, przełamanie, mostek, główny motyw, outro</td>
            </tr>
            <tr>
                <td>Emotions / Emocje</td>
                <td>(Link to emotions graph / Link do wykresu emocji)</td>
            </tr>
            <tr>
                <td>Link to the song / Link do utworu</td>
                <td><a href="https://drive.google.com/file/d/1wDoxxGhYYEFYoxxLyuBSJiZffkRHGxPj/view?usp=drive_link">Arcade Rush in WAV format / Arcade Rush w formacie wav</a></td>
            </tr>
            <tr>
                <td>Listen here / Posłuchaj tutaj</td>
                <td>
                    <audio controls>
                        <source src="https://drive.google.com/uc?export=view&id=1wDoxxGhYYEFYoxxLyuBSJiZffkRHGxPj" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
        </tbody>
    </table>
</body>
</html>


<table>
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

</table>

Technologies used: Python, PySide6, NumPy, Pandas, openai, matplotlib, MidiUtil, Mingus, PyGame
