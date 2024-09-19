from midiutil import MIDIFile
import mingus.core.chords as chords
from mingus.containers import Note
import random
import os
from generation.evolutionary import GeneticBassGenerator, GeneticMelodyGenerator, GeneticMotifGenerator
from generation.notes_generator import NotesGenerator
from generation.drummer import Drummer
from datetime import datetime

class MidiGenerator:
    """
    A key class handling the whole MIDI generation from JSON.

    Attributes:
        track_instrument_map (dict): A dictionary mapping instrument numbers to MIDI track numbers.
        next_available_track (int): The next available track number to assign for new instruments.
        chord_pattern (list): A list representing the pattern for chord repetitions, used for 'repeated' chord mode.

    """
    def __init__(self):
        self.track_instrument_map = {}
        self.next_available_track = 1
        self.chord_pattern = []

    def create_melody_for_section(self, chords_dict, scale, meter, measures, note_duration, mode, instrument, valence=1, arousal=1, motif=[], bass=[], generations=25, population=64):
        """
        Creates a melody using a genetic algorithm.

        :param chords_dict: A dictionary of chords on which melody is built.
        :param scale: The musical scale to be used for the melody (e.g., 'C Major', 'A Minor').
        :param meter: The meter or time signature of the piece (e.g., (4, 4) for four-four time).
        :param measures: The number of measures to generate the melody for.
        :param note_duration: The fundamental note length for the melody (default is 0.25 in relation to quarter note, which is a sixteenth note).
        :param mode: The playing mode ('melody', 'solo').
        :param instrument: The instrument for which the melody is being generated.
        :param valence: A value representing the emotional valence of the melody. 
        :param arousal: A value representing the emotional arousal of the melody.
        :param motif: A motif track used for harmony score and motif mutation.
        :param bass: A bass track used for harmony score.
        :param generations: The number of generations to run the genetic algorithm for.
        :param population: The population size for the genetic algorithm.

        :return: A generated melody as a sequence of notes with -1 indicating pauses and -2 indicating extensions.
        """
        tracks = []
        notes_gen = NotesGenerator(key=scale)
        note_range = self.get_instrument_range(instrument)
        print(f'Generating melody for: {instrument}')
        NOTES = notes_gen.generate_chromatic_notes(note_range)
        scale_notes = notes_gen.generate_notes(number_of_octaves=7, start_octave=2)
        gen_music = GeneticMelodyGenerator(notes=NOTES, scale_notes=scale_notes, note_duration=note_duration,
                                           meter=meter, chords_dict=chords_dict, mode=mode,
                                           valence=valence, arousal=arousal, motif=motif, bass=bass, num_generations=generations, population_size=population)
        melody = gen_music.run(measures=measures)
        tracks.append(melody)
        return melody

    def create_bass_for_section(self, chords_dict, scale, meter, measures, note_duration, mode, instrument, valence=1, arousal=1, generations=25, population=64):
        """
        Creates a bass for sections using a genetic algorithm or probabilistic methods.

        :param chords_dict: A dictionary of chords on which melody is built.
        :param scale: The musical scale to be used for the melody (e.g., 'C Major', 'A Minor').
        :param meter: The meter or time signature of the piece (e.g., (4, 4) for four-four time).
        :param measures: The number of measures to generate the melody for.
        :param note_duration: The fundamental note length for the bass.
        :param mode: The playing mode ('riffs', 'grooves').
        :param instrument: The instrument for which the bass is being generated.
        :param valence: A value representing the emotional valence of the bass melody. 
        :param arousal: A value representing the emotional arousal of the bass melody.
        :param generations: The number of generations to run the genetic algorithm for.
        :param population: The population size for the genetic algorithm.

        :return: A generated bass as a sequence of notes with -1 indicating pauses and -2 indicating extensions.
        """
        tracks = []
        notes_gen = NotesGenerator(key=scale)
        print(f'Generating bass for: {instrument}')
        NOTES = notes_gen.generate_notes(number_of_octaves=2, start_octave=2)
        scale_notes = notes_gen.generate_notes(number_of_octaves=4, start_octave=1)
        gen_music = GeneticBassGenerator(notes=NOTES, scale_notes=scale_notes, note_duration=note_duration,
                                           meter=meter, chords_dict=chords_dict, mode=mode,
                                           valence=valence, arousal=arousal, num_generations=generations, population_size=population)
        melody = gen_music.run(measures=measures)
        tracks.append(melody)
        return melody

    def create_motif_for_section(self, chords_dict, scale, meter, measures, note_duration, mode, instrument, valence=1, arousal=1, start_octave=6, generations=25, population=64):
        """
        Creates a motif for section using a genetic algorithm.

        :param chords_dict: A dictionary of chords on which melody is built.
        :param scale: The musical scale to be used for the melody (e.g., 'C Major', 'A Minor').
        :param meter: The meter or time signature of the piece (e.g., (4, 4) for four-four time).
        :param measures: The number of measures to generate the melody for.
        :param note_duration: The fundamental note length for the motif.
        :param mode: The playing mode ('e.g. 'short_motif', 'start_motif').
        :param instrument: The instrument for which the motif is being generated.
        :param valence: A value representing the emotional valence of the melody. 
        :param arousal: A value representing the emotional arousal of the melody.
        :param start_octave: The lowest octave for which the motif will be generated.
        :param generations: The number of generations to run the genetic algorithm for.
        :param population: The population size for the genetic algorithm.

        :return: A generated melody as a sequence of notes with -1 indicating pauses and -2 indicating extensions.
        """
        tracks = []
        notes_gen = NotesGenerator(key=scale)
        print(f'Generating motif for: {instrument}')
        NOTES = notes_gen.generate_notes(number_of_octaves=2, start_octave=start_octave)
        scale_notes = notes_gen.generate_notes(number_of_octaves=4, start_octave=5)
        gen_music = GeneticMotifGenerator(notes=NOTES, scale_notes=scale_notes, note_duration=note_duration,
                                           meter=meter, chords_dict=chords_dict, mode=mode,
                                           valence=valence, arousal=arousal, num_generations=generations, population_size=population)
        melody = gen_music.run(measures=measures)
        tracks.append(melody)
        return melody

    def add_melody_to_midi(self, midi, melody, instrument, arousal=0.8, start_time=0, default_note_duration=0.25, limiter=0, tempo=120):
        """
        Adds a melodic tracks to MIDI.

        This method adds a sequence of notes (melody) to a MIDI track associated with a specified instrument.
        It configures the track with the appropriate instrument program change, tempo, and then adds notes to the track
        with calculated velocities based on arousal.

        :param midi: The MIDI object to which the melody will be added.
        :param melody: A list of notes to be added to the MIDI track. Notes should be integers representing MIDI note numbers, or special values (-1, -2) for rests or duration extensions.
        :param instrument: The instrument for which the melody is being added. If the instrument is not already assigned, it will be assigned to the next available track.
        :param arousal: A value between 0 and 1 representing the arousal level, used to calculate note velocities.
        :param start_time: The starting time in the MIDI track where the melody begins.
        :param default_note_duration: The fundamental note length (in relation to quarter note).
        :param limiter: A value to adjust the velocity of notes, added to the calculated velocity, used for better audibility of all tracks.
        :param tempo: The tempo of the MIDI track in beats per minute.

        """
        if instrument is not None:

            if instrument in self.track_instrument_map:
                track = self.track_instrument_map[instrument]
            else:
                track = self.next_available_track
                self.next_available_track += 1
                if self.next_available_track == 3:
                    self.next_available_track += 1

                self.track_instrument_map[instrument] = track

                midi.addProgramChange(track, track, start_time, int(instrument))
            midi.addTempo(track, start_time, tempo)

            time = 0
            i = 0
            max_iterations = 1000
            iteration_count = 0
            while i < len(melody):

                iteration_count += 1
                if iteration_count > max_iterations:
                    break
                note = melody[i]
                note_duration = default_note_duration

                if note == -1 or note == -2:
                    time += default_note_duration
                    i += 1

                elif note >= 0:
                    j = i + 1

                    while j < len(melody) and melody[j] == -2:
                        note_duration += default_note_duration
                        j += 1

                    velocity = self.calculate_velocity_from_arousal(arousal) + limiter
                    midi.addNote(track, track, note, start_time + time, note_duration, max(0,velocity))

                    i = j
                    time += note_duration

    def calculate_velocity_from_arousal(self, arousal):
        """
        Calculates the velocity of a note based on the arousal level.

        This method calculates the velocity for a note in the MIDI track based on the given arousal level. Higher arousal values result in higher velocities.

        :param arousal: A value between 0 and 1 representing the arousal level.

        :return: An integer representing the velocity of the note, adjusted within the range of 15 to 100.
        """
        return random.randint(max(15, int(arousal * 70 + 10)), min(100, int(arousal * 70 + 25)))

    def normalize_val_aro_plane(self, value):
        """
        Normalizes a value to a scale between 0 and 1.

        :param value: The input value to be normalized, expected to be within the range of [-1, 1].

        :return: A float representing the normalized value in the range of [0, 1].
        """
        return (value + 1) / 2

    def add_chords(self, midi, time, duration, chord_notes, valence, arousal, instrument_dict, tempo):
        """
        Adds chords base to a MIDI track with varying techniques based on the specified parameters.

        This method inserts chords into a MIDI track, adjusting their characteristics (such as note duration and velocity)
        based on the given valence and arousal levels. It handles different chord techniques (sustained, repeated, arpeggiated)
        and ensures that chords are added with appropriate timing and velocity.

        :param midi: The MIDI object to which the chords will be added.
        :param time: The starting time in the MIDI track for the chord progression.
        :param duration: The total duration of the chords to generate.
        :param chord_notes: A list of notes (in letter notation).
        :param valence: A value between 0 and 1 representing the valence level, affecting notes pitch register.
        :param arousal: A value between 0 and 1 representing the arousal level, influencing note alterations and durations.
        :param instrument_dict: A dictionary specifying the instrument and technique for the chords.
        :param tempo: The tempo of the MIDI track in beats per minute.

        :return: None
        """

        if "chords_base" in instrument_dict and instrument_dict["chords_base"][0] not in [None, "None"]:
            for i, layer in enumerate(instrument_dict["chords_base"]):
                chord_time = time
                instrument, technique = layer.split()
                instrument_nr = self.get_general_midi_number(instrument, 'chord')
                if instrument_nr is None:
                    instrument_nr = 1

                if instrument_nr in self.track_instrument_map:
                    track = self.track_instrument_map[instrument_nr]
                else:

                    track = self.next_available_track
                    self.next_available_track += 1
                    if self.next_available_track == 3:
                        self.next_available_track += 1

                    self.track_instrument_map[instrument_nr] = track

                    midi.addProgramChange(track, track, chord_time, int(instrument_nr))
                midi.addTempo(track, chord_time, tempo)

                if technique == 'repeated':
                    avg_corrector = -3
                elif technique in ["broken", "arpeggiated"]:
                    avg_corrector = 3
                else:
                    avg_corrector = -3 * i

                target_avg = int(45 + valence * 24) + avg_corrector

                octave = 4
                notes = []
                for note in chord_notes:

                    note_value = int(Note(note, octave))
                    if notes and note_value <= notes[-1]:
                        note_value += 12
                    notes.append(note_value)
                    current_avg = sum(notes) / len(notes)

                if arousal < 0.3 or (len(notes) > 4 and arousal < 0.5):
                    notes.pop(2)
                elif arousal < 0.7:
                    pass
                elif arousal > 0.7 and len(notes) < 5:
                    notes.append(notes[0] + 12)
                if arousal > 0.9 and len(notes) < 6:
                    notes.append(notes[0] + 19)


                iterations = 0
                previous_solutions = {}
                closest_solution = None
                closest_difference = float('inf')

                while iterations < 100:
                    current_avg = sum(notes) / len(notes)
                    current_difference = abs(current_avg - target_avg)

                    if current_difference < closest_difference:
                        closest_solution = current_avg
                        closest_difference = current_difference

                    if current_avg in previous_solutions and current_avg == closest_solution:
                        break

                    if current_avg < target_avg:
                        min_note_index = notes.index(min(notes))
                        notes[min_note_index] += 12
                    elif current_avg > target_avg:
                        max_note_index = notes.index(max(notes))
                        notes[max_note_index] -= 12

                    previous_solutions[current_avg] = True
                    iterations += 1


                notes.sort()
                for i in range(len(notes) - 1):
                    if abs(notes[i] - notes[i + 1]) < 1:
                        if current_avg < target_avg:
                            notes[i] += 12
                        else:
                            notes[i + 1] -= 12

                        current_avg = sum(notes) / len(notes)

                notes.sort()
                if arousal >= 0.8:
                    note_duration = 0.25
                elif arousal >= 0.5:
                    note_duration = 0.5
                elif arousal > 0.2:
                    note_duration = 1
                else:
                    note_duration = 2

                if technique == 'repeated':
                    if not self.chord_pattern:
                        probability_of_playing = 0.5 + arousal * 0.4
                        repetitions = int(duration / note_duration)
                        self.chord_pattern = [1] + [random.random() < probability_of_playing for _ in range(repetitions - 1)]

                    remaining_duration = duration
                    pattern_index = 0
                    while remaining_duration > 0:
                        if remaining_duration < note_duration:
                            note_duration = remaining_duration

                        if self.chord_pattern[pattern_index % len(self.chord_pattern)]:
                            for note_value in notes:
                                velocity = self.calculate_velocity_from_arousal(arousal)
                                midi.addNote(track, track, note_value, chord_time, note_duration / 2, velocity)
                        pattern_index += 1
                        chord_time += note_duration
                        remaining_duration -= note_duration
                elif technique in ["broken", "arpeggiated", "arpeggio"]:
                    remaining_duration = duration
                    direction = 1
                    note_index = 0

                    while remaining_duration > 0:
                        if remaining_duration < note_duration:
                            note_duration = remaining_duration
                        velocity = self.calculate_velocity_from_arousal(arousal)
                        midi.addNote(track, track, notes[note_index], chord_time, note_duration, velocity)

                        chord_time += note_duration
                        remaining_duration -= note_duration

                        note_index += direction

                        if note_index == len(notes):
                            note_index = len(notes) - 2
                            direction = -1

                        elif note_index < 0:
                            note_index = 1
                            direction = 1
                else:
                    for note_value in notes:
                        velocity = self.calculate_velocity_from_arousal(arousal)
                        midi.addNote(track, track, note_value, chord_time, duration, velocity)

    def get_instrument_range(self, instrument_name):
        """
        Retrieves the note range for a specified instrument.

        This method provides the range of notes (octaves) that the given instrument is capable of playing.

        :param instrument_name: The name of the instrument whose note range is to be retrieved.

        :return: A tuple representing the range of notes (start and end) for the specified instrument, or a default range if the instrument is not recognized.
        """
        instrument_ranges = {
            'piano': ('A-3', 'C-7'),
            'guitar': ('E-3', 'E-7'),
            'electric_guitar': ('E-3', 'E-7'),
            'acoustic_guitar': ('E-3', 'E-7'),
            'strings': ('G-3', 'A-7'),
            'flute': ('C-4', 'D-7'),
            'trumpet': ('A-3', 'E-6'),
            'brass': ('A-3', 'E-6'),
            'sax': ('A-3', 'E-6'),
            'synth': ('A-3', 'C-7'),
            'organ': ('A-3', 'C-7'),
        }
        if instrument_name in instrument_ranges:
            return instrument_ranges.get(instrument_name.lower())
        else:
            return ('E-3', 'E-7')

    def get_general_midi_number(self, instrument_name, type):
        """
        Retrieves the General MIDI number for a specified instrument based on its type.

        This method maps the instrument name to its corresponding General MIDI number, considering its type (e.g., lead, bass, chord).
        It provides a fallback to default values if the instrument is not found.

        :param instrument_name: The name of the instrument to be mapped to a General MIDI number.
        :param type: The type of instrument, which can influence the resulting MIDI number (e.g., 'bass', 'melody', 'chord').

        :return: An integer representing the General MIDI number for the specified instrument and type, or 0 if not found.
        """
        if instrument_name is None or instrument_name.lower() == "none":
            return None
        lead_instruments = {
            "piano": 2,
            "electric_guitar": 31,
            "acoustic_guitar": 26,
            "viola": 42,
            "violin": 41,
            "synth": 88,
            "organ": 18,
            "brass": 57,
            "sax": 65,
            "flute": 74,
            "trumpet": 57,
            "choir": 55,
            "voice_oohs": 54,
            "choir_aahs": 53,
        }

        bass_instruments = {
            "piano": 1,
            "guitar": 26,
            "acoustic_guitar": 33,
            "acoustic": 33,
            "acoustic_bass": 33,
            "bass": 34,
            "electric_bass": 34,
            "slapped": 37,
            "plucked": 36,
            "picked": 35,
            "pizzicato_strings": 46,
            "contrabass": 44,
            "synth": 39,
            "organ": 19,
            "brass": 59,
        }

        chord_instruments = {
            "piano": 0,
            "electric_piano": 5,
            "electric_guitar": 30,
            "acoustic_guitar": 25,
            "strings": 49,
            "synth_strings": 52,
            "synth": 51,
            "organ": 17,
            "hammond_organ": 19,
            "rock_organ": 19,
            "church_organ": 20,
            "pad": 89,
            "harpsichord": 7,
            "brass": 62,
            "choir": 55,
            "voice_oohs": 54,
            "choir_aahs": 53,
        }

        motif_instruments = {
            "piano": 0,
            "guitar": 28,
            "strings": 46,
            "synth": 99,
            "glockenspiel": 10,
            "marimba": 13,
            "harp": 47,
            "violin": 41,
            "flute": 74,
        }
        all_instruments = [lead_instruments, bass_instruments, chord_instruments, motif_instruments]
        if type.lower() == 'bass':
            result = bass_instruments.get(instrument_name.lower())
        elif type.lower() == 'melody':
            result = lead_instruments.get(instrument_name.lower())
        elif type.lower() in ['motif', 'ornaments']:
            result = motif_instruments.get(instrument_name.lower())
        elif type.lower() == 'chord':
            result = chord_instruments.get(instrument_name.lower())
        else:
            return 0

        if result is None:
            for instruments_dict in all_instruments:
                result = instruments_dict.get(instrument_name.lower())
                if result is not None:
                    break

        return result

    def get_instruments(self, dictionary, section):
        """
        Retrieves the instrument and technique for a specified section from a dictionary.

        :param dictionary: A dictionary containing instrument and technique information.
        :param section: The key in the dictionary for which the instrument and technique are to be retrieved.

        :return: A tuple containing the instrument and technique. Returns (None, None) if the section is 'None' or similar.
        """
        if dictionary[section] in ['None', 'none', 'None']:
            return None, None
        instrument, technique = dictionary[section].split()

        return instrument, technique

    def generate_midi_from_json(self, data, generations, population):
        """
        The main method to generates a MIDI file from a JSON structure provided by the LLM.

        :param data: A dictionary containing data from JSON file.
        :param generations: Number of generations for genetic algorithm.
        :param population: Population size for genetic algorithm.
    
        :return: The file path to the generated MIDI file.
        """
        sections = data['sections']
        structure = data['structure']
        comment = data['com']
        song_name = data['name']
        print("JSON Structure:")
        print(structure)
        print(f'Chat: {comment}')

        midi = MIDIFile(50)
        time = 0
        melody_time = 0
        note_duration = 0.25 # we employ 16th notes as the fundamental note duration
        section_data = {}
        for section_dict in structure:
            section_name = section_dict['s']
            # the valence range in JSON is <-1,1> and valence is <0,1> as these ranges are better understandable for the LLM
            valence = self.normalize_val_aro_plane(section_dict['val'])
            arousal = section_dict['ar']

            scale = sections[section_name]['scale']
            tempo = sections[section_name]['bpm']
            meter = tuple(map(int, sections[section_name]['time_signature'].split('/')))
            repeats = sections[section_name]['repeats']

            instrument_dict = sections[section_name]['parts']
            bass_sound, bass_technique = self.get_instruments(instrument_dict, "bass")
            motif_sound, motif_technique = self.get_instruments(instrument_dict, "motif")
            perc_sound, perc_technique = self.get_instruments(instrument_dict, "percussion")
            melody_sound, melody_technique = self.get_instruments(instrument_dict, "melody")
            motif_nr = self.get_general_midi_number(motif_sound, 'motif')
            bass_nr = self.get_general_midi_number(bass_sound, 'bass')
            melody_nr = self.get_general_midi_number(melody_sound, 'melody')

            # possible adjustments to fundamental note duration
            if melody_technique == 'melody':
                lead_multiplier = 2
            else:
                lead_multiplier = 1

            if arousal < 0.5 and meter[1] <= 4:
                motif_multiplier = 2
            else:
                motif_multiplier = 1

            chords_dict = {}

            if section_name in section_data:
                lead_melody, melodies, basses, drums, section_note_duration = section_data[section_name]
                if perc_sound is not None:
                    drummer = Drummer(note_duration=note_duration * 2, bpm=tempo, meter=meter,
                                      valence=valence, arousal=arousal, kit=perc_sound)
                    drummer.write_drums(midi, 3, drums, start_time=time)
                for _ in range(repeats):
                    for chord in sections[section_name]['chords']:
                        chords_dict[section_duration * 4 * meter[0] / meter[1]] = chord['c']
                        chord_notes = chords.from_shorthand(chord['c'])
                        for _ in range(chord['dur']):
                            self.add_chords(midi, time, 4 * meter[0] / meter[1], chord_notes, valence,
                                            arousal, instrument_dict, tempo)
                            time += 4 * meter[0] / meter[1]

            else:
                melodies = []
                basses = []
                lead_melody = []
                drums = []
                section_duration = 0
                for _ in range(repeats):
                    for chord in sections[section_name]['chords']:
                        print(chord)
                        chords_dict[section_duration * 4 * meter[0] / meter[1]] = chord['c']
                        chord_notes = chords.from_shorthand(chord['c'])
                        for _ in range(chord['dur']):
                            self.add_chords(midi, time, 4 * meter[0] / meter[1], chord_notes, valence, arousal, instrument_dict, tempo)
                            time += 4 * meter[0] / meter[1]
                        section_duration += chord['dur']

                if perc_technique == 'only_beat':
                    fill_probability = 0
                    fill_end_boost = 0
                    fill_frequency = 1
                    fill_beats = 1
                elif perc_technique == 'drum_solo':
                    fill_beats = meter[0]
                    fill_frequency = 1
                    fill_probability = 1
                    fill_end_boost = 1
                else:
                    fill_probability = 0.4
                    fill_frequency = 2
                    fill_beats = meter[0] // 2
                    fill_end_boost = 0.5

                if perc_sound is not None:
                    drummer = Drummer(note_duration=note_duration * 2, bpm=tempo, meter=meter,
                                      valence=valence, arousal=arousal, kit=perc_sound)
                    drums = drummer.generate_section(measures=section_duration, repeats=1,
                                                     fill_probability=fill_probability,
                                                     fill_beats=fill_beats, fill_frequency=fill_frequency,
                                                     fill_end_boost=fill_end_boost)
                    drummer.write_drums(midi, 3, drums, start_time=melody_time)
                if bass_sound not in ["None", None]:
                    basses = self.create_bass_for_section(chords_dict, scale, meter, measures=section_duration,
                                                                 note_duration=note_duration * 2,
                                                                 mode=bass_technique, instrument=bass_sound,
                                                                 valence=valence, arousal=arousal, generations=generations, population=population)
                if motif_sound not in ["None", None]:
                    if melody_sound in ["None", None] or melody_technique in ['melody']:
                        start_octave = 6
                    else:
                        start_octave = 7
                    melodies = self.create_motif_for_section(chords_dict, scale, meter, measures=section_duration,
                                                                 note_duration=note_duration * motif_multiplier,
                                                                 mode=motif_technique, instrument=motif_sound,
                                                                 valence=valence, arousal=arousal, start_octave=start_octave, generations=generations, population=population)

                if melody_sound not in ["None", None]:
                    lead_melody = self.create_melody_for_section(chords_dict, scale, meter, measures=section_duration,
                                                                 note_duration=note_duration * lead_multiplier,
                                                                 mode=melody_technique, instrument=melody_sound,
                                                                 valence=valence, arousal=arousal, motif=melodies, bass=basses, generations=generations, population=population)
                section_note_duration = section_duration * 4 * meter[0] / meter[1]
                section_data[section_name] = (lead_melody, melodies, basses, drums, section_note_duration)
            self.add_melody_to_midi(midi=midi, melody=lead_melody, arousal=arousal, start_time=melody_time,
                                    default_note_duration=note_duration * lead_multiplier, instrument=melody_nr, tempo=tempo)
            if motif_sound not in ["None", None]:
                if melody_sound in ["None", None]:
                    limiter = -0
                elif melody_technique in ['melody']:
                    limiter = -10
                else:
                    limiter = -20
                self.add_melody_to_midi(midi=midi, melody=melodies, arousal=max(0, arousal - 0.2), start_time=melody_time,
                                        default_note_duration=note_duration * motif_multiplier, instrument=motif_nr, limiter=limiter, tempo=tempo)
            self.add_melody_to_midi(midi=midi, melody=basses, arousal=arousal, start_time=melody_time,
                                    default_note_duration=note_duration * 2, instrument=bass_nr, tempo=tempo)
            melody_time += section_note_duration
        ending = [int(Note(scale.split()[0], 2))] + [-2] * (meter[0] - 1)
        self.add_melody_to_midi(midi=midi, melody=ending, arousal=arousal, start_time=melody_time,
                                default_note_duration=8 / meter[1], instrument=1, tempo=tempo)

        generated_midi_dir = os.path.join('.', 'generated_midi')

        if not os.path.exists(generated_midi_dir):
            os.makedirs(generated_midi_dir)

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        midi_filename = f'{song_name}_{current_time}.mid'
        midi_filepath = os.path.join(generated_midi_dir, midi_filename)

        with open(midi_filepath, 'wb') as midi_file:
            midi.writeFile(midi_file)

        return midi_filepath

    def generate_midi_random(self, generations, population):
        """
        Variation of the main method with randomized parameters. Used for generating "Randomized" songs in evaluation.

        :param generations: Number of generations for genetic algorithm.
        :param population: Population size for genetic algorithm.

        :return: The file path to the generated MIDI file.
        """
        sections = {
        "Intro": {
            "scale": "C Major",
            "bpm": 120,
            "time_signature": "4/4",
            "parts": {
                "bass": "guitar riff",
                "motif": "marimba long",
                "percussion": "standard full",
                "melody": "sax solo",
                "chords_base": [
                    "piano sustained"
                ]
            },
            "chords": [
                {
                    "c": "C",
                    "dur": 1
                },
                {
                    "c": "G",
                    "dur": 1
                },
                {
                    "c": "Am",
                    "dur": 1
                },
                {
                    "c": "F",
                    "dur": 1
                }
            ],
            "repeats": 5
        }
    }
        structure = [
            {
                "s": "Intro",
                "val": 0.0,
                "ar": 0.5
            }
        ]
        print("JSON Structure:")
        print(structure)

        midi = MIDIFile(50)
        time = 0
        melody_time = 0
        note_duration = 0.25
        section_data = {}
        for section_dict in structure:
            section_name = section_dict['s']
            valence = random.uniform(0, 1)
            arousal = random.uniform(0, 1)
            scale = random.choice(["C Major", "A Minor"])
            tempo = random.randint(40, 240)
            first_meter_value = random.randint(3, 15)
            second_meter_value = random.choice([4, 8, 16])
            while first_meter_value > 2 * second_meter_value:
                second_meter_value *= 2
            while first_meter_value * 2 < second_meter_value:
                second_meter_value /= 2
            meter = (first_meter_value, second_meter_value)
            repeats = sections[section_name]['repeats']

            print(f"Valence: {valence}")
            print(f"Arousal: {arousal}")
            print(f"Scale: {scale}")
            print(f"Tempo: {tempo}")
            print(f"Meter: {meter}")

            instrument_dict = sections[section_name]['parts']

            # bass
            if random.random() < 0.3:
                bass_sound, bass_technique = None, None
            else:
                bass_sound = 'guitar'
                bass_technique = random.choice(["groove", "riff"])

            # motif
            if random.random() < 0.3:
                motif_sound, motif_technique = None, None
            else:
                motif_sound = 'marimba'
                motif_technique = random.choice(
                    ['repeated_motif', 'repeated_short_motif', 'end_motif', 'start_motif', 'long_motif'])

            # percussion
            if random.random() < 0.3:
                perc_sound, perc_technique = None, None
            else:
                perc_sound = random.choice(["standard", "ethnic", "bells_and_cymbals"])
                perc_technique = random.choice(["only_beat", "full", "drum_solo"])

            # melody
            if random.random() < 0.3:
                melody_sound, melody_technique = None, None
            else:
                melody_sound = 'sax'
                melody_technique = random.choice(["solo", "melody"])

            print(f"Bass Sound: {bass_sound}, Bass Technique: {bass_technique}")
            print(f"Motif Sound: {motif_sound}, Motif Technique: {motif_technique}")
            print(f"Perc Sound: {perc_sound}, Perc Technique: {perc_technique}")
            print(f"Melody Sound: {melody_sound}, Melody Technique: {melody_technique}")

            motif_nr = self.get_general_midi_number(motif_sound, 'motif')
            bass_nr = self.get_general_midi_number(bass_sound, 'bass')
            melody_nr = self.get_general_midi_number(melody_sound, 'melody')

            if melody_technique == 'melody':
                lead_multiplier = 2
            else:
                lead_multiplier = 1

            if arousal < 0.5 and meter[1] <= 4:
                motif_multiplier = 2
            else:
                motif_multiplier = 1

            chords_dict = {}

            if section_name in section_data:
                lead_melody, melodies, basses, drums, section_note_duration = section_data[section_name]
                if perc_sound is not None:
                    drummer = Drummer(note_duration=note_duration * 2, bpm=tempo, meter=meter,
                                      valence=valence, arousal=arousal, kit=perc_sound)
                    drummer.write_drums(midi, 3, drums, start_time=time)
                for _ in range(repeats):
                    for chord in sections[section_name]['chords']:
                        chord_random = random.choice(['C', 'Dm', 'Em', 'F', 'G', 'Am', 'Bdim'])
                        print(chord_random)
                        chords_dict[section_duration * 4 * meter[0] / meter[1]] = chord_random
                        chord_notes = chords.from_shorthand(chord_random)
                        for _ in range(chord['dur']):
                            self.add_chords(midi, time, 4 * meter[0] / meter[1], chord_notes, valence,
                                            arousal, instrument_dict, tempo)
                            time += 4 * meter[0] / meter[1]

            else:
                melodies = []
                basses = []
                lead_melody = []
                drums = []
                section_duration = 0
                for _ in range(repeats):
                    for chord in sections[section_name]['chords']:
                        chord_random = random.choice(['C', 'Dm', 'Em', 'F', 'G', 'Am', 'Bdim'])
                        print(chord_random)
                        chords_dict[section_duration * 4 * meter[0] / meter[1]] = chord_random
                        chord_notes = chords.from_shorthand(chord_random)
                        for _ in range(chord['dur']):
                            self.add_chords(midi, time, 4 * meter[0] / meter[1], chord_notes, valence, arousal,
                                            instrument_dict, tempo)
                            time += 4 * meter[0] / meter[1]
                        section_duration += chord['dur']

                if perc_technique == 'only_beat':
                    fill_probability = 0
                    fill_end_boost = 0
                    fill_frequency = 1
                    fill_beats = 1
                elif perc_technique == 'drum_solo':
                    fill_beats = meter[0]
                    fill_frequency = 1
                    fill_probability = 1
                    fill_end_boost = 1
                else:
                    fill_probability = 0.4
                    fill_frequency = 2
                    fill_beats = meter[0] // 2
                    fill_end_boost = 0.5

                if perc_sound is not None:
                    drummer = Drummer(note_duration=note_duration * 2, bpm=tempo, meter=meter,
                                      valence=valence, arousal=arousal, kit=perc_sound)
                    drums = drummer.generate_section(measures=section_duration, repeats=1,
                                                     fill_probability=fill_probability,
                                                     fill_beats=fill_beats, fill_frequency=fill_frequency,
                                                     fill_end_boost=fill_end_boost)
                    drummer.write_drums(midi, 3, drums, start_time=melody_time)
                if bass_sound not in ["None", None]:
                    basses = self.create_bass_for_section(chords_dict, scale, meter, measures=section_duration,
                                                          note_duration=note_duration * 2,
                                                          mode=bass_technique, instrument=bass_sound,
                                                          valence=valence, arousal=arousal, generations=generations,
                                                          population=population)
                if motif_sound not in ["None", None]:
                    if melody_sound in ["None", None] or melody_technique in ['melody']:
                        start_octave = 6
                    else:
                        start_octave = 7
                    melodies = self.create_motif_for_section(chords_dict, scale, meter, measures=section_duration,
                                                             note_duration=note_duration * motif_multiplier,
                                                             mode=motif_technique, instrument=motif_sound,
                                                             valence=valence, arousal=arousal,
                                                             start_octave=start_octave, generations=generations,
                                                             population=population)

                if melody_sound not in ["None", None]:
                    lead_melody = self.create_melody_for_section(chords_dict, scale, meter, measures=section_duration,
                                                                 note_duration=note_duration * lead_multiplier,
                                                                 mode=melody_technique, instrument=melody_sound,
                                                                 valence=valence, arousal=arousal, motif=melodies,
                                                                 bass=basses, generations=generations,
                                                                 population=population)
                section_note_duration = section_duration * 4 * meter[0] / meter[1]
                section_data[section_name] = (lead_melody, melodies, basses, drums, section_note_duration)
            self.add_melody_to_midi(midi=midi, melody=lead_melody, arousal=arousal, start_time=melody_time,
                                    default_note_duration=note_duration * lead_multiplier, instrument=melody_nr,
                                    tempo=tempo)
            if motif_sound not in ["None", None]:
                if melody_sound in ["None", None]:
                    limiter = -0
                elif melody_technique in ['melody']:
                    limiter = -10
                else:
                    limiter = -20
                self.add_melody_to_midi(midi=midi, melody=melodies, arousal=max(0, arousal - 0.2),
                                        start_time=melody_time,
                                        default_note_duration=note_duration * motif_multiplier, instrument=motif_nr,
                                        limiter=limiter, tempo=tempo)
            self.add_melody_to_midi(midi=midi, melody=basses, arousal=arousal, start_time=melody_time,
                                    default_note_duration=note_duration * 2, instrument=bass_nr, tempo=tempo)
            melody_time += section_note_duration

        generated_midi_dir = os.path.join('.', 'generated_midi')

        if not os.path.exists(generated_midi_dir):
            os.makedirs(generated_midi_dir)

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        midi_filename = f'Randomized_{current_time}.mid'
        midi_filepath = os.path.join(generated_midi_dir, midi_filename)

        with open(midi_filepath, 'wb') as midi_file:
            midi.writeFile(midi_file)

        return midi_filepath
