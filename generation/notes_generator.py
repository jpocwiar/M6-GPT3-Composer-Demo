import numpy as np
from mingus.containers import Note
import mingus.core.scales as scales
import re


class NotesGenerator:
    """
    A class for generating needed notes lists as MIDI values.

    Attributes:
        key (str): The key or scale as note (letter) and scale name (i.e. 'C Major') to generate notes in.
    """
    def __init__(self, key='C Major'):
        self.key = key

    def parse_scale_name(self):
        """
        Parses the scale name from the `self.key` attribute. The method extracts the root note and the scale type from
        the key string.

        :return: A tuple (note, scale_type) where `note` is the root note and `scale_type`
                 is the type of the scale (e.g., Major, Minor).
        :raises ValueError: If the scale name does not match the expected format.
        """
        match = re.match(r"([A-G]#?|Bb)(.*)", self.key)
        if match:
            note = match.group(1)
            scale_type = match.group(2).strip()

            if note == 'D#': # mingus has problems with D# note
                note = 'Eb'

            return note, scale_type
        else:
            raise ValueError("Scale name is not valid.")

    def choose_scale(self):
        """
        Returns a scale based on the parsed scale name.

        :return: A numpy array representing the ascending notes of the chosen scale, excluding the last note,
        so that it does not include the root note twice.
        """
        note, scale_type = self.parse_scale_name()
        scale_methods = {
            "Harmonic Minor": scales.HarmonicMinor,
            "Natural Minor": scales.NaturalMinor,
            "Melodic Minor": scales.MelodicMinor,
            "Major": scales.Major,
            "Ionian": scales.Ionian,
            "Dorian": scales.Dorian,
            "Phrygian": scales.Phrygian,
            "Lydian": scales.Lydian,
            "Mixolydian": scales.Mixolydian,
            "Aeolian": scales.Aeolian,
            "Locrian": scales.Locrian,
            "Chromatic": scales.Chromatic,
            "Whole Tone": scales.WholeTone,
            "Octatonic": scales.Octatonic,
        }

        if scale_type in scale_methods:
            return np.array(scale_methods[scale_type](note).ascending()[:-1])
        else:
            for word in scale_type.split():
                for key in scale_methods:
                    if word in key:
                        return np.array(scale_methods[key](note).ascending()[:-1])

            print("Unknown scale type. Using Chromatic scale.")
            return np.array(scales.Chromatic(note).ascending()[:-1])

    def generate_notes(self, number_of_octaves=1, start_octave=4):
        """
        Generates a list of notes in scale, spanning a specified number of octaves.

        :param number_of_octaves: The number of octaves to span, default is 1.
        :param start_octave: The starting octave, default is 4.
        :return: A list of note values (integers) representing the notes generated.
        """
        scale = self.choose_scale()
        notes = []

        for octave in range(start_octave, start_octave + number_of_octaves):
            for note in scale:
                note_value = int(Note(note, octave))
                if notes and note_value <= notes[-1]:
                    note_value += 12
                notes.append(note_value)

        return notes

    def generate_chromatic_notes(self, note_range):
        """
        Generates a list of chromatic notes within a specified range.

        :param note_range: A tuple (start_note, end_note) specifying the range of notes.
        :return: A list of note values (integers) within the specified range.
        :raises ValueError: If the `note_range` arguments are not both integers or both strings.
        """
        start_note, end_note = note_range
        if isinstance(start_note, int) and isinstance(end_note, int):
            return list(range(start_note, end_note + 1))
        elif isinstance(start_note, str) and isinstance(end_note, str):
            return list(range(int(Note(start_note)), int(Note(end_note)) + 1))
        else:
            raise ValueError("Wrong argument for notes range")

