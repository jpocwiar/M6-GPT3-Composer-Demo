from midiutil import MIDIFile
import random
from mingus.containers import Note
import mingus.core.notes as NOTES
import mingus.core.chords as Chords
import numpy as np
from collections import defaultdict


class GeneticMelodyGenerator:
    """
        A class for generating melodies using a genetic algorithm.

        Attributes:
            NOTES (list): A list of possible notes that can be used in the melody.
            meter (tuple): A tuple representing the time signature of the piece (e.g., (4, 4)).
            note_duration (float): Fundamental duration of a note in the melody in relation to quarter note.
            mode (str): Playing mode of the melody ('melody' or 'solo').
            valence (float): A value representing the emotional valence for the generated melody, range [0, 1].
            arousal (float): A value representing the emotional arousal for the generated melody, range [0, 1].
            POPULATION_SIZE (int): The number of candidate melodies in each generation.
            NUM_GENERATIONS (int): The number of generations to evolve the population of melodies.
            MUTATION_RATE (float): The probability of mutation occurring in the genetic algorithm.
            CROSSOVER_RATE (float): The probability of crossover occurring between melodies in the genetic algorithm.
            expected_length (int): The expected length of average note, here: a quarter note.
            chords_dict (dict): A dictionary mapping time points to current chords.
            scale_notes (list): A list of notes belonging to scale melody will be generated in.
            notes_range (int): The range of notes available for melody generation.
            bass_line (np.array): An array of notes representing the bass line.
            motif_line (np.array): An array of notes representing the motif line.
    """
    def __init__(self, notes, scale_notes, chords_dict, mode, valence, arousal, motif=[], bass=[], meter=(4, 4), note_duration=0.5, population_size=64, num_generations=50, mutation_rate=0.3, crossover_rate=0.9):

        self.NOTES = notes
        self.meter = meter
        self.note_duration = note_duration

        self.mode = mode
        self.valence = valence
        self.arousal = arousal

        self.POPULATION_SIZE = population_size
        self.NUM_GENERATIONS = num_generations
        self.MUTATION_RATE = mutation_rate
        self.CROSSOVER_RATE = crossover_rate

        self.expected_length = int(1 / note_duration)

        self.chords_dict = chords_dict
        self.scale_notes = scale_notes
        self.notes_range = max(self.NOTES) - min(self.NOTES)

        self.bass_line = np.array(bass)
        self.motif_line = np.array(motif)

    def custom_tanh(self, x, scale=5):
        return np.tanh(x / scale)

    def score_valued_harmony_intervals(self, melody_np):
        """
        Computes the harmony interval scores for the melody compared to previously generated bass and motif lines.

        :param melody_np: An array of integers representing the melody, where -1 indicates a rest and -2 indicates an extension.
        :return: A tuple containing the scaled scores for bass and motif lines.
        """
        harmony_intervals_table = {
            0: 8, 1: -20, 2: -20, 3: 8, 4: 8, 5: 15,
            6: -30, 7: 15, 8: 8, 9: 8, 10: -20, 11: -20, 12: 8
        }

        def replace_minus_two(arr):
            """
            Replaces -2 values with the previous valid value in the array for interval calculation.

            :param arr: An array of melody notes.
            :return: The array with -2 values replaced.
            """
            previous_values = np.roll(arr, 1)
            previous_values[0] = arr[0]
            return np.where(arr == -2, previous_values, arr)

        def calculate_score(melody, line):
            """
            Calculates the harmony score between the melody and a reference line.

            :param melody: An array of melody notes.
            :param line: An array of reference line notes (bass or motif).
            :return: The normalized harmony score.
            """
            if len(melody) == 0 or len(line) == 0:
                return 0

            # Determine valid notes (not rests or extensions)
            melody_minus_one = melody == -1
            line_minus_one = line == -1
            neither_minus_one = ~(melody_minus_one | line_minus_one)

            # Score for cases where one is -1 (rest) and the other is not
            score_either_minus_one = np.where((melody_minus_one ^ line_minus_one), 15, 0)

            # Calculate intervals and their scores
            intervals = np.abs(melody[neither_minus_one] - line[neither_minus_one]) % 12
            score_intervals = np.array([harmony_intervals_table.get(interval, 0) for interval in intervals])

            # Compute total score and the number of elements considered
            total_score = np.sum(score_either_minus_one) + np.sum(score_intervals)
            total_elements = np.sum(neither_minus_one) + np.sum(melody_minus_one ^ line_minus_one)

            return total_score / total_elements if total_elements > 0 else 0

        # Replace -2 in melody with the previous valid value
        melody_np_replaced = replace_minus_two(melody_np)

        # Calculate scores for bass and motif lines
        bass_score = calculate_score(melody_np_replaced, self.bass_line) if len(self.bass_line) > 0 else 0
        motif_score = calculate_score(melody_np_replaced, self.motif_line) if len(self.motif_line) > 0 else 0

        return self.custom_tanh(bass_score), self.custom_tanh(motif_score)

    def fitness_melodic_contour(self, melody_np):
        """
        Calculates the melodic contour fitness score based on the proportion of ascending intervals.

        Pauses (-1) and extensions (-2) are ignored. If no intervals exist, a default 0.5 score is returned.

        :param melody_np: A NumPy array representing the melody.
        :return: A score indicating the proportion of ascending intervals (0 to 1).
        """

        notes = melody_np[(melody_np != -1) & (melody_np != -2)]

        if len(notes) < 2:
            return 0

        intervals = np.diff(notes)
        positive_intervals_count = np.sum(intervals > 0)
        total_intervals_count = np.sum(intervals != 0)

        if total_intervals_count == 0:
            return 0.5

        melodic_contour_score = positive_intervals_count / total_intervals_count
        return melodic_contour_score

    def fitness_note_range(self, melody_np):
        """
        Calculates the fitness score based on the pitch range of the melody.

        Pauses (-1) and extensions (-2) are ignored. If no notes are present, returns 0.

        :param melody_np: A NumPy array representing the melody.
        :return: A score indicating the pitch range relative to the total note range.
        """

        notes = melody_np[(melody_np != -1) & (melody_np != -2)]

        if len(notes) < 1:
            return 0

        pitch_range = np.max(notes) - np.min(notes)
        pitch_range_score = pitch_range / self.notes_range

        return pitch_range_score

    def fitness_average_pitch(self, melody_np):
        """
        Calculates the fitness score based on the average pitch of the melody.

        Ignores pauses (-1) and extensions (-2). If no valid notes are present, returns 0.

        :param melody_np: A NumPy array representing the melody.
        :return: A score indicating the average pitch relative to the total note range.
        """

        valid_notes = melody_np[(melody_np != -1) & (melody_np != -2)]

        if len(valid_notes) == 0:
            return 0

        average_pitch = np.mean(valid_notes)
        average_pitch_score = average_pitch / self.notes_range

        return average_pitch_score

    def fitness_pause_proportion(self, melody_np):
        """
        Calculates the proportion of the melody that consists of pauses (-1) including their extensions

        :param melody_np: A NumPy array representing the melody.
        :return: A score representing the proportion of pauses in the melody.
        """

        total_length = len(melody_np)

        if total_length == 0:
            return 0

        pause_length_counter = 0
        in_pause = False

        for note in melody_np:
            if note == -1 or (in_pause and note == -2):
                pause_length_counter += 1
                in_pause = True
            else:
                in_pause = False

        pause_proportion_score = pause_length_counter / total_length

        return pause_proportion_score

    def fitness_intervals(self, melody_np):
        """
        Calculates dissonance and large interval scores based on the melody's intervals.

        :param melody_np: A NumPy array representing the melody.
        :return: A tuple of (dissonance_score, large_intervals_score).
        """

        notes = melody_np[(melody_np != -1) & (melody_np != -2)]

        if len(notes) < 2:
            dissonance_score = 0
            large_intervals_score = 0
        else:
            intervals = np.abs(np.diff(notes))
            intervals = np.mod(intervals, 12)  # Normalize intervals to within an octave
            dissonance_values = np.where(
                np.isin(intervals, [10]), 0.5,  # Minor 7th is slightly dissonant
                np.where(np.isin(intervals, [0, 1, 2, 3, 4, 5, 7, 8, 9, 12]), 0, 1)  # Other intervals rated
            )
            dissonance_score = np.mean(dissonance_values)

            large_intervals_count = np.sum(intervals > 12)
            total_intervals_count = len(intervals)
            large_intervals_score = large_intervals_count / total_intervals_count

        return dissonance_score, large_intervals_score

    def fitness_scale_and_chord(self, melody_np):
        """
        Calculates the conformance of the melody to the scale and chords.

        :param melody_np: A NumPy array representing the melody.
        :return: A tuple of (scale_conformance_score, chord_conformance_score).
        """
        scale_notes_np = np.array(self.scale_notes)

        scale_length_counter = 0
        chord_length_counter = 0
        total_length_counter = 0

        for i, note in enumerate(melody_np):
            if note != -2:
                in_scale = note in scale_notes_np
                chord_notes = Chords.from_shorthand(self.find_chord_at_time(i * self.note_duration))
                in_chord = NOTES.int_to_note(note % 12) in chord_notes

                total_length_counter += 1
                if in_scale:
                    scale_length_counter += 1
                if in_chord:
                    chord_length_counter += 1

        if total_length_counter != 0:
            scale_conformance_score = scale_length_counter / total_length_counter
            chord_conformance_score = chord_length_counter / total_length_counter
        else:
            return 0

        return scale_conformance_score, chord_conformance_score

    def fitness_pitch_variation(self, melody_np):
        """
        Calculates the pitch variation score based on the standard deviation of notes.

        :param melody_np: A NumPy array representing the melody.
        :return: A float representing the pitch variation score.
        """
        valid_notes = melody_np[(melody_np != -1) & (melody_np != -2)]

        if len(valid_notes) < 2:
            return 0

        standard_deviation = np.std(valid_notes)
        max_possible_std = self.notes_range / np.sqrt(12)

        pitch_variation_score = standard_deviation / max_possible_std

        return pitch_variation_score

    def fitness_strong_beat(self, melody_np):
        """
        Calculates scores for notes placed on strong beats and their conformance with chords.

        :param melody_np: A NumPy array representing the melody.
        :return: A tuple containing the strong beat score and in-chord beat score.
        """
        strong_beat_interval = int(self.meter[0] / self.note_duration * 4 / self.meter[1])
        extension_interval = int(self.meter[0] / 2 / self.note_duration * 4 / self.meter[1])

        strong_beat_counter = 0
        in_chord_beat_counter = 0

        for i in range(0, len(melody_np), strong_beat_interval):
            if melody_np[i] > 0:
                strong_beat_counter += 1
                chord_notes = Chords.from_shorthand(self.find_chord_at_time(i * self.note_duration))
                in_chord = NOTES.int_to_note(melody_np[i] % 12) in chord_notes
                if in_chord:
                    in_chord_beat_counter += 1

                for j in range(1, extension_interval):
                    if i + j < len(melody_np) and melody_np[i + j] == -2:
                        strong_beat_counter += 1
                        if in_chord:
                            in_chord_beat_counter += 1

        denominator = len(melody_np) / strong_beat_interval * extension_interval
        if denominator == 0:
            return 0

        strong_beat_score = strong_beat_counter / denominator
        in_chord_beat_score = in_chord_beat_counter / denominator

        return strong_beat_score, in_chord_beat_score

    def fitness_odd_index_notes(self, melody_np):
        """
        Computes the ratio of notes at odd indices within each beat.

        :param melody_np: A NumPy array representing the melody.
        :return: The average ratio of notes and extensions at odd indices.
        """
        beat_length = int(self.meter[0] / self.note_duration * 4 / self.meter[1])
        num_beats = len(melody_np) // beat_length
        odd_index_scores = []

        for i in range(num_beats):
            beat = melody_np[i * beat_length:(i + 1) * beat_length]
            odd_note_indices = np.where((beat[1::2] > 0))[0] * 2 + 1
            note_and_extension_count = 0

            for idx in odd_note_indices:
                extension_idx = idx + 1
                while extension_idx < len(beat) and beat[extension_idx] == -2:
                    note_and_extension_count += 1
                    extension_idx += 1

            beat_length_adjusted = len(beat) - 2
            if beat_length_adjusted > 0:
                odd_index_note_and_extension_ratio = note_and_extension_count / beat_length_adjusted
                odd_index_scores.append(odd_index_note_and_extension_ratio)
            else:
                odd_index_scores.append(0)

        average_odd_index_ratio = np.mean(odd_index_scores) if odd_index_scores else 0

        return average_odd_index_ratio

    def fitness_note_diversity(self, melody_np):
        """
        Calculates the average diversity of notes within each beat of the melody.

        :param melody_np: A NumPy array representing the melody.
        :return: The average note diversity score.
        """
        beat_length = int(self.meter[0] / self.note_duration * 4 / self.meter[1])
        num_beats = len(melody_np) // beat_length

        if num_beats == 0:
            return 0

        diversity_scores = []

        for i in range(num_beats):
            beat = melody_np[i * beat_length:(i + 1) * beat_length]
            beat = beat[(beat != -2) & (beat != -1)]

            if len(beat) > 1:
                unique_notes = np.unique(beat)
                diversity_score = len(unique_notes) / len(beat)
                diversity_scores.append(diversity_score)
            else:
                diversity_scores.append(0)

        return np.mean(diversity_scores) if diversity_scores else 0

    def fitness_diversity_intervals(self, melody_np):
        """
        Computes the average diversity of intervals within each beat of the melody.

        :param melody_np: A NumPy array representing the melody.
        :return: The average interval diversity score.
        """
        beat_length = int(self.meter[0] / self.note_duration * 4 / self.meter[1])
        num_beats = len(melody_np) // beat_length

        if num_beats == 0:
            return 0

        diversity_scores = []

        for i in range(num_beats):
            beat = melody_np[i * beat_length:(i + 1) * beat_length]
            beat = beat[(beat != -2) & (beat != -1)]

            if len(beat) > 1:
                intervals = np.diff(beat)
                intervals = intervals[intervals <= 12]
                unique_intervals = len(np.unique(np.abs(intervals)))
                diversity_score = unique_intervals / len(beat)
                diversity_scores.append(diversity_score)
            else:
                diversity_scores.append(0)

        return np.mean(diversity_scores) if diversity_scores else 0

    def fitness_rhythm(self, melody_np):
        """
        Calculates the average rhythmic diversity within each beat of the melody.

        :param melody_np: A NumPy array representing the melody.
        :return: The average rhythmic diversity score.
        """
        beat_length = int(self.meter[0] / self.note_duration * 4 / self.meter[1])
        num_beats = len(melody_np) // beat_length

        if num_beats == 0:
            return 0

        rhythmic_diversity_scores = []

        for i in range(num_beats):
            beat = melody_np[i * beat_length:(i + 1) * beat_length]

            extension_lengths = []
            current_length = 0
            for note in beat:
                if note == -2:
                    current_length += 1
                elif current_length > 0:
                    extension_lengths.append(current_length)
                    current_length = 0
            if current_length > 0:
                extension_lengths.append(current_length)

            if len(extension_lengths) > 1:
                unique_lengths_count = len(set(extension_lengths))
                total_extensions_count = len(extension_lengths)
                rhythmic_diversity_score = unique_lengths_count / total_extensions_count
                rhythmic_diversity_scores.append(rhythmic_diversity_score)
            else:
                rhythmic_diversity_scores.append(0)

        return np.mean(rhythmic_diversity_scores) if rhythmic_diversity_scores else 0

    def fitness_log_rhythmic_value(self, melody_np):
        """
        Evaluates the average and standard deviation of log-transformed rhythmic values of extensions in the melody.
        We use log because of the exponential nature of rhythmic values.

        :param melody_np: A NumPy array representing the melody.
        :return: Tuple of normalized average and standard deviation of log-transformed rhythmic values.
        """
        extension_indices = np.where(melody_np == -2)[0]
        rhythmic_values = []

        if len(extension_indices) > 0:
            start_index = extension_indices[0]
            for i in range(1, len(extension_indices)):
                if extension_indices[i] != extension_indices[i - 1] + 1:
                    rhythmic_values.append(extension_indices[i - 1] - start_index + 2)
                    start_index = extension_indices[i]
            rhythmic_values.append(extension_indices[-1] - start_index + 2)

        single_notes = len(melody_np) - len(extension_indices) - len(rhythmic_values)
        rhythmic_values.extend([1] * single_notes)
        log_rhythmic_values = np.log2(rhythmic_values) if rhythmic_values else []

        if rhythmic_values:
            average_rhythmic_value = np.mean(rhythmic_values)
            log_average_rhythmic_value = np.log2(average_rhythmic_value)

            normalized_log_rhythmic_value = (log_average_rhythmic_value - np.log2(1)) / (
                        np.log2(4 * self.expected_length) - np.log2(1))
            std_log_rhythmic_value = np.std(log_rhythmic_values)
            normalized_std_log_rhythmic_value = std_log_rhythmic_value / np.std([0, np.log2(4 * self.expected_length)])
        else:
            normalized_log_rhythmic_value = 0
            normalized_std_log_rhythmic_value = 0

        return normalized_log_rhythmic_value, normalized_std_log_rhythmic_value

    def fitness_average_intervals(self, melody_np):
        """
        Computes the average interval between consecutive notes in the melody.

        :param melody_np: A NumPy array representing the melody.
        :return: Ratio of the average interval to 12 (octave), or -1 if not enough valid notes are present.
        """
        valid_notes = melody_np[(melody_np != -1) & (melody_np != -2)]

        if len(valid_notes) < 2:
            return -1

        intervals = np.diff(valid_notes)
        valid_intervals = intervals[intervals <= 12]

        if len(valid_intervals) < 1:
            return -1

        average_interval = np.mean(np.abs(valid_intervals))
        average_interval_ratio = average_interval / 12

        return average_interval_ratio

    def fitness_small_intervals(self, melody_np):
        """
        Computes the proportion of consecutive small intervals in the melody.

        :param melody_np: A NumPy array representing the melody.
        :return: Ratio of consecutive small intervals to total intervals.
        """
        valid_notes = melody_np[melody_np != -2]

        if len(valid_notes) < 2:
            return 0

        intervals = np.diff(valid_notes)
        small_intervals_mask = (intervals >= -3) & (intervals <= 3) & (intervals != 0)
        small_consecutive_intervals = np.sum(small_intervals_mask[:-1] & small_intervals_mask[1:])
        total_intervals = len(intervals)

        if total_intervals == 0:
            return 0

        small_interval_ratio = small_consecutive_intervals / total_intervals

        return small_interval_ratio

    def fitness_repeated_short_notes(self, melody_np):
        """
        Calculates the proportion of consecutive short notes in the melody.

        :param melody_np: A NumPy array representing the melody.
        :return: Ratio of consecutive short notes to total short notes.
        """
        shortest_notes_mask = (melody_np >= 0)
        shortest_notes_indices = np.nonzero(shortest_notes_mask)[0]

        consecutive_notes_mask = np.diff(shortest_notes_indices) <= 2
        total_consecutive_short_notes = np.sum(consecutive_notes_mask)

        total_short_notes = np.count_nonzero(shortest_notes_mask)

        if total_short_notes == 0:
            return 0

        short_notes_ratio = total_consecutive_short_notes / total_short_notes

        return short_notes_ratio

    def fitness_repeated_fragments(self, melody_np, fragment_length):
        """
        Calculates a score based on the presence of repeated fragments of a given length in the melody.

        :param melody_np: A NumPy array representing the melody.
        :param fragment_length: The length of the fragment to check for repetitions.
        :return: Score based on the amount of repeated fragments.
        """
        if fragment_length <= 1 or len(melody_np) <= fragment_length:
            return 0

        total_repeated_length = 0
        seen_fragments = set()

        # Iterate over all possible fragments of the given length
        for i in range(len(melody_np) - fragment_length + 1):
            fragment = tuple(melody_np[i:i + fragment_length])

            # Check if the fragment has been seen before
            if len(set(fragment)) > 1:  # Ignore fragments with only one unique note
                if fragment in seen_fragments:
                    total_repeated_length += fragment_length
                else:
                    seen_fragments.add(fragment)

        total_possible_repeats = len(melody_np) - fragment_length
        score = total_repeated_length / total_possible_repeats if total_possible_repeats > 0 else 0

        return score

    def get_musical_coefficients(self):
        """
        Specifies the mean and standard deviation values conditioned on mode and emotions for each musical feature,
        as well as their weights, used for fitness calculation.

        :return: A tuple of dictionaries containing the mean and standard deviation values, and a dictionary of weights.
        """

        valence = self.valence * 2 - 1
        arousal = self.arousal * 2 - 1

        mu_values = {
            'diversity': 0.8,
            'diversity_interval': 0.7 + arousal * 0.1 + valence * 0.1,
            'dissonance': 0.25 - valence * 0.25,
            'rhythmic_diversity': 0.7,
            'rhythmic_average_value': 0.5 - arousal * 0.25,
            'very_long_notes_score': 0,
            'deviation_rhythmic_value': 0.5 + arousal * 0.2 - valence * 0.2,
            'scale_conformance': 0.9 - arousal * 0.1,
            'chord_conformance': 0.5 - arousal * 0.1,
            'melodic_contour': 0.1 + self.valence * 0.8,
            'pitch_range': 0.3 + arousal * 0.25,
            'pause_proportion': 0.3 - arousal * 0.15 - valence * 0.15,
            'large_intervals': 0.0,
            'average_pitch': 0.6 + arousal * 0.3 + valence * 0.1,
            'pitch_variation': 0.4 + arousal * 0.2,
            'strong_beat': 0.5 - arousal * 0.2,
            'odd_index_notes': 0.1,
            'average_interval': 0.3 + valence * 0.3,
            'scale_playing': 0.8,
            'short_consecutive_notes': 0.75 + arousal * 0.25,
            'repetition_2': 0.5 + 0.2 * arousal,
            'repetition_3': 0.5 + 0.2 * arousal,
            'repetition_4': 0.5 + 0.2 * arousal,
            'repetition_5': 0.5 + 0.2 * arousal,
            'repetition_6': 0.5 + 0.2 * arousal
        }
        sigma_values = {
            'diversity': 0.1,
            'diversity_interval': 0.1,
            'dissonance': 0.1,
            'rhythmic_diversity': 0.1,
            'rhythmic_average_value': 0.1,
            'deviation_rhythmic_value': 0.1,
            'very_long_notes_score': 0.1,
            'scale_conformance': 0.1,
            'chord_conformance': 0.1,
            'melodic_contour': 0.1,
            'pitch_range': 0.1,
            'pause_proportion': 0.1,
            'large_intervals': 0.1,
            'average_pitch': 0.1,
            'pitch_variation': 0.1,
            'strong_beat': 0.1,
            'odd_index_notes': 0.1,
            'average_interval': 0.1,
            'scale_playing': 0.1,
            'short_consecutive_notes': 0.1,
            'repetition_2': 0.2,
            'repetition_3': 0.2,
            'repetition_4': 0.2,
            'repetition_5': 0.2,
            'repetition_6': 0.2,
        }
        weights = {
            'diversity': 2,
            'diversity_interval': 2,
            'dissonance': 3,
            'rhythmic_diversity': 1,
            'rhythmic_average_value': 3,
            'deviation_rhythmic_value': 2,
            'very_long_notes_score': 3,
            'scale_conformance': 3,
            'chord_conformance': 3,
            'melodic_contour': 1,
            'pitch_range': 1,
            'pause_proportion': 1,
            'large_intervals': 5,
            'average_pitch': 1,
            'pitch_variation': 1,
            'strong_beat': 3,
            'in_chord_beat': 2,
            'odd_index_notes': 3,
            'average_interval': 1,
            'repetition': 2,
            'scale_playing': 2,
            'short_consecutive_notes': 2,
            'repetition_2': 2,
            'repetition_3': 3,
            'repetition_4': 4,
            'repetition_5': 5,
            'repetition_6': 6,
        }

        if self.mode == 'melody':
            mu_values.update({
                'diversity': 0.4,
                'diversity_interval': 0.6 + arousal * 0.1 + valence * 0.1,
                'rhythmic_diversity': 0.5,
                'rhythmic_average_value': 0.5 - arousal * 0.1,
                'deviation_rhythmic_value': 0.3 + arousal * 0.2 - valence * 0.1,
                'scale_conformance': 1.0,
                'chord_conformance': 0.8 - arousal * 0.1,
                'pitch_range': 0.3 + arousal * 0.2,
                'average_pitch': 0.5 + arousal * 0.2 + valence * 0.1,
                'pitch_variation': 0.3 + arousal * 0.2,
                'strong_beat': 0.8 - arousal * 0.2,
                'odd_index_notes': 0.00,
                'scale_playing': 0.8,
                'short_consecutive_notes': 0.5 + arousal * 0.25
            })
            weights.update({
                'dissonance': 3,
                'scale_conformance': 3,
                'chord_conformance': 3,
                'melodic_contour': 1,
                'pitch_range': 1,
                'pause_proportion': 2,
                'average_pitch': 2,
                'pitch_variation': 2,
                'strong_beat': 3,
                'in_chord_beat': 3,
                'odd_index_notes': 3,
                'average_interval': 1,
                'scale_playing': 1,
                'short_consecutive_notes': 1,
                'repetition': 3,
            })

        return mu_values, sigma_values, weights

    def fitness(self, melody):
        """
        Evaluates the fitness of a given melody based on measured and expected values of musical features.

        :param melody: A list or array representing the melody to be evaluated.
        :return: Fitness value computed based on the weighted combination of various scores.
        """
        mu_values, sigma_values, weights = self.get_musical_coefficients()
        melody_np = np.array(melody)

        # Compute individual fitness scores
        dissonance_score, large_intervals_score = self.fitness_intervals(melody_np)
        scale_conformance_score, chord_conformance_score = self.fitness_scale_and_chord(melody_np)
        pause_proportion_score = self.fitness_pause_proportion(melody_np)
        diversity_score = self.fitness_note_diversity(melody_np)
        diversity_interval_score = self.fitness_diversity_intervals(melody_np)
        rhythmic_diversity_score = self.fitness_rhythm(melody_np)
        melodic_contour_score = self.fitness_melodic_contour(melody_np)
        pitch_range_score = self.fitness_note_range(melody_np)
        average_pitch_score = self.fitness_average_pitch(melody_np)
        pitch_variation_score = self.fitness_pitch_variation(melody_np)
        strong_beat_score, in_chord_beat_score = self.fitness_strong_beat(melody_np)
        odd_index_notes_score = self.fitness_odd_index_notes(melody_np)
        average_interval_score = self.fitness_average_intervals(melody_np)
        scale_playing_score = self.fitness_small_intervals(melody_np)
        short_consecutive_score = self.fitness_repeated_short_notes(melody_np)
        rhythmic_average_value, deviation_rhythmic_value = self.fitness_log_rhythmic_value(melody_np)
        repetition_2_score = self.fitness_repeated_fragments(melody_np, 2)
        repetition_3_score = self.fitness_repeated_fragments(melody_np, 3)
        repetition_4_score = self.fitness_repeated_fragments(melody_np, 4)
        repetition_5_score = self.fitness_repeated_fragments(melody_np, 5)
        repetition_6_score = self.fitness_repeated_fragments(melody_np, 6)

        scores = {
            'diversity': diversity_score,
            'diversity_interval': diversity_interval_score,
            'dissonance': dissonance_score,
            'rhythmic_diversity': rhythmic_diversity_score,
            'rhythmic_average_value': rhythmic_average_value,
            'deviation_rhythmic_value': deviation_rhythmic_value,
            'scale_conformance': scale_conformance_score,
            'chord_conformance': chord_conformance_score,
            'melodic_contour': melodic_contour_score,
            'pitch_range': pitch_range_score,
            'pause_proportion': pause_proportion_score,
            'large_intervals': large_intervals_score,
            'average_pitch': average_pitch_score,
            'pitch_variation': pitch_variation_score,
            'strong_beat': strong_beat_score,
            'odd_index_notes': odd_index_notes_score,
            'average_interval': average_interval_score,
            'scale_playing': scale_playing_score,
            'short_consecutive_notes': short_consecutive_score,
            'repetition_2': repetition_2_score,
            'repetition_3': repetition_3_score,
            'repetition_4': repetition_4_score,
            'repetition_5': repetition_5_score,
            'repetition_6': repetition_6_score,
        }

        # Compute the fitness value using a weighted sum of Gaussian-like scores
        fitness_value = sum(
            weights[feature] * np.exp(-0.5 * ((score - mu_values[feature]) / sigma_values[feature]) ** 2)
            for feature, score in scores.items()
        )

        # Add harmony interval scores if available
        try:
            multitrack_intervals_bass, multitrack_intervals_motif = self.score_valued_harmony_intervals(melody_np)
            fitness_value += (multitrack_intervals_bass * 2 + multitrack_intervals_motif * 4)
        except Exception:
            pass

        return fitness_value

    def mutate_from_motif(self, melody_np, motif_line):
        """
        Mutates a segment of the melody by replacing it with a portion of the motif line, adjusted by the difference
        between the starting note of the segment and the first note of the motif.

        :param melody_np: An array of integers representing the melody, where -1 indicates a rest and -2 indicates an extension.
        :param motif_line: An array of integers representing the motif.
        :return: The mutated melody array.
        """
        if len(melody_np) == 0 or len(motif_line) == 0:
            return melody_np

        start_index = np.random.randint(0, len(melody_np))
        max_num_elements = min(self.meter[0] * 4 // self.meter[1], len(melody_np))
        num_elements = np.random.randint(1, max_num_elements + 1)

        num_elements = min(num_elements, len(melody_np) - start_index, len(motif_line))

        if num_elements == 0:
            return melody_np

        difference = melody_np[start_index] - motif_line[0]
        sequence_to_insert = motif_line[:num_elements] + difference

        melody_np[start_index:start_index + num_elements] = sequence_to_insert

        return melody_np

    def mutate_interval(self, melody_np, valid_indices):
        """
        Randomly changes the interval between two notes. Range of new interval is [-12, 12].

        :param melody_np: Numpy array representing the melody.
        :param valid_indices: Indices of notes that can be mutated.
        :return: Mutated melody as a numpy array.
        """
        first_note_index, second_note_index = np.random.choice(valid_indices, 2, replace=False)
        interval = random.randint(-12, 12)
        melody_np[second_note_index] = melody_np[first_note_index] + interval
        return melody_np

    def mutate_transpose(self, melody_np):
        """
        Shifts a sequence segment by a random interval in ⟨−12, 12⟩ range. 

        :param melody_np: Numpy array representing the melody.
        :return: Mutated melody as a numpy array.
        """
        start_index = np.random.randint(0, len(melody_np))
        length = np.random.randint(1, min(self.meter[0] * 8 / self.meter[1], len(melody_np)))
        end_index = min(start_index + length, len(melody_np) - 1)
        transpose_value = random.randint(-12, 12)
        notes_mask = (melody_np[start_index:end_index] > 0)
        melody_np[start_index:end_index][notes_mask] += transpose_value
        melody_np[start_index:end_index][notes_mask] = np.clip(
            melody_np[start_index:end_index][notes_mask], self.NOTES[0], self.NOTES[-1]
        )
        return melody_np

    def mutate_extend(self, melody_np, valid_indices):
        """
        Extends a random note, shortening (or removing) the preceding.

        :param melody_np: Numpy array representing the melody.
        :param valid_indices: Indices of notes that can be mutated.
        :return: Mutated melody as a numpy array.
        """
        extend_index = np.random.choice(valid_indices)
        if extend_index == 0:
            melody_np = np.insert(melody_np, 1, -2)
            melody_np = np.delete(melody_np, -1)
        else:
            melody_np[extend_index - 1], melody_np[extend_index] = melody_np[extend_index], -2
        return melody_np

    def mutate_pause(self, melody_np, indices):
        """
        Converts random note to a pause or a pause to a note.

        :param melody_np: Numpy array representing the melody.
        :param indices: Indices of notes in the melody.
        :return: Mutated melody as a numpy array.
        """
        replace_index = np.random.choice(indices)
        if melody_np[replace_index] == -1:
            melody_np[replace_index] = np.random.choice(self.NOTES)
        else:
            melody_np[replace_index] = -1
        return melody_np

    def mutate_long_notes(self, melody_np, indices):
        """
        Changes a random series of notes into extensions (-2)

        :param melody_np: Numpy array representing the melody.
        :param indices: Indices of notes in the melody.
        :return: Mutated melody as a numpy array.
        """
        start_index = np.random.choice(indices)
        num_notes_to_extend = np.random.randint(1, min(self.meter[0] * 8 / self.meter[1], len(melody_np)))
        end_index = min(start_index + num_notes_to_extend, len(melody_np) - 1)
        melody_np[start_index:end_index] = -2
        return melody_np

    def mutate_replace_extension(self, melody_np):
        """
        Converts random extension (-2) into a random note.

        :param melody_np: Numpy array representing the melody.
        :return: Mutated melody as a numpy array.
        """
        extension_indices = np.where(melody_np == -2)[0]
        if len(extension_indices) > 0:
            chosen_index = np.random.choice(extension_indices)
            new_note = np.random.choice(self.NOTES)
            melody_np[chosen_index] = self.find_nearest(new_note)
        return melody_np

    def mutate_length_norm(self, melody_np, valid_indices):
        """
        Divides long notes in two or extends short notes. We treat a quarter note as the ”normal” length.
        Notes longer than this are randomly split, while shorter ones are extended.

        :param melody_np: Numpy array representing the melody.
        :param valid_indices: Indices of valid notes that can be used to adjust the extension.
        :return: Mutated melody as a numpy array.
        """
        chosen_index = np.random.choice(valid_indices)
        chosen_note = melody_np[chosen_index]
        extension_count = 0
        next_index = chosen_index + 1
        while next_index < len(melody_np) and melody_np[next_index] == -2:
            extension_count += 1
            next_index += 1

        if extension_count + 1 > self.expected_length:
            replace_index = np.random.randint(chosen_index + 1, chosen_index + 1 + extension_count)
            melody_np[replace_index] = chosen_note
        elif extension_count + 1 < self.expected_length:
            additional_extensions = np.random.randint(1, self.expected_length - extension_count + 1)
            end_index = min(chosen_index + 1 + additional_extensions, len(melody_np))
            melody_np[chosen_index + 1:end_index] = -2
        return melody_np

    def mutate_sort(self, melody_np):
        """
        Sorts a random slice of a sequence in a random direction.

        :param melody_np: Numpy array representing the melody.
        :return: Mutated melody as a numpy array.
        """
        start_index = np.random.randint(0, len(melody_np) - 1)
        length = np.random.randint(1, min(self.meter[0] * 8 / self.meter[1], len(melody_np)))
        end_index = min(start_index + length, len(melody_np) - 1)
        fragment = melody_np[start_index:end_index]
        valid_notes_fragment = fragment[(fragment != -2) & (fragment != -1)]

        if len(valid_notes_fragment) > 1:
            ascending = random.choice([True, False])
            sorted_fragment = np.sort(valid_notes_fragment) if ascending else np.sort(valid_notes_fragment)[::-1]
            valid_notes_indices = np.arange(start_index, end_index)[(fragment != -2) & (fragment != -1)]
            melody_np[valid_notes_indices] = sorted_fragment

        return melody_np

    def mutate_repeat1(self, melody_np):
        """
        Randomly chooses a sequence and pastes it to another measure with optional transposition.

        :param melody_np: Numpy array representing the melody.
        :return: Mutated melody as a numpy array.
        """
        beat_length = int(self.meter[0] / self.note_duration * 4 / self.meter[1])
        num_beats = len(melody_np) // beat_length

        if num_beats > 1:
            beat1, beat2 = np.random.choice(num_beats, 2, replace=False)
            start1, start2 = beat1 * beat_length, beat2 * beat_length
            num_notes_to_copy = np.random.randint(0, beat_length)

            end1 = min(start1 + num_notes_to_copy, len(melody_np))
            end2 = min(start2 + num_notes_to_copy, len(melody_np))

            melody_np[start2:end2] = melody_np[start1:end1]

            if random.random() < self.MUTATION_RATE:
                transpose_values = [0, 12, 5, 7, -12, -5, -7]
                transpose_value = random.choice(transpose_values)
                notes_mask = (melody_np[start2:end2] > 0)
                melody_np[start2:end2][notes_mask] += transpose_value
                melody_np[start2:end2][notes_mask] = np.clip(
                    melody_np[start2:end2][notes_mask], self.NOTES[0], self.NOTES[-1]
                )

        return melody_np

    def mutate_repeat2(self, melody_np):
        """
        Randomly chooses a sequence and pastes it immediately after.

        :param melody_np: Numpy array representing the melody.
        :return: Mutated melody as a numpy array.
        """
        beat_length = int(self.meter[0] / self.note_duration * 4 / self.meter[1])
        num_beats = len(melody_np) // beat_length

        if num_beats > 1:
            selected_beat = np.random.choice(num_beats, 1)[0]
            start = selected_beat * beat_length
            num_notes_to_copy = np.random.randint(1, beat_length // 2 + 1)
            end = min(start + num_notes_to_copy, len(melody_np))
            replace_start = start + num_notes_to_copy
            replace_end = min(replace_start + num_notes_to_copy, len(melody_np))

            melody_np[replace_start:replace_end] = melody_np[start:end]

        return melody_np

    def mutate(self, melody):
        """
        Main mutation function. Applies mutations with musical significance to the melody
        based on a mutation rate.

        :param melody: A list representing the melody.
        :return: A mutated melody as a list.
        """
        melody_np = np.array(melody)
        indices = np.arange(len(melody))
        valid_indices = np.where((melody_np != -1) & (melody_np != -2))[0]

        if random.random() < self.MUTATION_RATE and len(self.motif_line) > 0:
            melody_np = self.mutate_from_motif(melody_np, self.motif_line)

        if random.random() < self.MUTATION_RATE and len(valid_indices) > 1:
            melody_np = self.mutate_interval(melody_np, valid_indices)

        if random.random() < self.MUTATION_RATE:
            melody_np = self.mutate_transpose(melody_np)

        valid_indices = np.where(melody_np != -2)[0]
        if random.random() < self.MUTATION_RATE and len(valid_indices) > 0:
            melody_np = self.mutate_extend(melody_np, valid_indices)

        if random.random() < self.MUTATION_RATE:
            melody_np = self.mutate_pause(melody_np, indices)

        if random.random() < self.MUTATION_RATE and len(indices) > 1:
            melody_np = self.mutate_long_notes(melody_np, indices)

        if random.random() < self.MUTATION_RATE:
            melody_np = self.mutate_replace_extension(melody_np)

        valid_indices = np.where(melody_np != -2)[0]
        if random.random() < self.MUTATION_RATE and len(valid_indices) > 0:
            melody_np = self.mutate_length_norm(melody_np, valid_indices)

        if random.random() < self.MUTATION_RATE:
            melody_np = self.mutate_sort(melody_np)

        if random.random() < self.MUTATION_RATE:
            melody_np = self.mutate_repeat1(melody_np)

        if random.random() < self.MUTATION_RATE:
            melody_np = self.mutate_repeat2(melody_np)

        return melody_np.tolist()

    def crossover(self, parent1, parent2):
        """
        Performs single-point crossover between two parent sequences to produce two offspring.

        :param parent1: The first parent sequence as a list of integers.
        :param parent2: The second parent sequence as a list of integers.
        :return: A tuple containing two offspring sequences resulting from the crossover.
        """
        if len(parent1) <= 1 or len(parent2) <= 1:
            raise ValueError("Parent sequences must have at least two elements.")

        index = random.randint(1, len(parent1) - 1)
        child1 = parent1[:index] + parent2[index:]
        child2 = parent2[:index] + parent1[index:]
        return child1, child2

    def tournament_selection(self, population, tournament_size=4):
        """
        Selects the best individual from a randomly chosen subset of the population.

        :param population: A list of individuals, where each individual is represented as a list of integers.
        :param tournament_size: The number of individuals to randomly select for the tournament.
        :return: The best individual from the randomly selected subset based on fitness.
        """
        # Ensure the tournament size is valid
        if tournament_size > len(population):
            raise ValueError("Tournament size cannot be larger than the population size.")

        tournament = random.sample(population, tournament_size)

        best = max(tournament, key=self.fitness)

        return best

    def get_range(self, start_note=45, end_note=105):
        """
        Returns a tuple representing the range of musical notes as MIDI values.

        :param start_note: The starting note of the range, either as an integer (MIDI number) or a string (note name).
        :param end_note: The ending note of the range, either as an integer (MIDI number) or a string (note name).
        :return: A tuple (start_note, end_note) where both are integers representing the MIDI numbers of the notes.
        """
        if isinstance(start_note, int) and isinstance(end_note, int):
            return start_note, end_note
        elif isinstance(start_note, str) and isinstance(end_note, str):
            return int(Note(start_note)), int(Note(end_note))
        else:
            raise ValueError("Wrong argument for notes range")

    def find_nearest(self, value):
        """
        Finds the nearest note in a scale to the given value.

        :param value: The value to find the nearest for. It should be an integer or float.
        :return: The nearest value from `self.scale_notes`, which matches the type of the input value.
        """
        array = np.asarray(self.scale_notes)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def generate_population(self, note_amount):
        """
        Generates a population of individuals, where each individual is a sequence of musical notes.

        Each individual starts with a random note and then generates subsequent notes by applying a random
        interval change. If the new note is out of range or not in the scale, it adjusts the note and ensures
        it conforms to the scale.

        :param note_amount: The number of notes in each individual sequence.
        :return: A list of individuals, where each individual is a list of notes.
        """
        population = []
        for _ in range(self.POPULATION_SIZE):
            individual = [random.choice(self.NOTES)]
            for _ in range(note_amount - 1):
                change = random.randint(-12, 12)
                next_note = individual[-1] + change
                if next_note < self.NOTES[0] or next_note > self.NOTES[len(self.NOTES)-1]:
                    next_note = individual[-1] - 2 * change
                
                if next_note not in self.scale_notes:
                    next_note = self.find_nearest(next_note)
                individual.append(next_note)
            population.append(individual)
        return population

    def run(self, measures=1):
        """
       Runs the genetic algorithm to evolve a population of melodies over a specified number of generations.

       This method initializes a population of melodies, then iteratively selects parents, performs crossover and mutation
       to produce a new generation of melodies. The process repeats for a specified number of generations. Finally, it returns
       the best melody found.

       :param measures: Number of measures for which to generate melodies.
       :return: The best melody found after all generations.
       """
        note_amount = int(self.meter[0] / self.note_duration * 4 / self.meter[1] * measures)
        population = self.generate_population(note_amount)
        self.adjust_lines(note_amount)
        for generation in range(self.NUM_GENERATIONS):
            print(f'Generation {generation + 1}/{self.NUM_GENERATIONS}')
            new_population = []
            while len(new_population) < self.POPULATION_SIZE:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                if random.random() < self.CROSSOVER_RATE:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2

                new_population.extend([self.mutate(child1), self.mutate(child2)])
            population = new_population

        best_melody = max(population, key=self.fitness)

        return best_melody

    def adjust_arrays(self, bass, motif, note_amount):
        """
        Adjusts the lengths of the bass and motif arrays to match the specified note amount. It is used for calculating
        intervals between melody and the bass and motif lines for harmonic coherence score, when the tracks have different
        fundamental note lengths.

        :param bass: Array-like object representing the bass line.
        :param motif: Array-like object representing the motif line.
        :param note_amount: The target length for the arrays. Should be an integer.

        :return: A tuple of numpy arrays, where:
            - The first element is the adjusted bass array.
            - The second element is the adjusted motif array.
        """

        def adjust_array(arr, note_amount):
            """
            Adjusts the length of a single array to the target note amount.

            :param arr: Numpy array to adjust.
            :param note_amount: The target length for the array.

            :return: Numpy array adjusted to the target length.
            """
            length_ratio = len(arr) / note_amount

            if length_ratio < 1:
                times_shorter = int(round(1 / length_ratio))
                extended_arr = np.empty(len(arr) * times_shorter, dtype=arr.dtype)
                extended_arr[::times_shorter] = arr
                extended_arr.fill(-2)
                extended_arr[::times_shorter] = arr
                return extended_arr
            elif length_ratio > 1:
                for _ in range(int(np.log2(length_ratio))):
                    arr = arr[::2]
                return arr
            else:
                return arr

        def replace_minus_two(arr):
            """
            Replaces -2 values in the array with the preceding non -2 values.

            :param arr: Numpy array where -2 values are to be replaced.
            :return: Numpy array with -2 values replaced by the previous non -2 values.
            """
            previous_values = np.roll(arr, 1)
            previous_values[0] = arr[0]
            return np.where(arr == -2, previous_values, arr)

        bass_line = np.array(bass)
        motif_line = np.array(motif)
        bass_line = replace_minus_two(bass_line) if len(bass_line) > 0 else bass_line
        motif_line = replace_minus_two(motif_line) if len(motif_line) > 0 else motif_line

        adjusted_bass = adjust_array(bass_line, note_amount) if len(bass_line) > 0 else bass_line
        adjusted_motif = adjust_array(motif_line, note_amount) if len(motif_line) > 0 else motif_line

        return adjusted_bass, adjusted_motif

    def adjust_lines(self, note_amount):
        """
        Adjusts the lengths of the bass and motif lines to match the specified note amount.

        :param note_amount: The target length for the arrays. Should be an integer.
        """
        self.bass_line, self.motif_line = self.adjust_arrays(self.bass_line, self.motif_line, note_amount)

    def find_chord_at_time(self, time):
        """
        Finds the chord being played at a specific time.

        :param time: The time at which to find the chord. Should be a numerical value representing time.
        :return: The chord being played at the specified time, or None if no chord is found for that time.
        """
        current_chord = None
        for start_time in sorted(self.chords_dict.keys()):
            if time >= start_time:
                current_chord = self.chords_dict[start_time]
            else:
                break
        return current_chord


class GeneticBassGenerator(GeneticMelodyGenerator):
    """
   A class for generating bass lines using a genetic algorithm and probabilistic methods, inheriting from GeneticMelodyGenerator.

   Inherits from:
       GeneticMelodyGenerator: Provides the base functionality for melody generation and genetic algorithm operations.

   Attributes:
       expected_length (int): The expected length of the bass line, calculated based on the emotional arousal value.
                              It is set to a value derived from the arousal level, influencing the expected duration of notes
   """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_length = int(3 * (1-self.arousal) + 1)

    def generate_population(self, note_amount):
        """
        Generates a population of melodies with a specified number of notes.

        Each melody starts with a tonic note derived from the chord at time 0. Subsequent notes are generated by
        adding random changes to the previous note, ensuring they stay within the defined scale. Notes outside the
        scale are adjusted to the nearest scale note.

        :param note_amount: The number of notes each melody should contain.
        :return: A list of melodies, where each melody is represented as a list of notes.
        """
        population = []
        first_chord_shorthand = self.find_chord_at_time(0)
        first_note = Chords.from_shorthand(first_chord_shorthand)[0]
        tonic = int(Note(first_note, 2))
        for _ in range(self.POPULATION_SIZE):
            individual = [tonic]
            for _ in range(note_amount - 1):
                change = random.randint(-12, 12)
                next_note = individual[-1] + change
                if next_note < self.NOTES[0] or next_note > self.NOTES[len(self.NOTES)-1]:
                    next_note = individual[-1] - 2 * change
                
                if next_note not in self.scale_notes:
                    next_note = self.find_nearest(next_note)
                individual.append(next_note)
            population.append(individual)
        return population

    def run(self, measures=1):
        """
        Executes the genetic algorithm or probablilistic methods to generate a bass track based on the specified
        mode and number of measures.

        Depending on the mode (`'short_riff'`, `'long_riff'`, or `'groove'`,`'repeated_groove'`), the method generates
        a population of melodies, applies selection, crossover, and mutation operations, and evolves the melodies
        over multiple generations or uses probabilistic methods to build bass lines on root notes with random mutations.
        The best melody is selected and extended to the desired number of measures.
        The method handles different types of melodies and variations according to the mode and arousal parameters.

        :param measures: Number of measures to generate the bass melody for. Defaults to 1.
        :return: A list of notes representing the final generated bass melody.
        """

        if self.mode in ['short_riff', 'riff']:
            note_amount = int(self.meter[0] / self.note_duration * 4 / self.meter[1])
            population = [random.choices(self.NOTES, k=note_amount) for _ in range(self.POPULATION_SIZE)]

            for generation in range(self.NUM_GENERATIONS):
                print(f'Generation {generation + 1}/{self.NUM_GENERATIONS}')
                new_population = []
                while len(new_population) < self.POPULATION_SIZE:
                    parent1 = self.tournament_selection(population)
                    parent2 = self.tournament_selection(population)

                    if random.random() < self.CROSSOVER_RATE:
                        child1, child2 = self.crossover(parent1, parent2)
                    else:
                        child1, child2 = parent1, parent2

                    new_population.extend([self.mutate(child1), self.mutate(child2)])
                population = new_population

            best_melody = max(population, key=self.fitness)

            first_chord_shorthand = self.find_chord_at_time(0)
            first_note = Chords.from_shorthand(first_chord_shorthand)[0]
            tonic = int(Note(first_note, 4))

            if self.arousal > 0.5:
                best_melody = self.replace_extensions_efficient(best_melody, (self.arousal - 0.5) * 2)
            
            melody_np = np.array(best_melody)
            concatenated_melody = best_melody
            for i in range(1, measures):
                root_note_shorthand = self.find_chord_at_time(i * self.meter[0] * 4 / self.meter[1])
                root = int(Note(Chords.from_shorthand(root_note_shorthand)[0], 4))

                interval = root - tonic

                mask = melody_np >= 0
                transformed_melody = melody_np.copy()
                transformed_melody[mask] += interval
                transformed_melody[mask] = np.vectorize(self.find_nearest)(transformed_melody[mask])

                concatenated_melody.extend(transformed_melody.tolist())

        elif self.mode == 'long_riff' and measures > 1:
            note_amount = int(self.meter[0] / self.note_duration * 4 / self.meter[1]) * 2
            population = [random.choices(self.NOTES, k=note_amount) for _ in range(self.POPULATION_SIZE)]

            for generation in range(self.NUM_GENERATIONS):
                print(f'Generation {generation + 1}/{self.NUM_GENERATIONS}')
                new_population = []
                while len(new_population) < self.POPULATION_SIZE:
                    parent1 = self.tournament_selection(population)
                    parent2 = self.tournament_selection(population)

                    if random.random() < self.CROSSOVER_RATE:
                        child1, child2 = self.crossover(parent1, parent2)
                    else:
                        child1, child2 = parent1, parent2

                    new_population.extend([self.mutate(child1), self.mutate(child2)])
                population = new_population

            best_melody = max(population, key=self.fitness)

            first_chord_shorthand = self.find_chord_at_time(0)
            first_note = Chords.from_shorthand(first_chord_shorthand)[0]
            tonic = int(Note(first_note, 4))

            if self.arousal > 0.5:
                best_melody = self.replace_extensions_efficient(best_melody, (self.arousal - 0.5) * 2)
            
            melody_np = np.array(best_melody)
            concatenated_melody = best_melody
            for i in range(2, measures, 2):
                left_root_note_shorthand = self.find_chord_at_time(i * self.meter[0] * 4 / self.meter[1])
                left_root = int(Note(Chords.from_shorthand(left_root_note_shorthand)[0], 4))
                right_root_note_shorthand = self.find_chord_at_time((i+1) * self.meter[0] * 4 / self.meter[1])
                right_root = int(Note(Chords.from_shorthand(right_root_note_shorthand)[0], 4))

                left_interval = left_root - tonic
                right_interval = right_root - tonic
                mask = melody_np >= 0
                
                half_length = len(melody_np) // 2
                first_half = melody_np[:half_length]
                second_half = melody_np[half_length:]

                first_half_transformed = first_half.copy()
                first_half_transformed[mask[:half_length]] += left_interval
                first_half_transformed[mask[:half_length]] = np.vectorize(self.find_nearest)(
                    first_half_transformed[mask[:half_length]])

                if len(second_half) > 0:
                    second_half_transformed = second_half.copy()
                    second_half_transformed[mask[half_length:]] += right_interval
                if len(second_half_transformed[mask[half_length:]]) > 0:
                    second_half_transformed[mask[half_length:]] = np.vectorize(self.find_nearest)(
                        second_half_transformed[mask[half_length:]])

                transformed_melody = np.concatenate([first_half_transformed, second_half_transformed])

                concatenated_melody.extend(transformed_melody.tolist())
        else:
            note_amount = int(self.meter[0] / self.note_duration * 4 / self.meter[1])
            concatenated_melody = []

            for i in range(measures):
                
                root_note_shorthand = self.find_chord_at_time(i * self.meter[0] * 4 / self.meter[1])
                root = int(Note(Chords.from_shorthand(root_note_shorthand)[0], 2))

                root_list = [root] * note_amount

                concatenated_melody.extend(root_list)

            melody_np = np.array(concatenated_melody)
            scale_np = np.array(self.NOTES)

            def find_scale_notes_between(start_note, end_note, scale):
                if abs(end_note - start_note) == 2:
                    
                    return np.array([(start_note + end_note) // 2])  
                elif start_note < end_note:
                    return scale[(scale > start_note) & (scale < end_note)]
                else:
                    return scale[(scale < start_note) & (scale > end_note)][::-1]

            for i in range(0, len(melody_np), note_amount):
                current_note = melody_np[i + note_amount - 1]

                if i + note_amount < len(melody_np):
                    next_note = melody_np[i + note_amount]
                else:
                    num_steps = random.randint(0, 4)
                    direction = random.choice([-1, 1])
                    next_note = current_note + num_steps * direction

                scale_notes_between = find_scale_notes_between(current_note, next_note, scale_np)

                num_notes_to_change = len(scale_notes_between)

                for j in range(min(num_notes_to_change, note_amount - 1)):
                    melody_np[i + note_amount - num_notes_to_change + j] = scale_notes_between[j]

            if self.mode == 'groove':
                
                mutation_probability = 0.4 * self.arousal
                mutations = np.array([-2, 1, 2, 5, 7, 12])
                probabilities = np.array([0.1, 0.05, 0.1, 0.1, 0.25, 0.4])

                mask = np.ones_like(melody_np, dtype=bool)
                mask[::note_amount] = False
                mask[note_amount - 1::note_amount] = False

                mutation_indices = np.random.rand(len(melody_np)) < mutation_probability
                mutation_indices = np.logical_and(mask, mutation_indices)

                melody_np[mutation_indices] += np.random.choice(mutations, size=np.sum(mutation_indices), p=probabilities)

                mutation_probability = 0.8 * (1-self.arousal)
                mutations = np.array([-1, -2])
                probabilities = np.array([0.01, 0.99])

                mutation_indices = np.random.rand(len(melody_np)) < mutation_probability
                mutation_indices = np.logical_and(mask, mutation_indices)

                melody_np[mutation_indices] = np.random.choice(mutations, size=np.sum(mutation_indices), p=probabilities)

                concatenated_melody = melody_np.tolist()
            elif self.mode == 'repeated_groove':
                
                shift_mutation_probability = 0.4 * self.arousal
                shift_mutations = np.array([-2, 1, 2, 5, 7, 12])
                shift_probabilities = np.array([0.1, 0.05, 0.1, 0.1, 0.25, 0.4])

                replace_mutation_probability = 0.8 * (1 - self.arousal)
                replace_mutations = np.array([-1, -2])
                replace_probabilities = np.array([0.05, 0.95])

                shift_indices = np.random.rand(note_amount) < shift_mutation_probability
                replace_indices = np.random.rand(note_amount) < replace_mutation_probability
                
                shift_indices[0] = False
                shift_indices[-1] = False
                replace_indices[0] = False
                replace_indices[-1] = False

                shifts_for_first_fragment = np.random.choice(shift_mutations, size=np.sum(shift_indices),
                                                             p=shift_probabilities)
                replacements_for_first_fragment = np.random.choice(replace_mutations, size=np.sum(replace_indices),
                                                             p=replace_probabilities)

                for i in range(0, len(melody_np), note_amount):
                    shift_fragment_indices = i + np.where(shift_indices)[0]
                    replace_fragment_indices = i + np.where(replace_indices)[0]

                    melody_np[shift_fragment_indices] += shifts_for_first_fragment

                    melody_np[replace_fragment_indices] = replacements_for_first_fragment

                concatenated_melody = melody_np.tolist()

        return concatenated_melody

    def replace_extensions_efficient(self, melody, probability):
        """
        Replaces extension notes (represented by -2) in the melody with preceding notes (the notes that are extended).

        Extension notes are replaced based on a specified probability. The method ensures that the replacements
        are consistent with the preceding notes and may add an octave shift to some replacements.

        :param melody: List of notes where -2 represents extension notes.
        :param probability: Probability of replacing extension notes.
        :return: Updated melody with extension notes replaced.
        """
        melody_np = np.array(melody)
        extension_indices = np.where(melody_np == -2)[0]
        non_extension_indices = np.where(melody_np != -2)[0]
        replace_indices = extension_indices[np.random.rand(len(extension_indices)) < probability]

        for idx in replace_indices:

            preceding_indices = non_extension_indices[non_extension_indices < idx]
            if len(preceding_indices) > 0:
                preceding_note = melody_np[preceding_indices[-1]]
            else:
                preceding_note = -1

            melody_np[idx] = preceding_note

            if preceding_note != -1 and np.random.rand() < 0.4 and melody_np[idx] < self.NOTES[0] + 19:
                melody_np[idx] += 12

        return melody_np.tolist()

    def get_musical_coefficients(self):
        """
        Specifies the mean and standard deviation values conditioned on mode and emotions for each musical feature,
        as well as their weights, used for fitness calculation.

        :return: A tuple of dictionaries containing the mean and standard deviation values, and a dictionary of weights.
        """

        valence = self.valence * 2 - 1
        arousal = self.arousal * 2 - 1

        mu_values = {
            'diversity': 0.6,
            'diversity_interval': 0.6 + arousal * 0.1 + valence * 0.1,
            'dissonance': 0.25 - valence * 0.25,
            'scale_conformance': 0.7 - arousal * 0.1,
            'chord_conformance': 0.8 - arousal * 0.1,
            'root_conformance': 0.4 - arousal * 0.1,
            'rhythmic_average_value': 0.5 - arousal * 0.25,
            'deviation_rhythmic_value': 0.3 + arousal * 0.2 - valence * 0.1,
            'melodic_contour': 0.1 + self.valence * 0.8,
            'pitch_range': 0.4 + arousal * 0.3,
            'large_intervals': 0.0,
            'average_pitch': 0.15 + arousal * 0.1 + valence * 0.05,
            'pitch_variation': 0.5 + arousal * 0.2,
            'strong_beat': 1,
            'chord_beat': 0.5,
            'root_beat': 0.5,
            'odd_index_notes': 0.1,
            'average_interval': 0.5 + valence * 0.3,
            'scale_playing': 0.5,
            'short_consecutive_notes': 0.5 + arousal * 0.25,
            'repetition_2': 0.5 + arousal * 0.2,
            'repetition_3': 0.5 + arousal * 0.2,
            'repetition_4': 0.5 + arousal * 0.2,
            'repetition_5': 0.5 + arousal * 0.2,
            'repetition_6': 0.5 + arousal * 0.2,
        }
        sigma_values = {
            'diversity': 0.1,
            'diversity_interval': 0.1,
            'dissonance': 0.1,
            'scale_conformance': 0.1,
            'chord_conformance': 0.1,
            'root_conformance': 0.1,
            'rhythmic_average_value': 0.1,
            'deviation_rhythmic_value': 0.1,
            'melodic_contour': 0.1,
            'pitch_range': 0.1,
            'large_intervals': 0.1,
            'average_pitch': 0.1,
            'pitch_variation': 0.1,
            'strong_beat': 0.1,
            'chord_beat': 0.1,
            'root_beat': 0.1,
            'odd_index_notes': 0.1,
            'average_interval': 0.1,
            'scale_playing': 0.1,
            'short_consecutive_notes': 0.1,
            'repetition_2': 0.2,
            'repetition_3': 0.2,
            'repetition_4': 0.2,
            'repetition_5': 0.2,
            'repetition_6': 0.2,
        }
        weights = {
            'diversity': 3,
            'diversity_interval': 5,
            'dissonance': 3,
            'scale_conformance': 1,
            'chord_conformance': 4,
            'root_conformance': 4,
            'rhythmic_average_value': 1,
            'deviation_rhythmic_value': 1,
            'melodic_contour': 1,
            'pitch_range': 1,
            'large_intervals': 10,
            'average_pitch': 5,
            'pitch_variation': 5,
            'strong_beat': 3,
            'chord_beat': 5,
            'root_beat': 5,
            'in_chord_beat': 2,
            'odd_index_notes': 5,
            'average_interval': 1,
            'repetition': 2,
            'scale_playing': 2,
            'short_consecutive_notes': 2,
            'repetition_2': 2,
            'repetition_3': 3,  
            'repetition_4': 4,
            'repetition_5': 5,
            'repetition_6': 6,
        }

        return mu_values, sigma_values, weights

    def fitness(self, melody):
        """
        Evaluates the fitness of a given melody based on measured and expected values of musical features.

        :param melody: A list or array representing the melody to be evaluated.
        :return: Fitness value computed based on the weighted combination of various scores.
        """

        mu_values, sigma_values, weights = self.get_musical_coefficients()
        melody_np = np.array(melody)

        dissonance_score, large_intervals_score = self.fitness_intervals(melody_np)
        scale_conformance_score, chord_conformance_score, root_conformance_score = self.fitness_scale_chord_and_root(melody_np)
        diversity_score = self.fitness_note_diversity(melody_np)
        diversity_interval_score = self.fitness_diversity_intervals(melody_np)
        melodic_contour_score = self.fitness_melodic_contour(melody_np)
        pitch_range_score = self.fitness_note_range(melody_np)
        average_pitch_score = self.fitness_average_pitch(melody_np)
        pitch_variation_score = self.fitness_pitch_variation(melody_np)
        strong_beat_score, in_chord_beat_score, root_beat_score = self.fitness_strong_beat(melody_np)
        odd_index_notes_score = self.fitness_odd_index_notes(melody_np)
        average_interval_score = self.fitness_average_intervals(melody_np)
        rhythmic_average_value, deviation_rhythmic_value = self.fitness_log_rhythmic_value(melody_np)
        repetition_2_score = self.fitness_repeated_fragments(melody_np, 2)
        repetition_3_score = self.fitness_repeated_fragments(melody_np, 3)
        
        scores = {
            'diversity': diversity_score,
            'diversity_interval': diversity_interval_score,
            'dissonance': dissonance_score,
            'rhythmic_average_value': rhythmic_average_value,
            'deviation_rhythmic_value': deviation_rhythmic_value,
            'scale_conformance': scale_conformance_score,
            'chord_conformance': chord_conformance_score,
            'melodic_contour': melodic_contour_score,
            'pitch_range': pitch_range_score,
            'large_intervals': large_intervals_score,
            'average_pitch': average_pitch_score,
            'pitch_variation': pitch_variation_score,
            'strong_beat': strong_beat_score,
            'chord_beat': in_chord_beat_score,
            'root_beat': root_beat_score,
            'odd_index_notes': odd_index_notes_score,
            'average_interval': average_interval_score,
            'repetition_2': repetition_2_score,
            'repetition_3': repetition_3_score,
        }

        fitness_value = sum(
            weights[feature] * np.exp(-0.5 * ((score - mu_values[feature]) / sigma_values[feature]) ** 2)
            for feature, score in scores.items()
        )

        # additional bonus for root notes
        first_chord_shorthand = self.find_chord_at_time(0)
        first_note = Chords.from_shorthand(first_chord_shorthand)[0]
        tonic = int(Note(first_note, 2))

        root_value = 0
        if melody_np[0] == tonic:
            root_value += 1
        elif melody_np[0] % 12 == tonic % 12:
            root_value += 0.8

        fitness_value += root_value * 10

        return fitness_value

    def fitness_strong_beat(self, melody_np):
        """
        Calculates scores for notes placed on strong beats and their conformance with chords and root note.

        :param melody_np: A NumPy array representing the melody.
        :return: A tuple containing the strong beat score and in-chord beat score.
        """
        strong_beat_interval = int(self.meter[0] / self.note_duration * 4 / self.meter[1])
        extension_interval = int(self.meter[0] / 2 / self.note_duration * 4 / self.meter[1])

        strong_beat_counter = 0
        in_chord_beat_counter = 0
        root_beat_counter = 0
        for i in range(0, len(melody_np), strong_beat_interval):
            if melody_np[i] > 0:  
                strong_beat_counter += 1
                chord_notes = Chords.from_shorthand(self.find_chord_at_time(i * self.note_duration))
                in_chord = NOTES.int_to_note(melody_np[i] % 12) in chord_notes
                root = NOTES.int_to_note(melody_np[i] % 12) == chord_notes[0]
                if in_chord:
                    in_chord_beat_counter += 1
                if root:
                    root_beat_counter += 1
                
                for j in range(1, extension_interval):
                    if i + j < len(melody_np) and melody_np[i + j] == -2:
                        strong_beat_counter += 1
                        if in_chord:
                            in_chord_beat_counter += 1
                        if root:
                            root_beat_counter += 1

        denominator = len(melody_np) / strong_beat_interval * extension_interval
        if denominator == 0:  
            return 0

        strong_beat_score = strong_beat_counter / denominator
        in_chord_beat_score = in_chord_beat_counter / denominator
        root_beat_score = root_beat_counter / denominator

        return strong_beat_score, in_chord_beat_score, root_beat_score

    def fitness_scale_chord_and_root(self, melody_np):
        """
        Calculates the conformance of the melody to the scale, chords and root note.

        :param melody_np: A NumPy array representing the melody.
        :return: A tuple of (scale_conformance_score, chord_conformance_score).
        """
        scale_notes_np = np.array(self.scale_notes)

        scale_length_counter = 0
        chord_length_counter = 0
        total_length_counter = 0
        root_length_counter = 0
        in_scale = False
        in_chord = False
        root = False

        for i, note in enumerate(melody_np):
            if note != -2:
                in_scale = note in scale_notes_np
                chord_notes = Chords.from_shorthand(self.find_chord_at_time(i * self.note_duration))
                in_chord = NOTES.int_to_note(note % 12) in chord_notes
                root = NOTES.int_to_note(note % 12) == chord_notes[0]
            total_length_counter += 1
            if in_scale:
                scale_length_counter += 1
            if in_chord:
                chord_length_counter += 1
            if root:
                root_length_counter += 1

        if total_length_counter != 0:  
            scale_conformance_score = scale_length_counter / total_length_counter
            chord_conformance_score = chord_length_counter / total_length_counter
            root_conformance_score = root_length_counter / total_length_counter
        else:
            return 0

        return scale_conformance_score, chord_conformance_score, root_conformance_score

    def mutate(self, melody):
        """
        Mutation override for Bass. Applies subset of mutations with musical significance to the bass track
        based on a mutation rate.

        :param melody: A list representing the bass melody.
        :return: A mutated bass melody as a list.
        """
        melody_np = np.array(melody)
        valid_indices = np.where((melody_np != -1) & (melody_np != -2))[0]

        if random.random() < self.MUTATION_RATE and len(valid_indices) > 1:
            melody_np = self.mutate_interval(melody_np, valid_indices)

        if random.random() < self.MUTATION_RATE and len(valid_indices) > 0:
            melody_np = self.mutate_extend(melody_np, valid_indices)

        if random.random() < self.MUTATION_RATE and len(valid_indices) > 1:
            melody_np = self.mutate_long_notes(melody_np, valid_indices)

        if random.random() < self.MUTATION_RATE:
            melody_np = self.mutate_replace_extension(melody_np)

        if len(valid_indices) > 0 and random.random() < self.MUTATION_RATE:
            melody_np = self.mutate_length_norm(melody_np, valid_indices)

        if random.random() < self.MUTATION_RATE:
            melody_np = self.mutate_sort(melody_np)

        if random.random() < self.MUTATION_RATE:
            melody_np = self.mutate_repeat1(melody_np)

        if random.random() < self.MUTATION_RATE:
            melody_np = self.mutate_repeat2(melody_np)

        return melody_np.tolist()


class GeneticMotifGenerator(GeneticMelodyGenerator):
    """
    A class for generating motifs using a genetic algorithm, inheriting from GeneticMelodyGenerator.

    Inherits from:
        GeneticMelodyGenerator: Provides the base functionality for melody generation and genetic algorithm operations.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, measures=1):
        """
        Executes the genetic algorithm to generate a motif track based on the specified mode and number of measures.

        Depending on the mode (`'long_motif'`, `'start_motif'`, `'end_motif'`, `'repeated_motif'`, `'repeated_short_motif'`),
        the method generates a population of melodies, applies selection, crossover, and mutation operations, and evolves
        the melodies over multiple generations. The best melody is selected and extended to the desired number of measures.
        The method handles different types of melodies and variations according to the mode and arousal parameters.

        :param measures: Number of measures to generate the motif for. Defaults to 1.
        :return: A list of notes representing the final generated motif.
        """

        concatenated_melody = []
        if self.mode in ['long_motif', 'long']:
            note_amount = int(self.meter[0] / self.note_duration * 4 / self.meter[1])
            population = [random.choices(self.NOTES, k=note_amount) for _ in range(self.POPULATION_SIZE)]

            for generation in range(self.NUM_GENERATIONS):
                print(f'Generation {generation + 1}/{self.NUM_GENERATIONS}')
                population.sort(key=self.fitness, reverse=True)
                new_population = []
                while len(new_population) < self.POPULATION_SIZE:
                    parent1, parent2 = random.choices(population[:10], k=2)
                    child1, child2 = self.crossover(parent1, parent2)
                    new_population.extend([self.mutate(child1), self.mutate(child2)])
                population = new_population

            best_melody = population[0]

            first_chord_shorthand = self.find_chord_at_time(0)
            first_note = Chords.from_shorthand(first_chord_shorthand)[0]
            tonic = int(Note(first_note, 4))
            
            melody_np = np.array(best_melody)
            concatenated_melody = best_melody
            for i in range(1, measures):
                root_note_shorthand = self.find_chord_at_time(i * self.meter[0] * 4 / self.meter[1])
                root = int(Note(Chords.from_shorthand(root_note_shorthand)[0], 4))

                interval = root - tonic
                
                mask = melody_np >= 0
                transformed_melody = melody_np.copy()
                transformed_melody[mask] += interval
                transformed_melody[mask] = np.vectorize(self.find_nearest)(transformed_melody[mask])

                concatenated_melody.extend(transformed_melody.tolist())
        elif self.mode in ['start_motif', 'start']:
            note_amount = int(self.meter[0] / self.note_duration * 4 / self.meter[1] / 2)
            population = [random.choices(self.NOTES, k=note_amount) for _ in range(self.POPULATION_SIZE)]

            for generation in range(self.NUM_GENERATIONS):
                print(f'Generation {generation + 1}/{self.NUM_GENERATIONS}')
                population.sort(key=self.fitness, reverse=True)
                new_population = []
                while len(new_population) < self.POPULATION_SIZE:
                    parent1, parent2 = random.choices(population[:10], k=2)
                    child1, child2 = self.crossover(parent1, parent2)
                    new_population.extend([self.mutate(child1), self.mutate(child2)])
                population = new_population

            best_melody = population[0]
            best_melody.extend([-1] * note_amount)

            first_chord_shorthand = self.find_chord_at_time(0)
            first_note = Chords.from_shorthand(first_chord_shorthand)[0]
            tonic = int(Note(first_note, 4))
            
            melody_np = np.array(best_melody)
            concatenated_melody = best_melody
            for i in range(1, measures):
                root_note_shorthand = self.find_chord_at_time(i * self.meter[0] * 4 / self.meter[1])
                root = int(Note(Chords.from_shorthand(root_note_shorthand)[0], 4))

                interval = root - tonic
                
                mask = melody_np >= 0
                transformed_melody = melody_np.copy()
                transformed_melody[mask] += interval
                transformed_melody[mask] = np.vectorize(self.find_nearest)(transformed_melody[mask])

                concatenated_melody.extend(transformed_melody.tolist())
        elif self.mode in ['end_motif', 'end']:
            note_amount = int(self.meter[0] / self.note_duration * 4 / self.meter[1] / 2)
            population = [random.choices(self.NOTES, k=note_amount) for _ in range(self.POPULATION_SIZE)]

            for generation in range(self.NUM_GENERATIONS):
                print(f'Generation {generation + 1}/{self.NUM_GENERATIONS}')
                population.sort(key=self.fitness, reverse=True)
                new_population = []
                while len(new_population) < self.POPULATION_SIZE:
                    parent1, parent2 = random.choices(population[:10], k=2)
                    child1, child2 = self.crossover(parent1, parent2)
                    new_population.extend([self.mutate(child1), self.mutate(child2)])
                population = new_population

            best_melody = population[0]
            melody = [-1] * note_amount
            melody.extend(best_melody)

            first_chord_shorthand = self.find_chord_at_time(0)
            first_note = Chords.from_shorthand(first_chord_shorthand)[0]
            tonic = int(Note(first_note, 4))
            
            melody_np = np.array(melody)
            concatenated_melody = melody
            for i in range(1, measures):
                root_note_shorthand = self.find_chord_at_time(i * self.meter[0] * 4 / self.meter[1])
                root = int(Note(Chords.from_shorthand(root_note_shorthand)[0], 4))

                interval = root - tonic
                
                mask = melody_np >= 0
                transformed_melody = melody_np.copy()
                transformed_melody[mask] += interval
                transformed_melody[mask] = np.vectorize(self.find_nearest)(transformed_melody[mask])

                concatenated_melody.extend(transformed_melody.tolist())
        elif self.mode in ['repeated_motif', 'repeated'] or (self.mode in ['repeated_short_motif', 'repeated_short'] and self.meter[0] / self.note_duration * 4 / self.meter[1] / 4 < 2):
            note_amount = self.meter[0] / self.note_duration * 4 / self.meter[1] / 2
            rounded_note_amount = int(note_amount)
            note_amount_difference = note_amount - rounded_note_amount

            population = [random.choices(self.NOTES, k=rounded_note_amount) for _ in range(self.POPULATION_SIZE)]

            for generation in range(self.NUM_GENERATIONS):
                print(f'Generation {generation + 1}/{self.NUM_GENERATIONS}')
                population.sort(key=self.fitness, reverse=True)
                new_population = []
                while len(new_population) < self.POPULATION_SIZE:
                    parent1, parent2 = random.choices(population[:10], k=2)
                    child1, child2 = self.crossover(parent1, parent2)
                    new_population.extend([self.mutate(child1), self.mutate(child2)])
                population = new_population

            best_melody = population[0]
            best_melody_repeated = best_melody * 2

            if note_amount_difference > 0:
                additional_notes_length = int(round(note_amount_difference * 4))
                additional_notes = best_melody[:additional_notes_length]
                best_melody_repeated += additional_notes

            best_melody = best_melody_repeated

            first_chord_shorthand = self.find_chord_at_time(0)
            first_note = Chords.from_shorthand(first_chord_shorthand)[0]
            tonic = int(Note(first_note, 4))
            
            melody_np = np.array(best_melody)
            concatenated_melody = best_melody
            for i in range(1, measures):
                root_note_shorthand = self.find_chord_at_time(i * self.meter[0] * 4 / self.meter[1])
                root = int(Note(Chords.from_shorthand(root_note_shorthand)[0], 4))

                interval = root - tonic
                
                mask = melody_np >= 0
                transformed_melody = melody_np.copy()
                transformed_melody[mask] += interval
                transformed_melody[mask] = np.vectorize(self.find_nearest)(transformed_melody[mask])

                concatenated_melody.extend(transformed_melody.tolist())

        elif self.mode in ['repeated_short_motif', 'repeated_short']:
            note_amount = self.meter[0] / self.note_duration * 4 / self.meter[1] / 4
            rounded_note_amount = int(note_amount)
            note_amount_difference = note_amount - rounded_note_amount

            population = [random.choices(self.NOTES, k=rounded_note_amount) for _ in range(self.POPULATION_SIZE)]

            for generation in range(self.NUM_GENERATIONS):
                print(f'Generation {generation + 1}/{self.NUM_GENERATIONS}')
                population.sort(key=self.fitness, reverse=True)
                new_population = []
                while len(new_population) < self.POPULATION_SIZE:
                    parent1, parent2 = random.choices(population[:10], k=2)
                    child1, child2 = self.crossover(parent1, parent2)
                    new_population.extend([self.mutate(child1), self.mutate(child2)])
                population = new_population

            best_melody = population[0]
            best_melody_repeated = best_melody * 4
            if note_amount_difference > 0:
                additional_notes_length = int(round(note_amount_difference * 4))
                additional_notes = best_melody[:additional_notes_length]
                best_melody_repeated += additional_notes

            best_melody = best_melody_repeated

            first_chord_shorthand = self.find_chord_at_time(0)
            first_note = Chords.from_shorthand(first_chord_shorthand)[0]
            tonic = int(Note(first_note, 4))
            melody_np = np.array(best_melody)
            concatenated_melody = best_melody
            for i in range(1, measures):
                root_note_shorthand = self.find_chord_at_time(i * self.meter[0] * 4 / self.meter[1])
                root = int(Note(Chords.from_shorthand(root_note_shorthand)[0], 4))

                interval = root - tonic
                mask = melody_np >= 0
                transformed_melody = melody_np.copy()
                transformed_melody[mask] += interval
                transformed_melody[mask] = np.vectorize(self.find_nearest)(transformed_melody[mask])

                concatenated_melody.extend(transformed_melody.tolist())

        return concatenated_melody

    def get_musical_coefficients(self):
        """
        Specifies the mean and standard deviation values conditioned on mode and emotions for each musical feature,
        as well as their weights, used for fitness calculation.

        :return: A tuple of dictionaries containing the mean and standard deviation values, and a dictionary of weights.
        """

        valence = self.valence * 2 - 1
        arousal = self.arousal * 2 - 1

        mu_values = {
            'diversity': 0.9,
            'diversity_interval': 0.8 + arousal * 0.1 + valence * 0.1,
            'dissonance': 0.25 - valence * 0.25,
            'scale_conformance': 0.9 - arousal * 0.1,
            'chord_conformance': 0.8 - arousal * 0.1 - valence * 0.1,
            'root_conformance': 0.4 - arousal * 0.1,
            'melodic_contour': 0.1 + self.valence * 0.8,
            'pitch_range': 0.6 + arousal * 0.2,
            'large_intervals': 0.0,
            'average_pitch': 0.4 + arousal * 0.1 + valence * 0.1,
            'scale_playing': 0.5 + valence + 0.1,
        }
        sigma_values = {
            'diversity': 0.1,
            'diversity_interval': 0.1,
            'dissonance': 0.1,
            'scale_conformance': 0.1,
            'chord_conformance': 0.1,
            'root_conformance': 0.1,
            'melodic_contour': 0.1,
            'pitch_range': 0.1,
            'large_intervals': 0.1,
            'average_pitch': 0.1,
            'scale_playing': 0.1,
        }
        weights = {
            'diversity': 5,
            'diversity_interval': 8,
            'dissonance': 5,
            'scale_conformance': 5,
            'chord_conformance': 5,
            'root_conformance': 3,
            'melodic_contour': 5,
            'pitch_range': 5,
            'large_intervals': 1,
            'average_pitch': 2,
            
            'scale_playing': 3,
        }


        return mu_values, sigma_values, weights

    def fitness(self, melody):
        """
        Evaluates the fitness of a given melody based on measured and expected values of musical features.

        :param melody: A list or array representing the melody to be evaluated.
        :return: Fitness value computed based on the weighted combination of various scores.
        """

        mu_values, sigma_values, weights = self.get_musical_coefficients()
        melody_np = np.array(melody)

        dissonance_score, large_intervals_score = self.fitness_intervals(melody_np)
        scale_conformance_score, chord_conformance_score = self.fitness_scale_and_chord(melody_np)
        diversity_score = self.fitness_note_diversity(melody_np)
        diversity_interval_score = self.fitness_diversity_intervals(melody_np)
        melodic_contour_score = self.fitness_melodic_contour(melody_np)  
        pitch_range_score = self.fitness_note_range(melody_np)
        average_pitch_score = self.fitness_average_pitch(melody_np)
        scale_playing_score = self.fitness_small_intervals(melody_np)
        
        scores = {
            'diversity': diversity_score,
            'diversity_interval': diversity_interval_score,
            'dissonance': dissonance_score,
            'scale_conformance': scale_conformance_score,
            'chord_conformance': chord_conformance_score,
            'melodic_contour': melodic_contour_score,
            'pitch_range': pitch_range_score,
            'large_intervals': large_intervals_score,
            'average_pitch': average_pitch_score,
            'scale_playing': scale_playing_score,
        }

        fitness_value = sum(
            weights[feature] * np.exp(-0.5 * ((score - mu_values[feature]) / sigma_values[feature]) ** 2)
            for feature, score in scores.items()
        )

        repeated_root_score = self.fitness_repeated_root(melody_np)
        fitness_value += repeated_root_score * 5 * self.arousal
        
        return fitness_value

    def fitness_diversity_intervals(self, melody_np):
        """
        Computes the average diversity of intervals.

        :param melody_np: A NumPy array representing the melody.
        :return: The average interval diversity score.
        """

        melody_np = melody_np[(melody_np != -1) & (melody_np != -2)]
        intervals = np.diff(melody_np)
        intervals = intervals[intervals <= 12]
        unique_intervals = len(np.unique(np.abs(intervals)))

        return unique_intervals / (len(intervals)) if len(intervals) > 1 else 0

    def fitness_repeated_root(self, melody_np):
        """
        Evaluates the fitness of a melody based on the repetition of notes.

        The fitness score is affected by penalties and rewards depending on note repetitions and other factors.

        :param melody_np: A numpy array of melody notes, where -1 and -2 are treated as non-notes.
        :return: The normalized fitness score, where higher scores indicate better fitness.
        """
        penalty = -20

        if self.valence < 0.5 and self.arousal >= 0.5:
            small_reward = 10 * (1 - self.valence)
            large_reward = 20 * (1 - self.valence)
        elif self.mode in ['long_motif']:
            small_reward = 1
            large_reward = 5
        else:
            small_reward = -10
            large_reward = -1

        first_chord_shorthand = self.find_chord_at_time(0)
        first_note = Chords.from_shorthand(first_chord_shorthand)[0]
        tonic = int(Note(first_note, 4))

        melody_np = melody_np[(melody_np != -1) & (melody_np != -2)]

        score = 0

        for i in range(1, len(melody_np)):
            if melody_np[i] == melody_np[i - 1]:
                score += penalty

            if i > 1 and melody_np[i] == melody_np[i - 2]:
                score += small_reward if melody_np[i] % 12 != tonic % 12 and melody_np[i] not in self.NOTES[:len(
                    self.NOTES)] else large_reward
            if i > 2 and melody_np[i] == melody_np[i - 3]:
                score += small_reward if melody_np[i] % 12 != tonic % 12 and melody_np[i] not in self.NOTES[:len(
                    self.NOTES)] else large_reward

            return score / len(melody_np) if len(melody_np) > 0 else 0

    def mutate(self, melody):
        """
        Mutation override for motif. Applies subset of mutations with musical significance to the motif track
        based on a mutation rate.

        :param melody: A list representing the motif.
        :return: A mutated motif as a list.
        """
        melody_np = np.array(melody)
        valid_indices = np.where((melody_np != -1) & (melody_np != -2))[0]

        if random.random() < self.MUTATION_RATE and len(valid_indices) > 1:
            melody_np = self.mutate_interval(melody_np, valid_indices)

        if random.random() < self.MUTATION_RATE and self.mode in ['long_motif']:
            melody_np = self.mutate_repeat2(melody_np)

        if random.random() < self.MUTATION_RATE:
            melody_np = self.mutate_sort(melody_np)


        return melody_np.tolist()


