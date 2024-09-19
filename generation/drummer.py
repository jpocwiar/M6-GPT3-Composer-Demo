import random
import numpy as np

class Drummer:
    """
    A class for generating drum patterns based on emotional parameters and rhythm structure.

    Attributes:
        meter (tuple): The time signature, default is (4, 4).
        note_duration (float): The fundamental duration of each note in the rhythm, default is an eight note.
        bpm (int): Beats per minute (tempo) for the drum pattern, default is 120 bpm.
        valence (float): A value between 0 and 1 representing the emotional positivity of the drum pattern.
        arousal (float): A value between 0 and 1 representing the energy level of the drum pattern.
        density (float): The calculated rhythmic density.
        midi_notes (dict): A dictionary mapping drum sounds to MIDI note numbers for the selected kit.
        tom_transitions (list): Predefined transitions between tom sounds for drum fills.
        pattern_length (int): The amount of states in a drum pattern, determined by the meter and density.
    """
    def __init__(self, meter=(4, 4), note_duration=0.5, bpm=120, valence=0.5, arousal=0.5, kit="classic"):
        self.meter = meter
        self.note_duration = note_duration
        self.bpm = bpm
        self.valence = valence
        self.arousal = arousal
        self.density = self.calculate_density_from_arbitrary()
        self.midi_notes = self.get_kit_sounds(kit=kit)
        self.tom_transitions = self.initialize_tom_transitions()
        self.pattern_length = int(self.meter[0] * 16 / self.meter[1] * self.density)

        #probabilities for kick and snare drum
        self.kick_snare_patterns = {
            (2, 4): {'bass_drum': np.array([0.95, 0.05]), 'snare': np.array([0.01, 0.95])},
            (3, 4): {'bass_drum': np.array([0.95, 0.025, 0.075]), 'snare': np.array([0.01, 0.95, 0.55])},
            (4, 4): {'bass_drum': np.array([0.95, 0.05, 0.85, 0.05]), 'snare': np.array([0.01, 0.95, 0.05, 0.95])},
            (5, 4): {'bass_drum': np.array([0.95, 0.05, 0.05, 0.9, 0.05]),
                     'snare': np.array([0.01, 0.95, 0.01, 0.01, 0.95])},
            (6, 4): {'bass_drum': np.array([0.95, 0.05, 0.05, 0.85, 0.05, 0.05]),
                     'snare': np.array([0.01, 0.95, 0.01, 0.95, 0.01, 0.95])},
            
            (4, 8): {'bass_drum': np.array([0.95, 0.45, 0.05, 0.05]), 'snare': np.array([0.01, 0.01, 0.9, 0.05])},
            (5, 8): {'bass_drum': np.array([0.95, 0.01, 0.01, 0.01, 0.01]),
                     'snare': np.array([0.01, 0.01, 0.01, 0.95, 0.01])},
            (6, 8): {'bass_drum': np.array([0.95, 0.01, 0.05, 0.01, 0.01, 0.01]),
                     'snare': np.array([0.01, 0.01, 0.01, 0.95, 0.01, 0.05])},
            (7, 8): {'bass_drum': np.array([0.95, 0.01, 0.05, 0.01, 0.9, 0.01, 0.01]),
                     'snare': np.array([0.01, 0.01, 0.95, 0.01, 0.01, 0.01, 0.95])},
        }

        multiplier = 0.5 + 0.5 * self.calculate_logarithmic_arousal_multiplier(0.0001)
        for pattern in self.kick_snare_patterns.values():
            pattern['snare'] *= multiplier

        self.kick_snare_probs = self.get_or_generate_probs(self.meter)

    def get_kit_sounds(self, kit='classic'):
        """
        Returns a list of MIDI notes corresponding to drum kit sounds based on the specified kit type.

        The method selects different sets of drum sounds (represented by General MIDI notes) based on the
        given kit type. The kit type determines which set of instruments and MIDI notes to return.
        If the kit type is not recognized, a default set of drum sounds is provided.

        :param kit: A string indicating the type of drum kit. Possible values include:
                    - 'african', 'ethnic', 'world': For a world percussion kit.
                    - 'cymbals', 'orchestral', 'accent', 'bells_and_cymbals': For a cymbals and orchestral kit.
                    - Any other value: Defaults to a standard drum kit.
        :return: A list of MIDI note numbers associated with the selected drum kit sounds.
        """

        if kit.lower() in ['african', 'ethnic', 'world']:
            # instruments list for reference
            instruments = ['high_woodblock', 'low_woodblock', 'low_timbale', 'high_timbale', 'low_bongo',
                                        'high_bongo',
                                        'mute_high_conga', 'open_high_conga', 'low_conga', 'short_guiro', 'Claves',
                                        'long_guiro']
            midi_notes = [76, 77, 45, 65, 61, 60, 62, 63, 64, 73, 75, 74][::-1]
        elif kit.lower() in ['cymbals', 'orchestral', 'accent', "bells_and_cymbals"]:
            instruments = ['closed_triangle', 'open_triangle', 'bass_drum', 'splash_cymbal',
                                       'chinese_cymbal',
                                       'ride_bell', 'tambourine', 'side_stick', 'cowbell', 'crash_cymbal_1',
                                       'ride_cymbal_1', 'ride_cymbal_2']
            midi_notes = [80, 81, 35, 55, 52, 53, 75, 37, 56, 49, 51, 59][::-1]
        else:
            instruments = ['closed_hi_hat', 'open_hi_hat', 'bass_drum', 'snare', 'low_floor_tom', 'low_tom',
                                'low_mid_tom',
                                'high_mid_tom', 'high_tom', 'crash', 'ride', 'bell']
            midi_notes = [42, 46, 36, 40, 41, 45, 47, 48, 50, 49, 51, 53][::-1]
        return midi_notes

    def calculate_density_from_arbitrary(self, bpm_thresh=140, arousal_thresh=0.5):
        """
        Calculates the density of drum patterns based on arbitrary thresholds for BPM and arousal.

        :param bpm_thresh: The threshold value for BPM. Default is 140.
        :param arousal_thresh: Float, the threshold value for arousal. Default is 0.5.
        :return: Float, the calculated density which is either 0.5 or 1 based on the conditions.
        """
        if ((self.bpm > bpm_thresh and self.arousal < 0.9) or (self.arousal < arousal_thresh and self.bpm > bpm_thresh // 2)) and self.meter[1] < 16:
            density = 0.5
        else:
            density = 1
        return density

    def calculate_logarithmic_arousal_multiplier(self, linearity=0.2):
        """
        Calculates a logarithmic multiplier based on arousal and a linearity parameter.

        The multiplier is used so that the relationship between arousal and the multiplier is not linear, which is more intuitive.

        :param linearity: Float, the linearity parameter used in the logarithmic transformation. Default is 0.2.
        :return: Float, the calculated multiplier which ranges from 0 to 1.
        """
        log_multiplier = np.log(self.arousal + linearity)
        max_log = np.log(1 + linearity)
        min_log = np.log(linearity)
        multiplier = (log_multiplier - min_log) / (max_log - min_log)
        return multiplier

    def get_or_generate_probs(self, meter):
        """
       Retrieves or generates probability patterns for kick and snare based on the time signature (meter).

       This function checks if the probability patterns for a given meter are already
       available. If they are, it returns them. If not, it recursively splits the meter
       and combines the probabilities from the resulting parts.

       If the time signature is based on notes shorter than eights, it is treated like it is based on eights for pattern
       matching purposes.

       :param meter: Tuple of two integers (beats, notes), representing the time signature.
       :return: Dictionary with keys 'bass_drum' and 'snare', each containing a numpy array of probabilities.
       """
        if meter[1] > 8:
            meter = (meter[0], 8)
        
        if meter in self.kick_snare_patterns:
            return self.kick_snare_patterns[meter]
        
        elif (meter[0], 4) in self.kick_snare_patterns:
            return self.kick_snare_patterns[(meter[0], 4)]
        else:
            
            parts = self.split_beats(meter[0], meter[1])
            
            probs_parts = [self.get_or_generate_probs((part, meter[1])) for part in parts]  
            
            combined_probs = {
                'bass_drum': np.concatenate([prob['bass_drum'] for prob in reversed(probs_parts)]),
                'snare': np.concatenate([prob['snare'] for prob in reversed(probs_parts)])
            }
            return combined_probs

    def split_beats(self, beats, notes):
        """
        Recursively splits beats into smaller parts to match existing kick and snare patterns.

        This function divides the given number of beats into smaller parts recursively until
        the beats can be matched with existing patterns. The function returns a list of
        split beats.

        :param beats: Integer, the total number of beats to be split (left number in the time signature).
        :param notes: Integer, the type of notes in the time signature (right number in the time signature).
        :return: List of integers representing the split parts of beats.
        """
        if (beats, notes) in self.kick_snare_patterns:
            return [beats]
        else:
            lower_part = beats // 2
            upper_part = beats - lower_part
            return self.split_beats(lower_part, notes) + self.split_beats(upper_part, notes)

    def initialize_tom_transitions(self):
        """
        Initializes the transition probabilities for tom drums.

        :return: Dictionary where keys are tom states (represented as binary strings) and values are dictionaries
        mapping possible transitions to their probabilities.
        """
        return {
            '00000': {'00000': 0.1, '10000': 0.1, '01000': 0.1, '00100': 0.2, '00010': 0.2, '00001': 0.1, '00110': 0.1, 'snare': 0.1},
            '10000': {'00000': 0.4, '10000': 0.1, '01000': 0.2, '11000': 0.2, 'snare': 0.1},
            '01000': {'00000': 0.4, '10000': 0.1, '01000': 0.1, '11000': 0.1, '01100': 0.1, '00100': 0.1, 'snare': 0.1},
            '11000': {'00000': 0.2, '10000': 0.2, '01000': 0.2, '11000': 0.1, '01100': 0.2, 'snare': 0.1},
            '01100': {'00000': 0.2, '01000': 0.2, '00100': 0.2, '11000': 0.1, '01100': 0.1, '00110': 0.1, 'snare': 0.1},
            '00100': {'00000': 0.4, '01000': 0.1, '00100': 0.1, '01100': 0.1, '00110': 0.1, '00010': 0.1, 'snare': 0.1},
            '00110': {'00000': 0.2, '00100': 0.2, '00010': 0.2, '01100': 0.1, '00110': 0.1, '00011': 0.1, 'snare': 0.1},
            '00010': {'00000': 0.4, '00100': 0.1, '00010': 0.1, '00110': 0.1, '00011': 0.1, '00001': 0.1, 'snare': 0.1},
            '00011': {'00000': 0.2, '00010': 0.2, '00001': 0.2, '00110': 0.2, '00011': 0.1, 'snare': 0.1},
            '00001': {'00000': 0.4, '00001': 0.1, '00010': 0.2, '00011': 0.2, 'snare': 0.1},
        }

    def generate_hi_hat_pattern(self, frequency, end_ohh_prob=1.0):
        """
        Generates a pattern for hi-hat hits based on the specified frequency and optional probability of ending with an open hi-hat.

        :param frequency: The frequency of closed hi-hat hits. Valid values are 1 (every sixteenth note), 0.5 (every eight note), and 0.25 (every quarter note).
        :param end_ohh_prob: Probability of ending with an open hi-hat at the last CHH position and middle of the pattern.
        :return: A tuple of two lists:
                 - `chh_pattern`: A list representing the closed hi-hat pattern.
                 - `ohh_pattern`: A list representing the open hi-hat pattern.
        """
        
        chh_pattern = ['0'] * self.pattern_length  
        ohh_pattern = ['0'] * self.pattern_length  

        if frequency == 1:
            chh_pattern = ['1'] * self.pattern_length
        elif frequency == 0.5:
            num_ones = (self.pattern_length + 1) // 2
            chh_pattern[::2] = ['1'] * num_ones
        elif frequency == 0.25:
            num_ones = (self.pattern_length + 1) // 4
            chh_pattern[::4] = ['1'] * len(chh_pattern[::4])
        else:
            return chh_pattern, ohh_pattern

        last_chh_position = len(chh_pattern) - 1 - chh_pattern[::-1].index('1')

        middle_position = self.pattern_length // 2 - 1 if self.pattern_length % 2 == 0 else self.pattern_length // 2

        if random.random() < end_ohh_prob and chh_pattern[last_chh_position] == '1':
            ohh_pattern[last_chh_position] = '1'
            chh_pattern[last_chh_position] = '0'

        middle_ohh_prob = min(end_ohh_prob, self.arousal)
        
        if random.random() < middle_ohh_prob and chh_pattern[middle_position] == '1':
            ohh_pattern[middle_position] = '1'
            chh_pattern[middle_position] = '0'

        return chh_pattern, ohh_pattern

    def generate_crash_pattern(self, crash_probability):
        """
        Generates a pattern for crash cymbals based on a given crash probability.

        The crash pattern is influenced by the arousal level and the provided crash probability.
        The pattern varies with a logarithmic multiplier based on arousal.

        :param crash_probability: Probability of a crash hit occurring at each position in the pattern.
        :return: A list representing the crash cymbal pattern.
        """
        crash_pattern = ['0'] * self.pattern_length
        multiplier = self.calculate_logarithmic_arousal_multiplier(0.1)
        for i in range(self.pattern_length):
            if i == 0:
                crash_pattern[i] = '1' if random.random() < multiplier * crash_probability else '0'
            elif i % int(16 / self.meter[1]) == 0:
                crash_pattern[i] = '1' if random.random() < multiplier * crash_probability / 10 else '0'
            else:
                crash_pattern[i] = '1' if random.random() < multiplier * crash_probability / 500 else '0'
        return crash_pattern

    def generate_ride_bell_pattern(self, interval=4):
        """
        Generates patterns for ride cymbal and Bell hits based on arousal and valence.

        The ride cymbal pattern is created based on the interval and arousal level, while the Bell pattern is generated 
        based on valence and arousal levels. The interval may be halved if arousal is high.

        :param interval: The interval (in beats) between ride cymbal hits. The default value is 4.
        :return: A tuple of two lists:
                 - `ride_pattern`: A list representing the ride cymbal pattern.
                 - `bell_pattern`: A list representing the Chinese cymbal pattern.
        """
        ride_pattern = ['0'] * self.pattern_length
        bell_pattern = ['0'] * self.pattern_length

        if self.arousal > 0.8:
            interval //= 2  

        if self.valence >= 0.5 and self.arousal > 0.5:
            for i in range(interval // 2, self.pattern_length, interval):
                ride_pattern[i] = '1'

        if self.arousal > 0.7 and self.valence < 0.5:
            for i in range(0, self.pattern_length, interval):
                bell_pattern[i] = '1'

        return ride_pattern, bell_pattern

    def generate_kick_snare_pattern(self):
        """
        Generates a pattern for kick drum and snare drum hits based on predefined probabilities.

        :return: A tuple of two lists:
                 - `kick_pattern`: A list representing the kick drum pattern.
                 - `snare_pattern`: A list representing the snare drum pattern.
        """
        
        kick_pattern = ['0'] * self.pattern_length
        snare_pattern = ['0'] * self.pattern_length
        interval = int(16 / self.meter[1])
        for i in range(0, self.pattern_length, interval):
            
            
            if random.random() < self.kick_snare_probs['bass_drum'][i // interval % len(self.kick_snare_probs['bass_drum'])]:
                kick_pattern[i] = '1'
            if random.random() < self.kick_snare_probs['snare'][i // interval % len(self.kick_snare_probs['bass_drum'])]:
                snare_pattern[i] = '1'
        return kick_pattern, snare_pattern


    def generate_tom_pattern(self, fill_probability, fill_beats, snare_pattern):
        """
        Generates a pattern for fills which include tom and snare hits.

        :param fill_probability: Probability of adding fill in the pattern.
        :param fill_beats: Number of beats over which fill can be added.
        :param snare_pattern: A list representing the snare drum pattern, which may be updated with additional snare hits.
        :return: A tuple of two lists:
                 - `pattern`: A list representing the tom pattern.
                 - `snare_pattern`: The updated snare drum pattern.
        """
        pattern = ['00000'] * self.pattern_length
        fill_length = min(int(self.meter[0] * fill_beats * self.density * 4 / self.meter[1]), self.pattern_length)
        if random.random() < fill_probability:
            current_state = '00000'  
            for i in range(-fill_length, 0, 1):
                current_state = self.choose_next_state(current_state, self.tom_transitions)
                if current_state == 'snare':
                    snare_pattern[i] = '1'
                    current_state = '00000'
                else:
                    pattern[i] = current_state
        return pattern, snare_pattern

    def choose_next_state(self, current_state, transitions):
        """
        Chooses the next state for fills based on the current state and transition probabilities.

        :param current_state: The current state from which the next state is to be chosen.
        :param transitions: A dictionary mapping states to their possible next states and associated probabilities.
        :return: The next state chosen based on the transition probabilities.
        """
        next_states = transitions[current_state]
        next_state = random.choices(list(next_states.keys()), weights=list(next_states.values()))[0]
        return next_state

    def mutate_pattern(self, pattern):
        """
        Mutates a given pattern To further increase diversity and realism

        Includes random repeating of snare or bass drum as well as zeroing out some bits so that the amount of drums
        played by drummers' hands is not greater than 2.

        :param pattern: A list representing the pattern to be mutated.
        :return: A list representing the mutated pattern.
        """
        mutated_pattern = pattern.copy()
        pattern_length = len(pattern)

        for i in range(pattern_length):
            state = list(pattern[i])

            multiplier = self.calculate_logarithmic_arousal_multiplier(0.1)
            if state[2] == '1':
                
                if i + 1 < pattern_length and random.random() < 0.2 * multiplier:
                    mutated_pattern[i + 1] = mutated_pattern[i + 1][:2] + '1' + mutated_pattern[i + 1][3:]
                
                if i + 2 < pattern_length and random.random() < 0.5 * multiplier:
                    mutated_pattern[i + 2] = mutated_pattern[i + 2][:2] + '1' + mutated_pattern[i + 2][3:]

            if state[3] == '1':
                
                if i + 1 < pattern_length and random.random() < 0.1 * multiplier:
                    mutated_pattern[i + 1] = mutated_pattern[i + 1][:3] + '1' + mutated_pattern[i + 1][4:]
                
                if i + 2 < pattern_length and random.random() < 0.1 * multiplier:
                    mutated_pattern[i + 2] = mutated_pattern[i + 2][:3] + '1' + mutated_pattern[i + 2][4:]

            bits_to_consider = [bit for index, bit in enumerate(state) if index not in [3, 4]]
            if sum(int(bit) for bit in bits_to_consider) > 2:
                
                while sum(int(bit) for bit in bits_to_consider) > 2:
                    
                    one_indices = [index for index, bit in enumerate(state) if
                                   bit == '1' and index not in [3, 4]]
                    
                    bit_to_flip = random.choice(one_indices)
                    state[bit_to_flip] = '0'
                    
                    bits_to_consider = [bit for index, bit in enumerate(state) if index not in [3, 4]]
                mutated_pattern[i] = ''.join(state)

            if random.random() < 0.6 * (1.00 - self.arousal) * self.density:

                state[4:-2] = ['0'] * (len(state) - 4 - 2)
                mutated_pattern[i] = ''.join(state)

        return mutated_pattern

    def generate_pattern(self, hi_hat_frequency, fill_probability, fill_beats=1):
        """
        Generates a complete drum pattern by combining multiple individual patterns and applying mutation.

        The method generates patterns for hi-hat, kick, snare, toms, crash, ride, and bell.

        :param hi_hat_frequency: Frequency of hi-hat hits (e.g., 1 for every sixteenth note, 0.5 for every eight etc.).
        :param fill_probability: Probability of adding fill in the pattern.
        :param fill_beats: Number of beats over which fill can be added in the tom pattern.
        :return: A list representing the mutated complete drum pattern.
        """
        chh_pattern, ohh_pattern = self.generate_hi_hat_pattern(hi_hat_frequency)
        kick_pattern, snare_pattern = self.generate_kick_snare_pattern()
        tom_pattern, snare_pattern = self.generate_tom_pattern(fill_probability, fill_beats, snare_pattern)
        crash_pattern = self.generate_crash_pattern(fill_probability)
        ride_pattern, bell_pattern = self.generate_ride_bell_pattern(interval=4)

        complete_pattern = []
        for i in range(self.pattern_length):
            combined_pattern = chh_pattern[i] + ohh_pattern[i] + kick_pattern[i] + snare_pattern[i] + tom_pattern[i] + crash_pattern[i] + ride_pattern[i] + bell_pattern[i]
            complete_pattern.append(combined_pattern)
        mutated_pattern = self.mutate_pattern(complete_pattern)
        return mutated_pattern

    def generate_section(self, fill_probability, fill_beats=1, fill_frequency=1, fill_end_boost=0, measures=1, repeats=1):
        """
        Generates a drum section composed of multiple patterns, considering arousal and fill parameters.

        The method generates a sequence of drum states for the specified number of measures. It adjusts the fill probability and
        hi-hat frequency based on arousal and applies additional fills at the end of the section if needed. The generated section
        is repeated a specified number of times.

        :param fill_probability: Probability of adding fills in the pattern.
        :param fill_beats: Number of beats over which fills can be added.
        :param fill_frequency: Frequency of fill occurrences in the pattern.
        :param fill_end_boost: Additional fill probability applied at the end of the section.
        :param measures: Number of measures to generate the drums for.
        :param repeats: Number of times to repeat the generated section.
        :return: A list representing the generated and repeated drum section.
        """
        if self.arousal > 0.9:
            fill_end_add_beats = self.meter[0]
            fill_end_boost += 0.4
        elif self.arousal > 0.8:
            fill_end_add_beats = self.meter[0] // 2
            fill_end_boost += 0.2
        elif self.arousal > 0.5:
            fill_end_add_beats = self.meter[0] // 4
            fill_end_boost += 0.1
        else:
            fill_end_add_beats = 0

        if self.arousal > 0.85:
            hi_hat_frequency = 1
        elif self.arousal > 0.4:
            hi_hat_frequency = 0.5
        elif self.arousal > 0.2:
            hi_hat_frequency = 0.25
        else:
            hi_hat_frequency = 0

        section = []
        for i in range(measures):
            if i == measures - 1:
                pattern = self.generate_pattern(hi_hat_frequency, fill_probability + fill_end_boost,  fill_beats + fill_end_add_beats) 
            elif (i + 1) % fill_frequency == 0: 
                pattern = self.generate_pattern(hi_hat_frequency, fill_probability, min(self.meter[0], fill_beats))
            else:
                pattern = self.generate_pattern(hi_hat_frequency, 0, min(self.meter[0], fill_beats))
            section.extend(pattern)
        section = section * repeats
        return section

    def calculate_velocity_from_arousal(self, arousal):
        """
        Calculates the MIDI velocity based on arousal level.

        :param arousal: A value representing the arousal level, influencing the velocity.
        :return: An integer representing the calculated MIDI velocity.
        """
        return random.randint(max(60, int(40 + (arousal * 40))), min(100, int(60 + (arousal * 40))))

    def write_drums(self, midi, track, patterns, start_time=0):
        """
        Writes drum patterns to a MIDI track with appropriate timing and velocity adjustments.

        :param midi: The MIDI object to which notes are added.
        :param track: The track in the MIDI object where notes will be added.
        :param patterns: A list of drum patterns to be written to the MIDI track.
        :param start_time: The starting time for the first note, used to offset the timing.
        :return: None
        """
        time_offset_range = (-0.01, 0.01)
        time_offset_range = tuple([x * (1 + self.arousal) for x in time_offset_range])

        for beat, pattern in enumerate(patterns):
            pattern = pattern.ljust(len(self.midi_notes),
                                    '0')  
            pattern_int = int(pattern, 2)  
            for i, note in enumerate(self.midi_notes):
                if pattern_int & (1 << i):
                    time_offset = round(random.uniform(*time_offset_range), 5)
                    
                    if note in [38, 40] and self.arousal < 0.4:
                        note = 37
                    midi.addNote(track, 9, note, max(0, beat / (4 * self.density) + start_time + time_offset), 0.25,
                                 self.calculate_velocity_from_arousal(self.arousal))
