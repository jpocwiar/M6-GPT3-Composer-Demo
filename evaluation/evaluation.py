import os
import muspy
import numpy as np

folder_path = "../generated_midi/prompt-driven"

pitch_class_entropy_list = []
scale_consistency_list = []
groove_consistency_list = []
bars_list = []

for file_name in os.listdir(folder_path):
    if file_name.endswith(".mid"):
        midi_file_path = os.path.join(folder_path, file_name)
        music = muspy.read_midi(midi_file_path)

        ticks_per_bar = music.resolution * 4
        total_ticks = music.get_end_time()
        total_bars = total_ticks / ticks_per_bar

        bars_list.append(total_bars)

        pitch_class_entropy_list.append(muspy.pitch_class_entropy(music))
        scale_consistency_list.append(muspy.scale_consistency(music))
        groove_consistency_list.append(muspy.groove_consistency(music, measure_resolution=4))

def weighted_mean(values, weights):
    return np.sum(values * weights) / np.sum(weights)

def weighted_std(values, weights):
    mean = weighted_mean(values, weights)
    variance = np.sum(weights * (values - mean) ** 2) / np.sum(weights)
    return np.sqrt(variance)

metrics = {
    "Pitch Class Entropy": np.array(pitch_class_entropy_list),
    "Scale Consistency (%)": np.array(scale_consistency_list) * 100,
    "Groove Consistency (%)": np.array(groove_consistency_list) * 100
}

bars = np.array(bars_list)
total_bars = np.sum(bars)

for metric_name, values in metrics.items():
    mean_value = weighted_mean(values, bars)
    std_dev = weighted_std(values, bars)
    print(f"{metric_name}: {mean_value:.3f} ± {std_dev:.3f}")

print(f"Total Bars: {total_bars:.0f}")