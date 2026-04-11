import numpy as np
from ChannelSimulation import Channel
from Modem import Modem

np.random.seed(69)

def generate_training_data(nts, bpf, snr_limit):
    print("Generating training data")
    modem = Modem(fs=44100, f0=2000, f1=8000, symbol_duration=0.02)
    channel = Channel(snr_db=snr_limit)
    training_signals = []

    for _ in range(nts):
        source_signal = np.random.randint(0, 2, bpf)
        snr = np.random.randint(20, snr_limit)
        channel.snr_db = snr

        modulated_signal = modem.modulate(source_signal)
        processed_signal = channel.process(modulated_signal)

        training_signals.append(processed_signal)

    return training_signals

def generate_frame(bpf, snr_limit, anomaly=None):
    modem = Modem(fs=44100 ,f0=2000, f1=8000, symbol_duration=0.02)
    channel = Channel(snr_db=np.random.randint(20, snr_limit))

    source_signal = np.random.randint(0, 2, bpf)
    modulated_signal = modem.modulate(source_signal)
    processed_signal = channel.process(modulated_signal, anomaly_type=anomaly)

    return processed_signal
