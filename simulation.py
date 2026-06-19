import numpy as np

from modem.channel import Channel
from modem.transmit import Transmitter
from modem.receive import Receiver
from modem.codec import Codec

def run_simulation(message_text, snr_db=30):
    codec = Codec()
    modulator = Transmitter()
    channel = Channel()
    demodulator = Receiver()
    
    print(f"Source: '{message_text}'")
    
    data_bits = codec.encode(message_text)
    length_bits = format(len(message_text), '08b') 
    
    payload_bits = length_bits + data_bits
    
    padding_needed = (32 - (len(payload_bits) % 32)) % 32
    payload_bits += '0' * padding_needed
    
    print(f"Modulated bits: {len(payload_bits)} bit")
    
    tx_signal = []
    for i in range(0, len(payload_bits), 32):
        s_tdata = payload_bits[i:i+32]
        ofdm_frame = modulator.modulate_symbol(s_tdata)
        tx_signal.extend(ofdm_frame)
        
    tx_signal = np.array(tx_signal)
    print(f"Complex frames to send: {len(tx_signal)}")
    
    rx_signal = channel.apply_noise(tx_signal, snr_db=snr_db)
    
    recovered_bits = ""
    for i in range(0, len(rx_signal), 10):
        frame_10 = rx_signal[i:i+10]
        recovered_bits += demodulator.demodulate_symbol(frame_10)
        
    rx_msg_len = int(recovered_bits[:8], 2)
    data_start = 8
    data_end = data_start + (rx_msg_len * 6) 
    
    rx_data_bits = recovered_bits[data_start:data_end]
    decoded_message = codec.decode(rx_data_bits)
    
    print(f"Recieved: '{decoded_message}'")
    
    return decoded_message

if __name__ == "__main__":
    run_simulation("Hello world", snr_db=100)