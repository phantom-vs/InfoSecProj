import wave
import numpy as np
import streamlit as st
from statistics import mean
from scipy.fftpack import fft, ifft
from random import uniform
import math
import cmath
from bitarray import bitarray

types = {
    1: np.int8,
    2: np.int16,
    4: np.int32
}

bits_types = {
    1: 2**8.  / 2,
    2: 2**16. / 2,
    4: 2**64. / 2,
}

def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)

def channel2byte_block(channel, block_len):
    st.write("To byte blocks")
    my_bar = st.progress(0)
    byte_channel = []
    k_ = len(channel) // block_len
    for i in range(k_):
        byte_channel.append(channel[i*block_len:(i+1)*block_len])
        my_bar.progress(int((i+1)*100 / k_))
    return byte_channel

def data2bits_block(data_bytes):
    data_bits = bitarray()
    data_bits.frombytes(data_bytes)

    blocks_data_bits = []
    for i in range(len(data_bits)//8):
        blocks_data_bits.append(data_bits[i*8:(i+1)*8])

    return blocks_data_bits

def upc1(c0, c1, m, delta_pi, eps):
    while mean(c1) - mean(c0) < delta_pi:
        c0 = [clamp(_c0 - delta_pi / m, -math.pi+eps, math.pi-eps) for _c0 in c0]
        c1 = [clamp(_c1 + delta_pi / m, -math.pi+eps, math.pi-eps) for _c1 in c1]
    return c0 + c1

def upc0(c0, c1, m, delta_pi, eps):
    while mean(c0) - mean(c1) < delta_pi:
        c0 = [clamp(_c0 + delta_pi / m, -math.pi+eps, math.pi-eps) for _c0 in c0]
        c1 = [clamp(_c1 - delta_pi / m, -math.pi+eps, math.pi-eps) for _c1 in c1]
    return c0 + c1

def furie_transform(byte_channel, total_bytes):
    st.write("Calculate fft")
    my_bar = st.progress(0)

    phase = []
    amplitude = []

    for i in range(total_bytes):
        fur  = fft(byte_channel[i])
        phase.append([cmath.phase(comp) for comp in fur])
        amplitude.append([abs(comp) for comp in fur])
        my_bar.progress(int((i+1)*100 / total_bytes))

    return phase, amplitude

def change_phase(phase, bits_data, block_len, m, k, delta_pi, eps):

    st.write("Change phase")
    my_bar = st.progress(0)

    # loop over bytes
    for i, bits in enumerate(bits_data):
        byte_ph = phase[i][1:(block_len - 1) // 2 + 1]
        new_ph = []
        # loop over bits
        for j, bit in enumerate(bits):
            # loop over repetitions
            bit_ph = byte_ph[j * m * k: (j + 1) * m * k]
            for g in range(k):
                c = bit_ph[g * m: (g + 1) * m]
                c0 = c[:m // 2]
                c1 = c[m // 2:]
                if bit == 0:
                    if mean(c0) - mean(c1) < delta_pi:
                        c = upc0(c0, c1, m, delta_pi, eps)
                if bit == 1:
                    if mean(c1) - mean(c0) < delta_pi:
                        c = upc1(c0, c1, m, delta_pi, eps)

                new_ph.extend(c)
        phase[i] = [0] + new_ph + [-x for x in new_ph[::-1]]
        my_bar.progress(int((i + 1)*100 / len(bits_data)))

    return phase

def decode_message(phase, num_bytes, block_len, k, m):
    st.write("Decode message")
    my_bar = st.progress(0)

    bytes = b''

    for i in range(num_bytes):
        bits = ""
        byte_ph = phase[i][1:(block_len - 1) // 2 + 1]
        for j in range(8):
            bit_ph = byte_ph[j * m * k: (j + 1) * m * k]
            bit_count = 0
            for g in range(k):
                c = bit_ph[g * m: (g + 1) * m]
                c0 = c[:m // 2]
                c1 = c[m // 2:]
                if mean(c1) > mean(c0): bit_count += 1

            if math.floor(bit_count / (k / 2.0)) == 0:
                bits += "0"
            else:
                bits += "1"
        bytes += bitarray(bits).tobytes()
        my_bar.progress(int((i + 1) * 100 / num_bytes))

    return bytes

def inverse_furie_transform(byte_channel, phase, amplitude):
    st.write("Calculate ifft")
    my_bar = st.progress(0)

    i = 0
    for ph, amp in zip(phase, amplitude):
        new_furie = [complex(amp_ * math.cos(ph_), amp_ * math.sin(ph_)) for ph_, amp_ in zip(ph, amp)]
        new_furie[0] = complex(0, 0)
        byte_channel[i] = ifft(new_furie).astype(np.float32)
        i += 1
        my_bar.progress(int(i * 100 / len(phase)))
    return byte_channel


def main():
    option = st.sidebar.radio(
        'Choose option',
        ('Embed', 'Extract'))

    st.sidebar.title("Algorithm parameters")

    m = int(st.sidebar.number_input('samples per bit', value=40))
    k = int(st.sidebar.number_input('repetitions per bit', value=5))
    delta_pi = st.sidebar.number_input('delta_pi', value=0.6)
    noise_eps = st.sidebar.number_input('noise range', value=0.01) / 10
    eps = 1e-1
    block_len = m * k * 8 * 2 + 1  # for 1 byte

    if   option == 'Embed': encrypt(m, k, delta_pi, noise_eps, eps, block_len)
    elif option == 'Extract': decrypt(m, k, delta_pi, noise_eps, eps, block_len)

def encrypt(m, k, delta_pi, noise_eps, eps, block_len):
    st.title("Embed the message into audio")

    st.title("Wav file")
    wav_file  = st.file_uploader("Choose a wav file", key ="0")
    if wav_file is None: return
    st.audio(wav_file)
    wav_file_name = wav_file.name
    wav_file = wave.open(wav_file)

    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav_file.getparams()
    content = wav_file.readframes(nframes)
    samples = np.fromstring(content, dtype=types[sampwidth])
    st.write(f"nchannels = {nchannels}, bytes per sample = {sampwidth}")
    st.write(f"frames per second = {framerate}, total number of frames = {nframes}")
    st.write(f"{nframes // block_len} Bytes of data can be encrypted")

    # secret data
    data = st.text_input('Enter the string to decode', '')
    if data == '': return
    byte_data = data.encode()
    bits_data = data2bits_block(byte_data)
    total_bytes = len(bits_data)

    st.title("Encryption")
    # read 1 channels
    channel = samples[0::nchannels]
    # normalize chanel
    st.write("Normalize channel")
    bar0 = st.progress(0)
    norm_channel = []
    for i, amp in enumerate(channel):
        noise = uniform(-noise_eps, noise_eps)
        norm_amp = amp / bits_types[sampwidth]
        norm_channel.append(clamp(norm_amp + noise, -1 + eps, 1  - eps))
        if i % 1000 == 0: bar0.progress(int((i + 1)*100/len(channel)))
    bar0.progress(100)
    # divide channel to byte block
    byte_channel = channel2byte_block(norm_channel, block_len)
    # calculate fft
    phase, amplitude = furie_transform(byte_channel, total_bytes)
    
    st.write("Initial block")
    st.bar_chart(byte_channel[0])
    
    st.write("Initial phase")
    st.bar_chart(phase[0])

    # change phase
    phase = change_phase(phase, bits_data, block_len, m, k, delta_pi, eps)
    
    st.write("New phase")
    st.bar_chart(phase[0])

    # calculate new furies
    byte_channel = inverse_furie_transform(byte_channel, phase, amplitude)

    # calculate new channel
    new_channel = []
    for block in byte_channel:
        amp_block = [amp*bits_types[sampwidth] for amp in block]
        new_channel.extend(amp_block)
    new_channel = np.array(new_channel, dtype=types[sampwidth])
    # calculate new sample
    for i in range(len(new_channel)):
        samples[i*nchannels] = new_channel[i]
    # calculate new content
    new_content = samples.tostring()

    st.write("New block")
    st.bar_chart(byte_channel[0])

    # new wav file
    wav1_file = wave.open(f"{wav_file_name}_enc", mode='w')
    wav1_file.setnchannels(nchannels)
    wav1_file.setsampwidth(sampwidth)
    wav1_file.setframerate(framerate)
    wav1_file.setnframes(nframes)
    wav1_file.writeframes(new_content)
    wav1_file.close()

    f = open(f"{wav_file_name}_enc", mode='rb').read()
    st.audio(f)
    st.download_button('Download result file', data = f, file_name=f"{wav_file_name}_enc.wav")

def decrypt(m, k, delta_pi, noise_eps, eps, block_len):
    st.title("Extract the message from audio")

    st.title("Wav file")
    wav_file = st.file_uploader("Choose a wav file", key ="2")
    if wav_file is None: return
    st.audio(wav_file)
    wav_file = wave.open(wav_file)

    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav_file.getparams()
    st.write(f"nchannels = {nchannels}, bytes per sample = {sampwidth}")
    st.write(f"frames per second = {framerate}, total number of frames = {nframes}")

    # get wav data
    content = wav_file.readframes(nframes)
    samples = np.fromstring(content, dtype=types[sampwidth])

    st.title("Decryption")

    num_bytes = int(st.number_input('num byte to decode', value=0))
    if num_bytes == 0: return

    # read 1 channels
    channel = samples[0::nchannels]

    # normalized chanel
    norm_channel = [amp / bits_types[sampwidth] for amp in channel]

    # divide channel
    byte_channel = channel2byte_block(norm_channel, block_len)

    # calculate fft
    phase, _ = furie_transform(byte_channel, num_bytes)
    
    st.write("Phase")
    st.bar_chart(phase[0])

    # message
    bytes = decode_message(phase, num_bytes, block_len, k, m)

    #secret_str = ''.join([byte for byte in bytes])
    st.write(bytes)

if __name__ == '__main__':
    main()
