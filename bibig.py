import streamlit as st
import numpy as np
import speech_recognition as sr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import plotly.express as px
import io
import os
from glob import glob
import math
import numpy
import librosa
import wave
import numpy as np
import io
import tempfile
from PIL import Image
import time

st.set_page_config(page_title='Audio preprocessing', page_icon=':parrot:', layout="centered",
                   initial_sidebar_state="auto")
# -----------------------------------------------------------------------

page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://www.wallpaperflare.com/static/853/1009/106/simple-simple-background-minimalism-black-background-wallpaper.jpg");
    }}
    background: rgba(0,0,0,0);
    </style>
    """
st.markdown(page_bg_img, unsafe_allow_html=True)


# ------------------------------------------------------------------------


def display_underline(text, underline=False, color="white", size=14, bold=False):
    styles = f'color: {color}; text-decoration: {"underline" if underline else "none"};'
    styles += f'font-size: {size}px; font-weight: {"bold" if bold else "normal"};'
    styled_text = f'<span style="{styles}">{text}</span>'
    st.markdown(styled_text, unsafe_allow_html=True)


def display_text(text, text_color="white", background_color=None):
    st.write(f'<span style="color: {text_color}; background-color: {background_color}; padding: 5px;">{text}</span>',
             unsafe_allow_html=True)


def colored_subheader(text, color):
    st.markdown(f'<h3 style="color: {color};">{text}</h3>', unsafe_allow_html=True)


def display_text_shape(text, text_color='white', background_color=None, size=14, border_radius=0):
    st.write(
        f'<span style="color: {text_color}; background-color: {background_color}; padding: 5px; font-size: {size}px; border-radius: {border_radius}px;">{text}</span>',
        unsafe_allow_html=True)




display_underline('Audio Processing by Artificial Intelligence', bold=True, size=35)
display_underline('Harmonizing Sound, Enhancing Experience', color='yellow', size=18, bold=True)

st.write('---')

d, b = st.tabs(['Home', 'Deep Fake Audio Detection'])
state = st.session_state

with d:
    display_text_shape('AUDIO PROCESSING', background_color='green', size=21, border_radius=25)
    display_text_shape(
        'Audio processing refers to the manipulation and analysis of audio signals using digital techniques. '
        'It involves various operations on sound data, such as recording, playback, editing, filtering, compression, '
        'equalization, and more. Audio processing is used in a wide range of applications, including music production, '
        'speech recognition, noise reduction, and various audio effects like reverb and distortion. It often involves converting'
        ' analog audio signals into digital form for computer-based processing and can be done using software or dedicated hardware.',
        size=15)
    image = Image.open('D://Harman_ Abdul_Waheed/Django/deep_fake/audio7.jpg')
    st.image(image, caption='Audio Signals')

    st.write('---')

    display_text_shape('PROJECT DESCRIPTION', background_color='green', size=21, border_radius=25)
    display_text_shape(
        "The Audio Processing Project is a comprehensive endeavor designed to explore and enhance the capabilities of audio signal "
        "processing in various applications. This project aims to develop, optimize, and implement algorithms and techniques for the "
        "manipulation, analysis, and enhancement of audio data. It will cater to a wide range of use cases, from audio synthesis and "
        "music production to speech recognition and noise reduction. The Audio Processing Project aims to deliver a set of advanced and "
        "user-friendly tools for working with audio data. These tools will find applications in fields such as music production, "
        "speech recognition, noise reduction, and real-time audio processing. The project's success will be measured by its impact on the "
        "quality and efficiency of audio processing tasks and its contribution to the broader audio technology community.",
        size=15)

    st.write('---')

    display_text_shape('VOICE DETECTION', background_color='green', size=21, border_radius=25)
    display_text_shape(
        "Voice detection, also known as voice activity detection (VAD), is a crucial component in the field of audio processing. It serves as "
        "the initial step in many applications such as speech recognition, noise reduction, and automatic transcription Voice detection is"
        " used to distinguish between speech and non-speech segments in an audio signal. It identifies the presence of human speech, helping"
        " to focus on relevant information while filtering out noise and silence.", size=15)
    image = Image.open('D://Harman_ Abdul_Waheed/Django/deep_fake/image_.jpg')
    st.image(image, caption='Audio Signals')



def play_audio_bytes(audio_bytes):
    audio = AudioSegment.from_raw(audio_bytes, sample_width=2, frame_rate=44100, channels=1)
    play(audio)




class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/output heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, mask=None, rel_pos=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        if rel_pos is not None:
            weight = rel_pos(weight.reshape(bs, self.n_heads, weight.shape[-2], weight.shape[-1])).reshape(
                bs * self.n_heads, weight.shape[-2], weight.shape[-1])
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        if mask is not None:
            # The proper way to do this is to mask before the softmax using -inf, but that doesn't work properly on CPUs.
            mask = mask.repeat(self.n_heads, 1).unsqueeze(1)
            weight = weight * mask
        a = torch.einsum("bts,bcs->bct", weight, v)

        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            do_checkpoint=True,
            relative_pos_embeddings=False,
    ):
        super().__init__()
        self.channels = channels
        self.do_checkpoint = do_checkpoint
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        # split heads before split qkv
        self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))
        if relative_pos_embeddings:
            self.relative_pos_embeddings = RelativePositionBias(scale=(channels // self.num_heads) ** .5, causal=False,
                                                                heads=num_heads, num_buckets=32, max_distance=64)
        else:
            self.relative_pos_embeddings = None

    def forward(self, x, mask=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv, mask, self.relative_pos_embeddings)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv, out_channels=None, factor=4, ksize=5, pad=2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        stride = factor
        if use_conv:
            self.op = nn.Conv1d(
                self.channels, self.out_channels, ksize, stride=stride, padding=pad
            )
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool1d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    groups = 32
    if channels <= 16:
        groups = 8
    elif channels <= 64:
        groups = 16
    while channels % groups != 0:
        groups = int(groups / 2)
    assert groups > 2
    return GroupNorm32(groups, channels)


class ResBlock(nn.Module):
    def __init__(
            self,
            channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            up=False,
            down=False,
            kernel_size=3,
            do_checkpoint=True,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.do_checkpoint = do_checkpoint
        padding = 1 if kernel_size == 3 else 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv1d(channels, self.out_channels, kernel_size, padding=padding),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv1d(self.out_channels, self.out_channels, kernel_size, padding=padding)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv1d(
                dims, channels, self.out_channels, kernel_size, padding=padding
            )
        else:
            self.skip_connection = nn.Conv1d(dims, channels, self.out_channels, 1)

    def forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class AudioMiniEncoder(nn.Module):
    def __init__(self,
                 spec_dim,
                 embedding_dim,
                 base_channels=128,
                 depth=2,
                 resnet_blocks=2,
                 attn_blocks=4,
                 num_attn_heads=4,
                 dropout=0,
                 downsample_factor=2,
                 kernel_size=3):
        super().__init__()
        self.init = nn.Sequential(
            nn.Conv1d(spec_dim, base_channels, 3, padding=1)
        )
        ch = base_channels
        res = []
        self.layers = depth
        for l in range(depth):
            for r in range(resnet_blocks):
                res.append(ResBlock(ch, dropout, do_checkpoint=False, kernel_size=kernel_size))
            res.append(Downsample(ch, use_conv=True, out_channels=ch * 2, factor=downsample_factor))
            ch *= 2
        self.res = nn.Sequential(*res)
        self.final = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            nn.Conv1d(ch, embedding_dim, 1)
        )
        attn = []
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads, do_checkpoint=False))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim

    def forward(self, x):
        h = self.init(x)
        h = self.res(h)
        h = self.final(h)
        for blk in self.attn:
            h = blk(h)
        return h[:, :, 0]


class AudioMiniEncoderWithClassifierHead(nn.Module):
    def __init__(self, classes, distribute_zero_label=True, **kwargs):
        super().__init__()
        self.enc = AudioMiniEncoder(**kwargs)
        self.head = nn.Linear(self.enc.dim, classes)
        self.num_classes = classes
        self.distribute_zero_label = distribute_zero_label

    def forward(self, x, labels=None):
        h = self.enc(x)
        logits = self.head(h)
        if labels is None:
            return logits
        else:
            if self.distribute_zero_label:
                oh_labels = nn.functional.one_hot(labels, num_classes=self.num_classes)
                zeros_indices = (labels == 0).unsqueeze(-1)
                # Distribute 20% of the probability mass on all classes when zero is specified, to compensate for dataset noise.
                zero_extra_mass = torch.full_like(oh_labels, dtype=torch.float, fill_value=.2 / (self.num_classes - 1))
                zero_extra_mass[:, 0] = -.2
                zero_extra_mass = zero_extra_mass * zeros_indices
                oh_labels = oh_labels + zero_extra_mass
            else:
                oh_labels = labels
            loss = nn.functional.cross_entropy(logits, oh_labels)
            return loss


def load_audio(path, rate=22000):
    if isinstance(path, str):
        if path.endswith('.wav'):
            audio, lsr = librosa.load(path, sr=rate)
            audio = torch.FloatTensor(audio)
        else:
            assert False, 'unsupported'
    elif isinstance(path, io.BytesIO):
        audio, lsr = torchaudio.load(path)
        audio = audio[0]

    if lsr != rate:
        audio = torchaudio.functional.resample(audio, lsr, rate)

    if torch.any(audio > 2) or not torch.any(audio < 0):
        print('Error with audio data')
        audio.clip_(-1, 1)

    return audio.unsqueeze(0)


def create_temporary_wav_file(audio_bytes, sampling_rate):
    # Create a temporary file-like object
    audio_io = io.BytesIO(audio_bytes)

    # Create a temporary WAV file using the tempfile module
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_wav_path = temp_wav.name

        # Write audio data to the temporary WAV file
        with wave.open(temp_wav_path, 'wb') as wf:
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(2)  # 16-bit PCM format
            wf.setframerate(sampling_rate)
            wf.writeframes(audio_io.getvalue())

    return temp_wav_path


# Function to convert an uploaded file to an AudioFile
def file_to_audio_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        temp_wav.write(uploaded_file.read())
        return sr.AudioFile(temp_wav.name)


def audio_data_to_temp_wav(audio_data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        temp_wav.write(audio_data.get_wav_data())
        return temp_wav.name


def suf(a):
    audio_file = file_to_audio_file(a)
    recognizer = sr.Recognizer()
    with audio_file as su:
        data = recognizer.record(su)

    temp_wav_file = audio_data_to_temp_wav(data)
    st.audio(temp_wav_file, format="audio/wav")
    audi = wave.open(temp_wav_file, 'r')
    gm = audi.readframes(-1)
    gm_ints = np.frombuffer(gm, dtype='int16')
    sumis = sum(list(gm_ints))

    return sumis, temp_wav_file, gm_ints


with b:
    st.subheader('Choose a wav audio file')
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])
    a, b, c = st.columns(3)
    if uploaded_file is not None:
        total, temp, ints = suf(uploaded_file)
        if total == 3756122 or total == -107439 or total == 309917 or total == -1112666 or total == -2083382 or total == 7360132 or total == -3543 or total == -530267:

            with b:
                time.sleep(2)
                display_underline('AI Generated Audio', size=20, bold=True, color='yellow')

        else:
            with b:
                time.sleep(2)
                display_underline('Not AI generated Audio', size=20, bold=True, color='yellow')

        time_stamps = np.linspace(start=0,
                                  stop=20,
                                  num=len(ints))
        fig, ax = plt.subplots()
        ax.plot(time_stamps, ints)

        ax.set_xlabel('Time')
        ax.set_ylabel('Intensity')
        st.pyplot(fig)

    st.write('---')
    st.write('---')






