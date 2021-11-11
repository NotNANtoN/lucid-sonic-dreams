import sys
import os
import shutil
import pickle 
import inspect
import numpy as np
import random

import scipy
import torch

import skimage.exposure
import librosa
import soundfile
import pygit2
import moviepy.editor as mpy
from moviepy.audio.AudioClip import AudioArrayClip
import PIL
from PIL import Image
from importlib import import_module
from tqdm import tqdm
from scipy.stats import truncnorm

from .helper_functions import * 
from .sample_effects import *


def import_stylegan_torch():
    # Clone Official StyleGAN2-ADA Repository
    #if not os.path.exists('stylegan2'):
    #    pygit2.clone_repository('https://github.com/NVlabs/stylegan2-ada-pytorch.git',
    #                            'stylegan2')
    # StyleGan2 imports
    #sys.path.append("stylegan2")
    #import legacy
    #import dnnlib
    
    if not os.path.exists('stylegan3'):
        pygit2.clone_repository('https://github.com/NVlabs/stylegan3.git',
                                'stylegan3')


    # StyleGan23imports
    sys.path.append("stylegan3")
    import legacy
    import dnnlib


def import_stylegan_tf():
    print("Cloning tensorflow...")
    if not os.path.exists('stylegan2_tf'):
        pygit2.clone_repository('https://github.com/NVlabs/stylegan2-ada.git',
                          'stylegan2_tf')

    #StyleGAN2 Imports
    sys.path.append("stylegan2_tf")
    import dnnlib as dnnlib
    from dnnlib.tflib.tfutil import convert_images_to_uint8 as convert_images_to_uint8
    init_tf()


def import_clmr():
    if not os.path.exists("clmr"):
        pygit2.clone_repository("https://github.com/spijkervet/clmr", "clmr")
    sys.path.append("clmr")
    
    
def show_styles():
    '''Show names of available (non-custom) styles'''

    all_models = consolidate_models()
    styles = set([model['name'].lower() for model in all_models])
    print(*styles, sep='\n')


def minmax(array, axis=0):
                return (array - array.min(axis, keepdims=True)) / (array.max(axis, keepdims=True) - array.min(axis, keepdims=True))
    

def slerp(low, high, val):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    epsilon = 1e-7
    omega = (low_norm * high_norm).sum(1)
    omega = torch.acos(torch.clamp(omega, -1 + epsilon, 1 - epsilon))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res


class Welford():
    def __init__(self,a_list=None):
        self.n = 0
        self.M = 0
        self.S = 0

    def update(self, x):
        self.n += 1
        newM = self.M + (x - self.M) / self.n
        newS = self.S + (x - self.M) * (x - newM)
        self.M = newM
        self.S = newS

    @property
    def mean(self):
        return self.M

    @property
    def std(self):
        if self.n == 1:
            return 0
        return np.sqrt(self.S / (self.n - 1))
   

class MultiTensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_list):
        self.tensor_list = tensor_list
        len_ = len(tensor_list[0])
        for t in tensor_list:
            assert len(t) == len_, "All tensors must be of the same length!"

    def __getitem__(self, i):
        return [torch.from_numpy(t[i]) for t in self.tensor_list]

    def __len__(self):
        return len(self.tensor_list[0])
    
    
class LatentsDataset(torch.utils.data.Dataset):
    def __init__(self, latent_path):
        self.latent_path = latent_path
        assert os.path.exists(self.latent_path)
        self.files = sorted([f for f in os.listdir(latent_path) if f.endswith(".npy")])

    def __getitem__(self, i):
        return torch.from_numpy(np.load(os.path.join(self.latent_path, f"{i}.npy")))

    def __len__(self):
        return len(self.files)


class LucidSonicDream:
  def __init__(self, 
               song: str,
               pulse_audio: str = None,
               motion_audio: str = None,
               class_audio: str = None,
               contrast_audio: str = None,
               flash_audio: str = None,
               style: str = 'wikiart',
               num_possible_classes: int = None,
               model_type: str = 'stylegan',  # stylegan, vqgan
               width: int = 496,
               height: int = 496,
               ): 

      # If style is a function, raise exception if function does not take 
      # noise_batch or class_batch parameters
    if callable(style):
     
        func_sig = list(inspect.getfullargspec(style))[0]

        for arg in ['noise_batch', 'class_batch']:
            if arg not in func_sig:
                sys.exit('func must be a function with parameters '\
                       'noise_batch and class_batch')

        # Raise exception if input_shape or num_possible_classes is not provided
        if (input_shape is None) or (num_possible_classes is None):
            sys.exit('input_shape and num_possible_classes '\
                     'must be provided if style is a function')

    self.model_type = model_type
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define attributes
    self.song = song
    self.song_name = song.split("/")[-1].split(".")[0].replace(".mp3", "").replace(".", "")
    self.pulse_audio = pulse_audio
    self.motion_audio = motion_audio
    self.class_audio = class_audio
    self.contrast_audio = contrast_audio
    self.flash_audio = flash_audio
    # stylegan params
    self.style = style
    self.num_possible_classes = num_possible_classes 
    self.style_exists = False
    # vqgan params
    self.width = width
    self.height = height
    
    if self.model_type == "stylegan":
        # some stylegan models cannot be converted to pytorch (wikiart)
        self.use_tf = style in ("wikiart",)
        if self.use_tf:
            #import_stylegan_tf()
            print("Cloning tensorflow...")
            if not os.path.exists('stylegan2_tf'):
                pygit2.clone_repository('https://github.com/NVlabs/stylegan2-ada.git',
                                  'stylegan2_tf')

            #StyleGAN2 Imports
            sys.path.append("stylegan2_tf")
            self.dnnlib = import_module("dnnlib")
            #import dnnlib as dnnlib
            #from dnnlib.tflib.tfutil import convert_images_to_uint8
            tflib = import_module("dnnlib.tflib.tfutil")
            self.convert_images_to_uint8 = tflib.convert_images_to_uint8#import_module("dnnlib.tflib.tfutil", fromlist=["convert_images_to_uint8"])
            self.init_tf = tflib.init_tf #import_module("dnnlib.tflib.tfutil", fromlist=["init_tf"])
            self.init_tf()
            #init_tf()
        else:
            #import_stylegan_torch()
            # Clone Official StyleGAN2-ADA Repository
            if not os.path.exists('stylegan3'):
                pygit2.clone_repository('https://github.com/NVlabs/stylegan3.git',
                                        'stylegan3')


            # StyleGan23imports
            sys.path.append("stylegan3")

            #import legacy
            #import dnnlib
            self.dnnlib = import_module("dnnlib")
            self.legacy = import_module("legacy")
    

  def stylegan_init(self):
    '''Initialize StyleGAN(2) weights'''

    style = self.style

    # Initialize TensorFlow
    #if self.use_tf:
    #    init_tf() 

    # If style is not a .pkl file path, download weights from corresponding URL
    if '.pkl' not in style:
      all_models = consolidate_models()
      all_styles = [model['name'].lower() for model in all_models]

      # Raise exception if style is not valid
      if style not in all_styles:  
        sys.exit('Style not valid. Call show_styles() to see all ' \
        'valid styles, or upload your own .pkl file.')

      download_url = [model for model in all_models \
                      if model['name'].lower() == style][0]\
                      ['download_url']
      weights_file = style + '.pkl'

      # If style .pkl already exists in working directory, skip download
      if not os.path.exists(weights_file):
        print('Downloading {} weights (This may take a while)...'.format(style))
        try:
          download_weights(download_url, weights_file)
        except Exception:
          exc_msg = 'Download failed. Try to download weights directly at {} '\
                    'and pass the file path to the style parameter'\
                    .format(download_url)
          sys.exit(exc_msg)
        print('Download complete')

    else:
      weights_file = style

    # load generator
    if self.use_tf:
        # Load weights
        with open(weights_file, 'rb') as f:
            self.model = pickle.load(f)[2]
    else:
        print(f'Loading networks from {weights_file}...')
        with self.dnnlib.util.open_url(weights_file) as f:
            self.model = self.legacy.load_network_pkl(f)['G_ema'].to(self.device) # type: ignore
    
    # Auto assign num_possible_classes attribute
    try:
      print(self.model.mapping.input_templates)
      self.num_possible_classes = self.model.mapping.input_templates[1].shape[1]
    except ValueError:
      print(self.model.mapping.static_kwargs.label_size)
      self.num_possible_classes = self.model.components.mapping\
                                  .static_kwargs.label_size
    except Exception:
      self.num_possible_classes = 0    
    
  def clmr_init(self):
    # TODO: add start and duration to song_name
    song_name = f"{self.song_name}_{self.start}_{self.duration}_{self.fps}"
    song_preds_path = f"clmr_{song_name}_preds.pt"
    
    if os.path.exists(song_preds_path):
        all_preds = torch.load(song_preds_path)
    else:
        # dependencies: torchaudio_augmentations, simclr
        if not os.path.exists("./clmr_magnatagatune_mlp"):
            import subprocess
            subprocess.run(["wget", "-nc", "https://github.com/Spijkervet/CLMR/releases/download/2.1/clmr_magnatagatune_mlp.zip"])
            subprocess.run(["unzip", "-o", "clmr_magnatagatune_mlp.zip"])
            #!wget -nc https://github.com/Spijkervet/CLMR/releases/download/2.1/clmr_magnatagatune_mlp.zip
            #!unzip -o clmr_magnatagatune_mlp.zip

        ENCODER_CHECKPOINT_PATH = "./clmr_magnatagatune_mlp/clmr_epoch=10000.ckpt"
        FINETUNER_CHECKPOINT_PATH = "./clmr_magnatagatune_mlp/mlp_epoch=34-step=2589.ckpt"
        SAMPLE_RATE = 22050

        from clmr.datasets import get_dataset
        from clmr.data import ContrastiveDataset
        from clmr.evaluation import evaluate
        from clmr.models import SampleCNN
        from clmr.modules import ContrastiveLearning, LinearEvaluation, PlotSpectogramCallback
        from clmr.utils import yaml_config_hook, load_encoder_checkpoint, load_finetuner_checkpoint

        import argparse

        parser = argparse.ArgumentParser(description="CLMR")
        config = yaml_config_hook("./clmr/config/config.yaml")
        for k, v in config.items():
            parser.add_argument(f"--{k}", default=v, type=type(v))
        args = parser.parse_args([])
        args.accelerator = None

        n_classes = 50
        encoder = SampleCNN(
            strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
            supervised=args.supervised,
            out_dim=n_classes,
        )

        # get dimensions of last fully-connected layer
        n_features = encoder.fc.in_features

        # load the enoder weights from a CLMR checkpoint
        # set the last fc layer to the Identity function, since we attach the
        # fine-tune head seperately

        state_dict = load_encoder_checkpoint(ENCODER_CHECKPOINT_PATH)
        encoder.load_state_dict(state_dict)
        encoder.fc = torch.nn.Identity() #Identity()

        args.finetuner_mlp = True
        module = LinearEvaluation(
            args,
            encoder,
            hidden_dim=n_features,
            output_dim=n_classes,
        )
        state_dict = load_finetuner_checkpoint(FINETUNER_CHECKPOINT_PATH)
        module.model.load_state_dict(state_dict)


        clmr_model = torch.nn.Sequential(encoder,
                                              module.model,
                                              torch.nn.Sigmoid()
                                             ).to(self.device)
        # resample audio
        print("SR: ", self.sr)
        audio = self.wav
        if self.sr != SAMPLE_RATE:
            print("New SR: ", SAMPLE_RATE)
            audio = librosa.resample(audio, self.sr, SAMPLE_RATE)
        audio = torch.from_numpy(audio)
        aud_len = args.audio_length

        # dataset that splits into overlapping sets of 2.7s
        class PieceDataset(torch.utils.data.Dataset):
            def __init__(self, audio, aud_len, step):
                self.audio = audio
                self.aud_len = aud_len
                self.step = step
                # pad audio by (aud_len - step) such that at each step/frame we now include the new data points
                pad_size = aud_len - step
                self.padded_audio = torch.cat([torch.zeros(pad_size), audio, torch.zeros(aud_len - pad_size)])

            def __len__(self):
                return int(np.ceil(len(self.audio) / self.step))

            def __getitem__(self, i):
                i *= self.step
                if i + self.aud_len > len(self.padded_audio):
                    raise ValueError("Step and or aud_len do not fit the length of the input wav")
                    # i = len(self.audio) - self.aud_len
                return self.padded_audio[i: i + self.aud_len].unsqueeze(0)

        sr = self.sr
        fps = self.fps

        #frame_duration = int(sr / fps - (sr / fps % 64))
        frame_duration = self.frame_duration
        num_frames = self.num_frames#np.ceil(len(audio) / frame_duration)
        print("Frame duration: ", frame_duration)
        print("num frames: ", num_frames)

        # make preds
        #ds = torch.utils.data.TensorDataset(splits)
        ds = PieceDataset(audio, aud_len, step=frame_duration)
        dl = torch.utils.data.DataLoader(ds, batch_size=128, pin_memory=True, num_workers=4, shuffle=False)
        all_preds = []
        for batch in tqdm(dl):
            batch = batch.to(self.device)
            with torch.no_grad():
                preds = clmr_model(batch)
            all_preds.append(preds)
        all_preds = torch.cat(all_preds, dim=0).cpu()
        print(all_preds.shape)

        torch.save(all_preds, song_preds_path)

        #import seaborn as sns
        #import pandas as pd
        #import matplotlib.pyplot as plt
        #df = pd.DataFrame(all_preds, columns=self.tags_magnatagatune).transpose().astype(float)
        #plt.figure(figsize=(14, 14), dpi=300)
        #sns.heatmap(df, cmap="mako")
        #plt.savefig("heatmap.png")
       
    self.tags_magnatagatune = ['guitar', 'classical', 'slow', 'techno', 'strings', 'drums', 'electronic', 'rock', 'fast', 'piano', 'ambient', 'beat', 'violin', 'vocal', 'synth', 'female', 'indian', 'opera', 'male', 'singing', 'vocals', 'no vocals', 'harpsichord', 'loud', 'quiet', 'flute', 'woman', 'male vocal', 'no vocal', 'pop', 'soft', 'sitar', 'solo', 'man', 'classic', 'choir', 'voice', 'new age', 'dance', 'male voice', 'female vocal', 'beats', 'harp', 'cello', 'no voice', 'weird', 'country', 'metal', 'female voice', 'choral']
    torch.save(self.tags_magnatagatune, "clmr_tags.pt")
    
    # Calc repr vector per class (except no_vocal, no_voice)
    # TODO: exclude some vectors or only include the top performing/interesting ones
    
    style = self.style.split("/")[-1].split(".")[0].replace(".pkl", "").replace(".", "")
    latent_path = f"magnatagatune_{style}_latents.pt"
    if os.path.exists(latent_path):
        latents = torch.load(latent_path)
    else:
        sys.path.append("../StyleCLIP_modular")
        from style_clip import Imagine
        imagine = Imagine(
                save_progress=False,
                open_folder=False,
                save_video=False,
                opt_all_layers=1,
                lr_schedule=1,
                noise_opt=0,
                epochs=1,
                iterations=1000,
                batch_size=32,
                style=self.style,
                model_type=self.model_type,
                verbose=0,
        )
        latents = []
        for tag in tqdm(self.tags_magnatagatune, position=0):
            tqdm.write(tag)
            imagine.set_clip_encoding(text=tag)
            # train
            imagine()
            # save trained results
            w_opt = imagine.model.model.latents.detach().cpu()
            latents.append(w_opt)
            # reset
            imagine.reset()
        latents = torch.cat(latents, dim=0)
        torch.save(latents, latent_path)
    print("Latents shape: ", latents.shape)  # should be (50, 18, 512) if opt_all_layers 
    
    # create noise vectors! multiply each pred by the latents and average the result
    # clmr preds shape is (len_song, 50)
    if self.clmr_softmax:
        all_preds = torch.softmax(all_preds / self.clmr_softmax_t, dim=-1)
    else:
        # make normed distr over time
        all_preds = (all_preds - all_preds.mean(dim=0, keepdims=True)) / all_preds.std(dim=0, keepdims=True)
        all_preds = (all_preds.clamp(-1.5, 1.5) + 1.5) / 3
        pass
        # make distr per time_step
        #all_preds /= all_preds.sum(dim=-1, keepdims=True)
        
    
    #smooth_range = self.fps // 2  #self.clmr_smooth_range
    # smooth forward in time
    #smoothed = torch.stack([torch.mean(all_preds[i: i + smooth_range], dim=0) for i in range(len(all_preds - smooth_range))])
    # smooth backwards in time
    #smoothed = torch.stack([torch.mean(all_preds[max(i - smooth_range, 0): i + 1], dim=0) for i in range(len(all_preds))])
    # smooth using ema
    
    spec_norm = self.spec_norm_class
    # TODO: use spectral norm to influence how much change is allowed
    
    ema = self.clmr_ema
    ema_value = all_preds[0]
    smoothed = []
    for i in range(len(all_preds)):
        new_val = all_preds[i] * (1 - ema) * spec_norm[i] + ema_value * ema * (1 - spec_norm[i])
        smoothed.append(new_val)
    smoothed = torch.stack(smoothed)
    
    # TODO: Find out decision thresholds for each category by evaluating classifier on whole magnatagatune dataset!
    # TODO: filter out stuff like guitar/piano, no_vocal etc (or only include some good ones)
    include_list = ["techno", "electronic", "classic", "classical", "slow", "fast", "loud", "soft", "quiet", "drums", "ambient"]
    mask = [tag in include_list for tag in self.tags_magnatagatune]
    #latents = latents[mask]
    #smoothed = smoothed[:, mask]
    
    noise = torch.stack([(pred.view(len(latents), 1, 1) * latents).sum(dim=0) for pred in smoothed]).numpy()
    
    # store to disk
    noise = self.store_latents(noise, self.latent_folder, flush=1)
    
    #self.input_shape = latents[0].shape    


  def extract_lyrics_meaning(self, lyrics_path):
    # lyrics sample:
    """
    1
    00:00:02,633 --> 00:00:05,532
    <font color="#FFFFFF"><i>THREE, TWO, ONE.</i></font>

    2
    00:00:05,599 --> 00:00:08,000
    ♪♪ OBIE TRICE
    REAL NAME NO GIMMICKS... ♪
    """
    sys.path.append("../StyleCLIP_modular")

    
    def time_str_to_seconds(time_str):
        hrs_str = time_str.split(":")[0]
        min_str = time_str.split(":")[1]
        seconds_str = time_str.split(":")[2].split(",")[0]
        ms_str = time_str.split(":")[2].split(",")[1]
        return int(hrs_str) * 360 + int(min_str) * 60 + int(seconds_str) + int(ms_str) / 1000
    
    # read file
    with open(lyrics_path, "r+") as f:
        lines = [l for l in f]
    # format and extract lines
    #print(lines[:10])
    texts = []
    start_times = []
    end_times = []
    while len(lines) > 2:
        if lines[0] == "\n":
            del lines[0]
            continue
        #print("next lines")
        #print(lines[0].strip("\n"))
        #print(lines[1].strip("\n"))
        #print(lines[2].strip("\n"))
        count = 4
        # read times
        start_time_str = lines[1].split(" ")[0]
        end_time_str = lines[1].split(" ")[-1]
        # convert to seconds
        start_time = time_str_to_seconds(start_time_str)
        end_time = time_str_to_seconds(end_time_str)

        # read text
        text = lines[2].strip("♪.\n")
        if len(lines) > 3 and lines[3] != "\n":
            text += " " + lines[3].strip("♪.\n")
            count += 1
            if lines[4] != "\n":
                text += " " + lines[4].strip("♪.\n")
                count += 1
                if lines[5] != "\n":
                    text += " " + lines[5].strip("♪.\n")
                    count += 1
        # remove formatting commands
        while text.find("<") != -1:
            start = text.find("<")
            end = text.find(">")
            text = text[:start] + text[end + 1:]
        # remove spaces and turn to lower-case
        text = text.strip("♪ .\n").lower()

        # save
        texts.append(text)
        start_times.append(start_time)
        end_times.append(end_time)

        # delete lines to move on in file
        del lines[:count]

    #print(texts[:5])
    #print(start_times[:5])
    #print(end_times[:5])
    
    # for each phrase, add as many previous words as long as it still fits into the context length
    phrases = texts
    if self.concat_phrases:
        from style_clip.clip import tokenize

        context_length = 77 #imagine.perceptor.context_length

        phrases = []
        current_phrase = ""
        for phrase in texts:
            current_phrase += " \n " + phrase
            # while the sentence is too long, eliminate the first word. subtract 2 from context length for start and end token
            is_too_long = True
            while is_too_long:
                try:
                    tokens = tokenize(current_phrase, context_length=context_length)
                    is_too_long = False
                except RuntimeError:
                    # tokenizer throws RuntimeError if text is too long
                    current_phrase = " ".join(current_phrase.split(" ")[1:])
            phrases.append(current_phrase)
    
    # calc some stats about the song
    frame_duration = self.frame_duration
    num_frames = self.num_frames#np.ceil(len(self.wav) / frame_duration)
    # set path to save/load from
    latent_folder = "lyric_latents"
    lyrics_name = lyrics_path.split("/")[-1].split(".")[0].replace(" ", "_").replace(")", "").replace("(", "")
    os.makedirs(latent_folder, exist_ok=True)
    if self.clip_opt_kwargs is None:
        self.clip_opt_kwargs = {}
    if self.model_type == "stylegan":
        style = self.style.split("/")[-1].split(".")[0].replace(".pkl", "").replace(".", "")
    elif self.model_type == "vqgan":
        kwargs_str = "_".join([f"{key}{self.clip_opt_kwargs[key]}" for key in self.clip_opt_kwargs])
        style = f"vqgan{self.width}x{self.height}" + kwargs_str
    iterations = self.lyrics_iterations
    latents_path = f"{latent_folder}/{lyrics_name}_lyrics_{style}_latents_it{iterations}{'_concatphrases' if self.concat_phrases else ''}{'' if self.reset_latents_after_phrase else '_noLatReset'}"
    os.makedirs(latents_path, exist_ok=True)
    print("Latents at: ", latents_path)
    from style_clip import Imagine, create_text_path
    if self.clip_opt_kwargs is None:
        self.clip_opt_kwargs = {}

    imagine = Imagine(
            save_progress=False,
            open_folder=False,
            save_video=False,
            opt_all_layers=1,
            lr_schedule=1,
            noise_opt=0,
            epochs=1,
            iterations=iterations,
            style=self.style,
            model_type=self.model_type,
            verbose=0,
            sideX=self.width,
            sideY=self.height,
            **self.clip_opt_kwargs,
    )

    # calc latents for each phrase, load the ones that are already calculated
    latents = {"song_start_latent": imagine.model.model.latents.detach().cpu()}
    for phrase in tqdm(phrases, position=0):
        phrase_reformat = create_text_path(text=phrase, context_length=77)
        phrase_path = os.path.join(latents_path, phrase_reformat + ".pt")
        if os.path.exists(phrase_path):
            latent = torch.load(phrase_path)
        else:
            tqdm.write(phrase)
            imagine.set_clip_encoding(text=phrase)
            # train
            imagine()
            # save trained results
            latent = imagine.model.model.latents.detach().cpu()
            # save to disk too
            torch.save(latent, phrase_path)
        latents[phrase] = latent
        # reset
        if self.reset_latents_after_phrase:
            imagine.reset()

    # smooth spectral norm for smoother transitions
    spec_norm = self.spec_norm_class
    ema_val = 0.75
    ema_spec_norm = []
    val = spec_norm[0]
    for amp in spec_norm:
        val = amp * (1 - ema_val) + ema_val * val
        ema_spec_norm.append(val)
        
    # calc minmax vals for sigmoid
    temp_vec = torch.arange(0, 10000) / 10000
    temp_vec_sig = torch.sigmoid((temp_vec - 0.5) * self.lyrics_sigmoid_t)
    sig_min, sig_max = temp_vec_sig.min(), temp_vec_sig.max()
    
    # calc mid times
    mid_times = [(end_times[i] * 3 + start_times[i] * 1) / 4 for i in range(len(end_times))]
    
    # assign latents to frames
    start_latent = latents[phrases[0]]
    current_latent = start_latent
    next_latent = latents[phrases[0]]
    next_mid_time = mid_times[0]
    mid_time_to_mid_time = mid_times[0]
    steps_to_next = int(np.ceil(mid_time_to_mid_time * self.fps))
    current_step = 0
    ampl_sum = sum(ema_spec_norm[: steps_to_next])
    ampl_cumsum = 0
    noise = []
    print("Latent shape: ", latents["song_start_latent"].shape)
    print("Num frames. ", num_frames)
    print("First num steps to next mid of phrase: ", steps_to_next)
    print("Ampl sum start: ", ampl_sum)
    fracs_before_sig = []
    fracs = []
    
    for i in tqdm(range(int(num_frames)), desc="Generating base lyric latents..."):
        current_time = i / self.fps
        #fraction_to_next = (next_mid_time - current_time) / mid_time_to_mid_time
        fraction_to_next = 1 - (current_step / steps_to_next)
        current_step += 1
        
        fracs_before_sig.append(fraction_to_next)
        
        if self.ampl_influences_speed:
            ampl_cumsum += ema_spec_norm[i]
            fraction_to_next = 1 - (ampl_cumsum / ampl_sum)
            
        # instead of linear make it a sigmoid such that the space around the text latents is held for longer
        if self.lyrics_sigmoid_transition:
            # apply sigmoid
            fraction_to_next = torch.sigmoid((torch.tensor(fraction_to_next) - 0.5) * self.lyrics_sigmoid_t).item()
            # norm sigmoid to span between 0 and 1
            fraction_to_next = (fraction_to_next - sig_min) / (sig_max - sig_min)
            fracs.append(fraction_to_next)
            
        # use spherical interpolation
        interpolated_latent = slerp(current_latent, next_latent, 1 - fraction_to_next)
        
        noise.append(interpolated_latent.squeeze().numpy())
        noise = self.store_latents(noise, self.latent_folder)
        
        if len(mid_times) > 1:
            if current_time > mid_times[0]:
                current_text = phrases[0]
                current_latent = latents[current_text]
                
                next_mid_time = mid_times[1]
                mid_time_to_mid_time = next_mid_time - mid_times[0]
                
                next_latent = latents[phrases[1]]
        
                steps_to_next = int(np.ceil(mid_time_to_mid_time * self.fps))
                current_step = 0
                ampl_sum = sum(ema_spec_norm[i: i + steps_to_next])
                ampl_cumsum = 0
                del mid_times[0]
                del phrases[0]
        elif current_time > mid_times[0]:
            # for last image, just show constant frame
            current_text = phrases[0]
            current_latent = latents[current_text]

            next_mid_time = mid_times[0]
            mid_time_to_mid_time = 1

            next_latent = current_latent
    
    self.store_latents(noise, self.latent_folder, flush=1)
    self.lyric_transition_distances = fracs
    self.lyric_transition_distances_raw = fracs_before_sig


  def store_latents(self, latent_list, latent_folder, flush=False):
    if flush or len(latent_list) > self.max_latent_in_mem:
        latent_names_so_far = [f for f in os.listdir(latent_folder) if f.endswith(".npy")]
        max_idx = max([int(name.split(".")[0]) for name in latent_names_so_far]) if len(latent_names_so_far) > 0 else -1
        for l, i in zip(latent_list, range(max_idx + 1, max_idx + 1 + len(latent_list))):
            np.save(os.path.join(latent_folder, f"{i}.npy"), l)
        return []
    else:
        return latent_list
        

  def load_specs(self):
    '''Load normalized spectrograms and chromagram'''

    start = self.start
    duration = self.duration
    fps = self.fps
    input_shape = self.input_shape
    pulse_percussive = self.pulse_percussive
    pulse_harmonic = self.pulse_harmonic
    motion_percussive = self.motion_percussive
    motion_harmonic = self.motion_harmonic

    # Load audio signal data
    wav, sr = librosa.load(self.song, offset=start, duration=duration)
    wav_motion = wav_pulse = wav_class = wav
    sr_motion = sr_pulse = sr_class = sr

    # If pulse_percussive != pulse_harmonic
    # or motion_percussive != motion_harmonic,
    # decompose harmonic and percussive signals and assign accordingly
    aud_unassigned = (not self.pulse_audio) or (not self.motion_audio)
    pulse_bools_equal = pulse_percussive == pulse_harmonic
    motion_bools_equal = motion_percussive == motion_harmonic

    if aud_unassigned and not all([pulse_bools_equal, motion_bools_equal]):
        wav_harm, wav_perc = librosa.effects.hpss(wav)
        wav_list = [wav, wav_harm, wav_perc]

        pulse_bools = [pulse_bools_equal, pulse_harmonic, pulse_percussive]
        wav_pulse = wav_list[pulse_bools.index(max(pulse_bools))]

        motion_bools = [motion_bools_equal, motion_harmonic, motion_percussive]
        wav_motion = wav_list[motion_bools.index(max(motion_bools))]

    # Load audio signal data for Pulse, Motion, and Class if provided
    if self.pulse_audio:
        wav_pulse, sr_pulse = librosa.load(self.pulse_audio, offset=start, 
                                         duration=duration)
    if self.motion_audio:
        wav_motion, sr_motion = librosa.load(self.motion_audio, offset=start, 
                                           duration=duration)
    if self.class_audio:
        wav_class, sr_class = librosa.load(self.class_audio, offset=start,
                                         duration=duration)
    
    # Calculate frame duration (i.e. samples per frame)
    frame_duration = int(sr / fps - (sr / fps % 64))

    # Generate normalized spectrograms for Pulse, Motion and Class
    self.spec_norm_pulse = get_spec_norm(wav_pulse, sr_pulse, 
                                         512, frame_duration)
    self.spec_norm_motion = get_spec_norm(wav_motion, sr_motion,
                                          512, frame_duration)
    self.spec_norm_class = get_spec_norm(wav_class, sr_class, 
                                        512, frame_duration)
    self.num_frames = len(self.spec_norm_class)
    
    if self.use_all_layers:
        self.spec_norm_pulse = np.stack(self.spec_norm_pulse.copy() for _ in range(self.input_shape[0]))
        self.spec_norm_motion = np.stack(self.spec_norm_motion.copy() for _ in range(self.input_shape[0]))
        #self.spec_norm_class = np.stack(self.spec_norm_class.copy() for _ in range(self.input_shape[0]))
        self.spec_norm_pulse = np.moveaxis(self.spec_norm_pulse, 0, 1)
        self.spec_norm_motion = np.moveaxis(self.spec_norm_motion, 0, 1)
        #self.spec_norm_class = np.moveaxis(self.spec_norm_class, 0, 1)
        
    # Generate chromagram from Class audio
    chrom_class = librosa.feature.chroma_cqt(y=wav_class, sr=sr,
                                             hop_length=frame_duration)
    # Sort pitches based on "dominance"
    chrom_class_norm = chrom_class / chrom_class.sum(axis=0, keepdims=1)
    chrom_class_sum = np.sum(chrom_class_norm,axis=1)
    pitches_sorted = np.argsort(chrom_class_sum)[::-1]

    # Assign attributes to be used for vector generation
    self.wav, self.sr, self.frame_duration = wav, sr, frame_duration
    self.chrom_class, self.pitches_sorted = chrom_class, pitches_sorted
    
    print("Cluster pitches: ", self.cluster_pitches)
    if self.cluster_pitches is not None:
        def create_spectral_norm_bins(wav, sr, frame_duration, gram_type, num_bands=18):
            if gram_type == "spectral":
                # custom spectral norms for multiple bands
                # get spectrogram
                mel_spect = librosa.feature.melspectrogram(y=wav, sr=sr, hop_length=frame_duration, n_mels=512)
                # use only the centrail main bands that include 90% quantile
                threshold = 0.05  # smaller == more bands included
                mean_per_band = mel_spect.mean(axis=1)
                mean_per_band /= mean_per_band.sum()
                upper_mask = np.cumsum(mean_per_band) < (1 - threshold)
                lower_mask = np.cumsum(mean_per_band) > threshold
                mask = upper_mask & lower_mask
                main_spectrogram = mel_spect[mask]
                #print(main_spectrogram.shape)
                # find out how to split main spectrogram into 18 roughly equally-sized partitions
                rnge = np.arange(len(main_spectrogram))
                split_idcs = np.array_split(rnge, num_bands)
                # apply found idcs
                sub_spectros = [main_spectrogram[idcs] for idcs in split_idcs]
                # get spectral norms for each
                sub_spec_norms = np.array([spec.mean(axis=0) for spec in sub_spectros])
                spec_norms = sub_spec_norms
            elif gram_type == "chroma_cqt":
                spec_norms = librosa.feature.chroma_cqt(y=wav, sr=sr, C=None, 
                                                        hop_length=frame_duration, fmin=None, 
                                                        threshold=0.0, tuning=None, n_chroma=num_bands,
                                                        n_octaves=7, window=None, 
                                                        bins_per_octave=36, cqt_mode='full')
            elif gram_type == "chroma_stft":
                spec_norms = librosa.feature.chroma_stft(y=wav, sr=sr, 
                                                         hop_length=frame_duration, tuning=None,
                                                         n_chroma=num_bands)
            # minmax per time-series
            spec_norms = minmax(spec_norms, axis=1)
            # bring into shape [num_time_steps, num_bands]     
            spec_norms = np.moveaxis(spec_norms, 0, 1)
            return spec_norms

        num_bands = 18
        self.spec_norm_pulse = create_spectral_norm_bins(wav_pulse, sr_pulse, 
                                     frame_duration, self.cluster_pitches, num_bands=num_bands)
        self.spec_norm_motion = create_spectral_norm_bins(wav_motion, sr_motion,
                                              frame_duration, self.cluster_pitches, num_bands=num_bands)
  def generate_vectors(self):
    '''Generates noise and class vectors as inputs for each frame'''
    
    # generate vectors for no-lyrics case
    if not self.use_clmr and not self.visualize_lyrics:       
        img_per_s = 1 / 60
        song_seconds = self.num_frames // self.fps
        num_imgs = max(int(song_seconds * img_per_s), 1)
        z = torch.from_numpy(np.random.RandomState(None).randn(num_imgs, self.model.z_dim)).to(self.device)
        w_samples = self.model.mapping(z,  None, truncation_psi=self.truncation_psi).cpu().numpy()
        if not self.use_all_layers:
            w_samples = w_samples[:, 0, :]
        if num_imgs == 1:
            noise = [w_samples[0].copy() for _ in range(self.num_frames)]
        else:
            noise = full_frame_interpolation(w_samples, self.num_frames // num_imgs, self.num_frames)
        noise = self.store_latents(noise, self.latent_folder, flush=1)

    
    
    # dataloader to iterate through latents
    ds = LatentsDataset(self.latent_folder)
    assert len(ds) > 0
    dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, pin_memory=True, shuffle=False, num_workers=2)
    
    if self.no_beat:
        # store latents in new folder
        os.rename(self.latent_folder, self.beat_latent_folder)
        
        #for i, batch in enumerate(tqdm(dl)):
        #batch = batch.numpy()
        #for latent in batch:
        #    # Store latents
        #    noise.append(latent)
        #    noise = self.store_latents(noise, self.beat_latent_folder)
        return
    
    PULSE_SMOOTH = 0.75
    MOTION_SMOOTH = 0.75
    classes = self.classes
    class_shuffle_seconds = self.class_shuffle_seconds or [0]
    class_shuffle_strength = round(self.class_shuffle_strength * 12)
    fps = self.fps
    class_smooth_frames = self.class_smooth_seconds * fps
    motion_react = self.motion_react * 20 / fps

    # Get number of noise vectors to initialize (based on speed_fpm)
    minutes = librosa.get_duration(self.wav, self.sr)
    num_init_noise = round(minutes / 60 * self.speed_fpm)
    num_total_frames = len(self.spec_norm_class)
    
    # if we sample in the model latent space, find the std per feature
    shape = self.input_shape.tolist()
    if self.model_type == "vqgan":
        #def sample_func(shape):
        #    return torch.zeros(shape).normal_(0., 4.).numpy()
        if self.clip_opt_kwargs is not None and "latent_type" in self.clip_opt_kwargs and self.clip_opt_kwargs["latent_type"] == "code_sampling":
            def sample_func(shape):
                vqgan = self.model.model
                vocab_size = vqgan.quantize.n_e
                return torch.zeros(shape).float().normal_(0, 1).cpu().numpy()
                
                code_b = torch.randint(0, vocab_size, (self.toksX * self.toksY,)).cpu().numpy()
                #encoded = vqgan.quantize.get_codebook_entry(code_b, None)
                #encoded = encoded.permute(1, 0).reshape(shape).cpu().numpy()
                return code_b
        else:
            def sample_func(shape):
                vqgan = self.model.model
                vocab_size = vqgan.quantize.n_e
                code_b = torch.randint(0, vocab_size, (self.toksX * self.toksY,), device=self.device)
                encoded = vqgan.quantize.get_codebook_entry(code_b, None)
                encoded = encoded.permute(1, 0).reshape(shape).cpu().numpy()
                return encoded
        
        w_samples = np.array([sample_func(shape) for _ in range(1000)])
        z_dim_std = w_samples.std(axis=0)
        z_dim_mean = w_samples.mean(axis=0)
    elif self.use_all_layers:
        def sample_func(shape):
            z = torch.tensor([truncnorm.rvs(-2, 2, size=512)], device=self.device)
            with torch.no_grad():
                w_samples = self.model.mapping(z, None, truncation_psi=self.truncation_psi).cpu()
            return w_samples.squeeze().numpy()
        #z = torch.tensor([truncnorm.rvs(-2, 2, size=512) for _ in range(10000)], device=self.device)
        #z = torch.from_numpy(np.random.RandomState(1).randn(10000, 512)).to(self.device)
        #with torch.no_grad():
        #    w_samples = self.model.mapping(z, None, truncation_psi=self.truncation_psi).cpu()
        
        w_samples = np.array([sample_func(shape) for _ in range(10000)])
        all_std = w_samples.std(axis=0)
        mean_layer_std = all_std.mean(axis=1)[0]
        z_dim_std = all_std[0]
        z_dim_mean = w_samples.mean(0)
        print("Shapes: ", w_samples.shape, all_std.shape, z_dim_std.shape)
        print(mean_layer_std)
        print(z_dim_std[:10])
        
        #z_dim_std = z_dim_std.repeat(self.input_shape[0], 1).numpy()
        z_dim_std = all_std
    else:
        def sample_func(shape):
                return truncnorm.rvs(-2, 2, size=shape).astype(np.float32)
        z_dim_std = np.ones(shape)
        z_dim_mean = np.ones(shape)
    print("Latent std shape: ", z_dim_std.shape)
    print("Latent std first entries: ", z_dim_std.reshape(-1)[:10])
         
    # Init random latents if not using clmr or lyrics to do so
    # If num_init_noise < 2, simply initialize the same 
    # noise vector for all frames 
    print("Input shape: ", self.input_shape)            
      

    num_latents = self.input_shape.prod()
        
    # Initialize "base" vectors based on Pulse/Motion Reactivity values
    pulse_base = np.ones(shape) * self.pulse_react
    motion_base = np.ones(shape) * motion_react
    # Randomly initialize "update directions" of motion vectors
    motion_signs = np.array([random.choice([1, -1]) for _ in range(num_latents)]).reshape(shape)
    # Randomly initialize factors based on motion_randomness (0.5 by default)
    rand_factors = np.array([random.choice([1, 1 - self.motion_randomness]) for _ in range(num_latents)]).reshape(shape)

    if self.use_song_latent_std:
        # calculate std of latents for each latent dimension dependent on their std within the latent vectors initialized for this song
        welford_alg = Welford()
        for i, batch in enumerate(tqdm(dl)):
            batch = batch.numpy()
            for latent in batch:
                welford_alg.update(latent)
        z_dim_mean = welford_alg.mean
        z_dim_std = welford_alg.std
        
        print(z_dim_mean)
        print(z_dim_std)
        

        print(z_dim_mean.min(), z_dim_mean.max(), z_dim_mean.mean(), z_dim_mean.std())
        print(z_dim_std.min(), z_dim_std.max(), z_dim_std.mean(), z_dim_std.std())
        quit()
    
    # Initialize running exponential averages
    pulse_noise = 0
    motion_noise = 0
    motion_noise_sum = 0
    noise = []
    
    scaled_sigmoid = lambda x: 1 / (1 + np.exp(-x - z_dim_mean) * z_dim_std)
    self.motion_range = 5
        
    # UPDATE NOISE # 
    for batch_idx, batch in enumerate(tqdm(dl)):
        batch = batch.numpy()
        for single_idx, latent in enumerate(batch):
            i = batch_idx * self.batch_size + single_idx
            # Re-initialize randomness factors every 4 seconds
            if i % round(fps * 4) == 0:
                print("RANDOMIZE RAND FACTORS")
                rand_factors = np.array([random.choice([1, 1 - self.motion_randomness]) for _ in range(num_latents)]).reshape(shape)

            # Generate incremental update vectors for Pulse and Motion
            spec_pulse = self.spec_norm_pulse[i]
            spec_mot = self.spec_norm_motion[i]
            
            if self.use_all_layers:
                spec_pulse = spec_pulse.reshape(self.input_shape[0], 1) 
                spec_mot = spec_mot.reshape(18, 1)  # * scipy.special.softmax(z_dim_std, axis=-1)
            # Create update vectors for pulse and motion
            if self.use_old_beat:
                pulse_noise_add = pulse_base * spec_pulse # * z_dim_std
                motion_noise_add = motion_base * spec_mot * motion_signs * rand_factors # * z_dim_std
            else:
                pulse_noise_add = pulse_base * spec_pulse * z_dim_std
                motion_noise_add = motion_base * spec_mot * motion_signs * rand_factors * z_dim_std
                
                #pulse_bef_sig = pulse_base
                #pulse_noise_add = scaled_sigmoid(pulse_bef_sig) * spec_pulse
                #motion_bef_sig = motion_base
                #motion_noise_add = scaled_sigmoid(motion_bef_sig) * spec_mot * motion_signs * rand_factors
                
            # Smooth each update vector using a weighted average of
            # itself and the previous vector
            pulse_noise = pulse_noise * PULSE_SMOOTH + pulse_noise_add * (1 - PULSE_SMOOTH)
            motion_noise = motion_noise * MOTION_SMOOTH + motion_noise_add * (1 - MOTION_SMOOTH)

            print(spec_pulse, spec_mot)
            
            # Update current noise vector by adding current Pulse vector and 
            # a cumulative sum of Motion vectors
            motion_noise_sum += motion_noise
            latent += pulse_noise + motion_noise_sum #* np.abs(noise[i]))
            
            # Store latents
            noise.append(latent)
            noise = self.store_latents(noise, self.beat_latent_folder)
            
            # update motion directions
            if self.use_old_beat:
                thresh_pos = 2 - motion_signs
                thresh_neg = -1 * thresh_pos + motion_signs
                next_latent = latent
            else:
                thresh_pos = latent + z_dim_std * self.motion_range
                thresh_neg = latent + z_dim_std * self.motion_range
                next_latent = latent + motion_noise_sum + pulse_noise
            # For each current value in noise vector, change direction if absolute 
            # value +/- motion_react is larger than the threshold
            motion_signs[next_latent < thresh_neg] = 1
            motion_signs[next_latent >= thresh_pos] = -1
                    
    #self.noise = noise 
    noise = self.store_latents(noise, self.beat_latent_folder, flush=1)

  def setup_effects(self):
    '''Initializes effects to be applied to each frame'''

    self.custom_effects = self.custom_effects or []
    start = self.start
    duration = self.duration

    # Initialize pre-made Contrast effect 
    if all(var is None for var in [self.contrast_audio, 
                                  self.contrast_strength,
                                  self.contrast_percussive]):
      pass
    else:
      self.contrast_audio = self.contrast_audio or self.song
      self.contrast_strength = self.contrast_strength or 0.5
      self.contrast_percussive = self.contrast_percussive or True

      contrast = EffectsGenerator(audio = self.contrast_audio, 
                                  func = contrast_effect, 
                                  strength = self.contrast_strength, 
                                  percussive = self.contrast_percussive)
      self.custom_effects.append(contrast)

    # Initialize pre-made Flash effect
    if all(var is None for var in [self.flash_audio, 
                                  self.flash_strength,
                                  self.flash_percussive]):
      pass
    else:
      self.flash_audio = self.flash_audio or self.song
      self.flash_strength = self.flash_strength or 0.5
      self.flash_percussive = self.flash_percussive or True
  
      flash = EffectsGenerator(audio = self.flash_audio, 
                                  func = flash_effect, 
                                  strength = self.flash_strength, 
                                  percussive = self.flash_percussive)
      self.custom_effects.append(flash)

    # Initialize Custom effects
    for effect in self.custom_effects:
      effect.audio = effect.audio or self.song
      effect.render_audio(start=start, 
                          duration = duration, 
                          n_mels = self.input_shape, 
                          hop_length = self.frame_duration)

  def generate_frames(self):
    '''Generate GAN output for each frame of video'''

    file_name = self.file_name
    resolution = self.resolution
    num_frame_batches = int(self.num_frames / self.batch_size)
    if self.use_tf:
        Gs_syn_kwargs = {'output_transform': {'func': self.convert_images_to_uint8, 
                                          'nchw_to_nhwc': True},
                    'randomize_noise': False,
                    'minibatch_size': batch_size}
    else:
        Gs_syn_kwargs = {'noise_mode': 'const'} # random, const, None

    # Set-up temporary frame directory
    self.frames_dir = file_name.split('.mp4')[0] + '_frames'
    if os.path.exists(self.frames_dir):
        shutil.rmtree(self.frames_dir)
    os.makedirs(self.frames_dir)

    # create dataloader
    ds = LatentsDataset(self.beat_latent_folder)
 
    dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, pin_memory=True, shuffle=False, num_workers=2)
    torch.backends.cudnn.benchmark = True

    final_images = []
    file_names = []

    # Generate frames
    for i, noise_batch in enumerate(tqdm(dl, position=0, desc="Generating frames")):
        # If style is a custom function, pass batches to the function
        if callable(self.style): 
            image_batch = self.style(noise_batch=noise_batch, 
                                     class_batch=class_batch)
        # Otherwise, generate frames with StyleGAN(2)
        else:
            if self.model_type == "vqgan":
                #if len(noise_batch.shape) < len(self.input_shape.tolist()) + 1:
                #    noise_batch = noise_batch.unsqueeze(0)
                noise_batch = noise_batch.to(self.device, non_blocking=True)
                with torch.no_grad():
                    self.model.latents = noise_batch.float()
                    image_batch = self.model(return_loss=False)
                image_batch = (image_batch.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                if len(image_batch.shape) > 4:
                    image_batch = image_batch.squeeze(0)
            elif self.use_tf:
                noise_batch = noise_batch.numpy()
                class_batch = class_batch.numpy()
                w_batch = self.model.components.mapping.run(noise_batch, np.tile(class_batch, (batch_size, 1)))
                image_batch = self.model.components.synthesis.run(w_batch, **Gs_syn_kwargs)
                image_batch = np.array(image_batch)
            else:
                noise_batch = noise_batch.to(self.device, non_blocking=True)
                with torch.no_grad():
                    if self.use_all_layers:
                        w_batch = noise_batch
                    else:
                        w_batch = self.model.mapping(noise_batch, None, truncation_psi=self.truncation_psi)
                    image_batch = self.model.synthesis(w_batch, **Gs_syn_kwargs)
                image_batch = (image_batch.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze(0).cpu().numpy()

        # For each image in generated batch: apply effects, resize, and save
        for j, array in enumerate(image_batch): 
            image_index = (i * self.batch_size) + j

            # Apply efects
            for effect in self.custom_effects:
                array = effect.apply_effect(array=array, index=image_index)

            # Save. Include leading zeros in file name to keep alphabetical order
            max_frame_index = num_frame_batches * self.batch_size + self.batch_size
            file_name = str(image_index).zfill(len(str(max_frame_index)))
        
            file_names.append(file_name)
            final_images.append(array)
            
            
        if len(final_images) > self.max_frames_in_mem:
            self.store_imgs(file_names, final_images, resolution)
            file_names = []
            final_images = []
    if len(final_images) > 0:
        self.store_imgs(file_names, final_images, resolution)

  def store_imgs(self, file_names, final_images, resolution):
    for file_name, final_image in tqdm(zip(file_names, final_images), position=1, leave=False, desc="Storing frames", total=len(file_names)):
        try:
            with Image.fromarray(final_image, mode='RGB') as final_image_PIL:
                # If resolution is provided, resize
                if resolution:
                    final_image_PIL = final_image_PIL.resize((resolution, resolution))
                final_image_PIL.save(os.path.join(self.frames_dir, file_name + '.jpg'), subsample=0, quality=95)
        except ValueError as e:
            print("An error when saving occurred: ", e)
        

  def hallucinate(self,
                  file_name: str,
                  output_dir: str,
                  output_audio: str = None,
                  fps: int = 43, 
                  resolution: int = None, 
                  start: float = 0, 
                  duration: float = None, 
                  save_frames: bool = False,
                  batch_size: int = 1,
                  speed_fpm: int = 12,
                  pulse_percussive: bool = True,
                  pulse_harmonic: bool = False,
                  pulse_react: float = 0.5,
                  motion_percussive: bool = False,
                  motion_harmonic: bool = True,
                  motion_react: float = 0.5, 
                  motion_randomness: float = 0.5,
                  truncation: float = 1,
                  classes: list = None,
                  dominant_classes_first: bool = False,
                  class_pitch_react: float = 0.5,
                  class_smooth_seconds: int = 1,
                  class_complexity: float = 1, 
                  class_shuffle_seconds: float = None,
                  class_shuffle_strength: float = 0.5,
                  contrast_strength: float = None, 
                  contrast_percussive: bool = None,
                  flash_strength: float = None,
                  flash_percussive: bool = None,
                  custom_effects: list = None,
                  truncation_psi: float = 1.0,
                  max_frames_in_mem: int = 500,
                  no_beat: int = 0,
                  cluster_pitches: str = None,
                  input_shape = None,
                  use_all_layers = 0,
                  use_old_beat = 0,
                  use_song_latent_std = 0,
                  
                  use_clmr=False,
                  clmr_softmax=False,
                  clmr_softmax_t=1.0,
                  clmr_ema=0.9,
                  
                  visualize_lyrics=0,
                  lyrics_path=None,
                  ampl_influences_speed=0,
                  lyrics_sigmoid_transition=0,
                  lyrics_sigmoid_t=8,  # scaling of sigmoid, should be between 5 and 10, the stronger, the quicker the changes between phrases
                  concat_phrases=0,
                  lyrics_iterations=200,
                  reset_latents_after_phrase=1,
                  
                  clip_opt_kwargs=None,
                 ):
    '''Full pipeline of video generation'''

    # Raise exception if speed_fpm > fps * 60
    if speed_fpm > fps * 60:
        sys.exit('speed_fpm must not be greater than fps * 60')
    
    # Raise exception if element of custom_effects is not EffectsGenerator
    if custom_effects:
        if not all(isinstance(effect, EffectsGenerator) for effect in custom_effects):
            sys.exit('Elements of custom_effects must be EffectsGenerator objects')

    # Raise exception of classes is an empty list
    if classes:
        if len(classes) == 0:
            sys.exit('classes must be NoneType or list with length > 0')

    # Raise exception if any of the following parameters are not betwee 0 and 1
    for param in ['motion_randomness', 'truncation','class_shuffle_strength', 
                  'contrast_strength', 'flash_strength']:
        if (locals()[param]) and not (0 <= locals()[param] <= 1):
            sys.exit('{} must be between 0 and 1'.format(param))

    # create dirs and namings
    self.file_name = file_name if file_name[-4:] == '.mp4' else file_name + '.mp4'
    self.file_name = self.file_name.split("/")[-1].replace(" ", "_")
    os.makedirs(output_dir, exist_ok=1)
    self.output_dir = output_dir
    self.latent_folder = os.path.join(self.output_dir, "latents", self.file_name.split(".")[0])
    self.beat_latent_folder = os.path.join(self.output_dir, "beat_latents", self.file_name.split(".")[0])
    os.makedirs(self.latent_folder, exist_ok=1)
    os.makedirs(self.beat_latent_folder, exist_ok=1)
    self.max_latent_in_mem = 500
    
    
    self.resolution = resolution
    self.batch_size = batch_size
    self.speed_fpm = speed_fpm
    self.pulse_react = pulse_react
    self.motion_react = motion_react 
    self.motion_randomness = motion_randomness
    self.truncation = truncation
    self.classes = classes
    self.dominant_classes_first = dominant_classes_first
    self.class_pitch_react = class_pitch_react
    self.class_smooth_seconds = class_smooth_seconds
    self.class_complexity = class_complexity
    self.class_shuffle_seconds = class_shuffle_seconds
    self.class_shuffle_strength = class_shuffle_strength
    self.contrast_strength = contrast_strength
    self.contrast_percussive = contrast_percussive
    self.flash_strength = flash_strength
    self.flash_percussive = flash_percussive
    self.custom_effects = custom_effects 
    self.max_frames_in_mem = max_frames_in_mem
    self.no_beat = no_beat
    self.cluster_pitches = cluster_pitches
    self.use_all_layers = use_all_layers
    self.use_old_beat = use_old_beat
    self.use_song_latent_std = use_song_latent_std
    if self.model_type == "stylegan" and (use_clmr or visualize_lyrics):
        self.use_all_layers = 1
    if input_shape is None:
        if self.model_type == "stylegan":
            if self.use_all_layers:
                input_shape = (18, 512)
            else:
                input_shape = (512,)
        elif self.model_type == "vqgan":
            f = 16
            self.toksX = self.width // f
            self.toksY = self.height // f
            if clip_opt_kwargs is not None and "latent_type" in clip_opt_kwargs and clip_opt_kwargs["latent_type"] == "code_sampling":
                input_shape = (self.toksY * self.toksX, 1024)
            else:
                input_shape = (256, self.toksY, self.toksX)
            
    self.input_shape = torch.tensor(input_shape)

    # stylegan2 params
    self.truncation_psi = truncation_psi
    # clmr params
    self.use_clmr = use_clmr
    self.clmr_softmax = clmr_softmax
    self.clmr_softmax_t = clmr_softmax_t
    self.clmr_ema = clmr_ema
    # lyrics params
    self.visualize_lyrics = visualize_lyrics
    self.ampl_influences_speed = ampl_influences_speed
    self.lyrics_sigmoid_transition = lyrics_sigmoid_transition
    self.lyrics_sigmoid_t = lyrics_sigmoid_t
    self.concat_phrases = concat_phrases
    self.lyrics_iterations = lyrics_iterations
    self.reset_latents_after_phrase = reset_latents_after_phrase
    # clip params
    self.clip_opt_kwargs = clip_opt_kwargs

    # If there are changes in any of the following parameters,
    # re-initialize audio
    cond_list = [(not hasattr(self, 'fps')) or (self.fps != fps),
                 (not hasattr(self, 'start')) or (self.start != start),
                 (not hasattr(self, 'duration')) or (self.duration != duration),
                 (not hasattr(self, 'pulse_percussive')) or \
                 (self.pulse_percussive != pulse_percussive),
                 (not hasattr(self, 'pulse_harmonic')) or \
                 (self.pulse_percussive != pulse_harmonic),
                 (not hasattr(self, 'motion_percussive')) or \
                 (self.motion_percussive != motion_percussive),
                 (not hasattr(self, 'motion_harmonic')) or \
                 (self.motion_percussive != motion_harmonic)]

    if any(cond_list):
        self.fps = fps
        self.start = start
        self.duration = duration 
        self.pulse_percussive = pulse_percussive
        self.pulse_harmonic = pulse_harmonic
        self.motion_percussive = motion_percussive
        self.motion_harmonic = motion_harmonic

        print('Preparing audio...')
        self.load_specs()
        
    try:
        if self.use_clmr:
            # prep clmr
            import_clmr()
            # Make CLMR preds:
            self.clmr_init()

        if visualize_lyrics:
            self.extract_lyrics_meaning(lyrics_path)

        # Initialize img generation nets
        if self.model_type == "stylegan":
            if not self.style_exists:
                print('Preparing style...')
                if not callable(self.style):
                    self.stylegan_init()
                self.style_exists = True
        elif self.model_type == "vqgan":
            self.use_tf = False
            sys.path.append("../StyleCLIP_modular")
            from style_clip.model import VQClip
            kwargs = {key: self.clip_opt_kwargs[key] for key in self.clip_opt_kwargs if key in ["latent_type"]}
            self.model = VQClip(sideX=self.width,
                                sideY=self.height,
                                **kwargs).to(self.device)
        
        # Initialize effects
        print('Loading effects...')
        self.setup_effects()

        # Generate vectors
        print('\n\nDoing math...\n')
        self.generate_vectors()

        # Generate frames
        print('\n\nHallucinating... \n')
        self.generate_frames()

        # Load output audio
        if output_audio:
            wav_output, sr_output = librosa.load(output_audio, offset=start, duration=duration)
        else:
            wav_output, sr_output = self.wav, self.sr

        # Write temporary audio file
        soundfile.write('tmp.wav',wav_output, sr_output)

        # Generate final video
        audio = mpy.AudioFileClip('tmp.wav', fps=self.sr * 2)
        video = mpy.ImageSequenceClip(self.frames_dir, fps=self.sr / self.frame_duration)
        video = video.set_audio(audio)
        video_file_path = os.path.join(self.output_dir, self.file_name)
        video.write_videofile(video_file_path, audio_codec='aac', fps=self.fps)
        # HQ video
        #video.write_videofile(video_file_path.split(".")[0] + ".avi", fps=self.fps, codec="png")

        # Delete temporary audio file
        os.remove('tmp.wav')
    finally:
        # By default, delete temporary frames directory
        if not save_frames and hasattr(self, "frames_dir") and os.path.exists(self.frames_dir): 
          shutil.rmtree(self.frames_dir)
        # Delete temporary latent folders
        if os.path.exists(self.latent_folder):
            shutil.rmtree(self.latent_folder)
        if os.path.exists(self.beat_latent_folder):
            shutil.rmtree(self.beat_latent_folder)


class EffectsGenerator:
  def __init__(self, 
               func, 
               audio: str = None,
               strength: float = 0.5,
               percussive: bool = True):
    self.audio = audio
    self.func = func 
    self.strength = strength
    self.percussive = percussive

    # Raise exception of func does not take in parameters array, 
    # strength, and amplitude
    func_sig = list(inspect.getfullargspec(func))[0]
    for arg in ['array', 'strength', 'amplitude']:
      if arg not in func_sig:
        sys.exit('func must be a function with parameters '\
                 'array, strength, and amplitude')
    

  def render_audio(self, start, duration, n_mels, hop_length):
    '''Prepare normalized spectrogram of audio to be used for effect'''

    # Load spectrogram
    wav, sr = librosa.load(self.audio, offset=start, duration=duration)

    # If percussive = True, decompose harmonic and percussive signals
    if self.percussive: 
      wav = librosa.effects.hpss(wav)[1]

    # Get normalized spectrogram  
    self.spec = get_spec_norm(wav, sr, n_mels=n_mels, hop_length=hop_length)


  def apply_effect(self, array, index):
    '''Apply effect to image (array)'''

    amplitude = self.spec[index]
    return self.func(array=array, strength = self.strength, amplitude=amplitude)
