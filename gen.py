import time
import os

#from main import LucidSonicDream
from lucidsonicdreams.main import LucidSonicDream
#from lucidsonicdreams import LucidSonicDream

styles = {"strangegods": "../stylegan3/strangegods_1620.pkl",
        "janregnar": "../stylegan2-ada-pytorch/jan_regnart_640.pkl", 
          "madziowa": "../stylegan2-ada-pytorch/madziowa_p_800.pkl",
          "microscope": "microscope images", 
          "imagenet": "imagenet", 
          "ffhq": "faces (ffhq config-f)", 
          "visionaryart": "../stylegan2-ada-pytorch/VisionaryArt.pkl", 
          "astromaniacmag": "../stylegan2-ada-pytorch/astromaniacmag_160.pkl",
          "vint_retro_scifi": "../stylegan2-ada-pytorch/vint_retro_scifi_3200_2map.pkl",
          "therealtheory": "../stylegan2-ada-pytorch/therealtheory_540.pkl",
          "floor_plans": "models/floor-plans_stylegan2.pkl"}

songs = ["songs/henne_song.mp3", "songs/gnossi_1.mp3", "songs/Einojuhani_RAUTAVAARA_Credo.mp3"]

lyric_songs = ["Dancing Queen.mp3", "like a rolling Stone.mp3", "Shia LaBeouf.mp3", "Smells Like Teen Spirit.mp3", "Space Oddity.mp3", "This Is America.mp3", "Without Me.mp3", "(bowie) space oddity.mp3", "be_my_weasel.mp3"]
lyrics = ["Dancing Queen.srt", "like a rolling Stone.srt", "Shia LaBeouf.srt", "Smells Like Teen Spirit.srt", "Space Oddity.srt", "This Is America.srt", "Without Me.srt", "(bowie) space oddity.srt", "space_oddity_bowie_custom_1.srt", "be_my_weasel.srt"]

lyric_songs = [os.path.join("songs_with_lyrics", l) for l in lyric_songs]
lyrics = [os.path.join("songs_with_lyrics", l) for l in lyrics]

def hallus(style, song, height=496, width=496, model_type="stylegan", output_dir="outputs", **kwargs):
    time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())
    if model_type == "stylegan":
        style_name = style.split("/")[-1]
        replace_dict = {" ": "_", "-": "_", "(": "", ")": "", ".pkl": "", ":":"_"}
        for key in replace_dict:
            style_name = style_name.replace(key, replace_dict[key])
    else:
        style_name = "vqgan"
    
    L = LucidSonicDream(song=song, style=style, height=height, width=width, model_type=model_type)
    song_name = song.split("/")[-1].split(".")[0]
    file_name = f"{time_str}_{song_name}_{style_name}.mp4"

    L.hallucinate(file_name=file_name, output_dir=output_dir, **kwargs)


    
hallus(styles["strangegods"], songs[2], batch_size=4, duration=10, offset=5, use_all_layers=0, use_old_beat = 1, use_song_latent_std = 1, 
                  pulse_percussive = True,
                  pulse_harmonic = False,
                  pulse_react = 0.3,
                  motion_percussive = False,
                  motion_harmonic = True,
                  motion_react = 0.85)
    
quit()
    
hallus(styles["strangegods"], songs[2], batch_size=4, duration=None, use_all_layers=0, use_old_beat = 1, use_song_latent_std = 0, pulse_percussive = True,
                  pulse_harmonic = False,
                  pulse_react = 0.3,
                  motion_percussive = False,
                  motion_harmonic = True,
                  motion_react = 0.85)

quit()

hallus(styles["strangegods"], songs[2], batch_size=4, duration=None, use_all_layers=0, use_old_beat = 1, use_song_latent_std = 0, pulse_percussive = True,
                  pulse_harmonic = False,
                  pulse_react = 0.25,
                  motion_percussive = False,
                  motion_harmonic = True,
                  motion_react = 0.65)

quit()

hallus(styles["strangegods"], songs[2], batch_size=4, duration=30, use_all_layers=0, use_old_beat = 1, use_song_latent_std = 0, 
                  pulse_percussive = 0,
                  pulse_harmonic = 1,
                  pulse_react = 0.2,
                  motion_percussive = 1,
                  motion_harmonic = 0,
                  motion_react = 0.6)

hallus(styles["strangegods"], songs[2], batch_size=4, duration=30, use_all_layers=0, use_old_beat = 1, use_song_latent_std = 0, 
                  pulse_percussive = 0,
                  pulse_harmonic = 1,
                  pulse_react = 0.6,
                  motion_percussive = 1,
                  motion_harmonic = 0,
                  motion_react = 0.2)


quit()

# bowie#


clip_opt_kwargs = {"batch_size": 32, "use_tv_loss": 1, "neg_text": 'incoherent, confusing, cropped, watermarks',
                  "decay_cutout": 0}

#hallus(None, lyric_songs[7], width=1080, height=720, lyrics_path=lyrics[8], batch_size=4, visualize_lyrics=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=1.0, motion_react=0.00, lyrics_iterations=1000, reset_latents_after_phrase=1, model_type="vqgan", use_all_layers=0, no_beat=1, clip_opt_kwargs=clip_opt_kwargs)

#hallus(None, lyric_songs[2], width=1080, height=720, lyrics_path=lyrics[2], batch_size=4, visualize_lyrics=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=1.0, motion_react=0.00, lyrics_iterations=2000, reset_latents_after_phrase=1, model_type="vqgan", use_all_layers=0, no_beat=1, clip_opt_kwargs=clip_opt_kwargs)

#hallus(None, lyric_songs[2], width=1080, height=720, lyrics_path=lyrics[2], batch_size=4, visualize_lyrics=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=0.5, motion_react=0.00, lyrics_iterations=2000, reset_latents_after_phrase=1, model_type="vqgan", use_all_layers=0, no_beat=0, clip_opt_kwargs=clip_opt_kwargs)

#hallus(None, lyric_songs[2], width=1080, height=720, lyrics_path=lyrics[2], batch_size=4, visualize_lyrics=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=0.5, motion_react=0.5, lyrics_iterations=2000, reset_latents_after_phrase=1, model_type="vqgan", use_all_layers=0, no_beat=0, clip_opt_kwargs=clip_opt_kwargs)

# no reset

#hallus(None, lyric_songs[2], width=1080, height=720, lyrics_path=lyrics[2], batch_size=4, visualize_lyrics=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=1.0, motion_react=0.00, lyrics_iterations=1500, reset_latents_after_phrase=0, model_type="vqgan", use_all_layers=0, no_beat=1, clip_opt_kwargs=clip_opt_kwargs)

clip_opt_kwargs = {"batch_size": 32}
#hallus(None, lyric_songs[7], width=1080, height=720, lyrics_path=lyrics[8], batch_size=1, visualize_lyrics=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=1.0, motion_react=0.00, lyrics_iterations=1000, reset_latents_after_phrase=1, model_type="vqgan", use_all_layers=0, no_beat=0, clip_opt_kwargs=clip_opt_kwargs)
#hallus(None, lyric_songs[7], width=1080, height=720, lyrics_path=lyrics[8], batch_size=4, visualize_lyrics=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=1.0, motion_react=0.00, lyrics_iterations=1000, reset_latents_after_phrase=1, model_type="vqgan", use_all_layers=0, no_beat=1, clip_opt_kwargs=clip_opt_kwargs)
#shia
#hallus(None, lyric_songs[2], width=1080, height=720, lyrics_path=lyrics[2], batch_size=4, visualize_lyrics=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=1.0, motion_react=0.00, lyrics_iterations=1000, reset_latents_after_phrase=1, model_type="vqgan", use_all_layers=0, no_beat=0, clip_opt_kwargs=clip_opt_kwargs)
#hallus(None, lyric_songs[2], width=1080, height=720, lyrics_path=lyrics[2], batch_size=4, visualize_lyrics=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=1.0, motion_react=0.00, lyrics_iterations=1000, reset_latents_after_phrase=1, model_type="vqgan", use_all_layers=0, no_beat=1, clip_opt_kwargs=clip_opt_kwargs)


quit()

hallus(styles["vint_retro_scifi"], lyric_songs[0], batch_size=4, duration=25, pulse_react=1.0, motion_react=1.0, use_all_layers=0, use_old_beat = 1, use_song_latent_std = 0)
hallus(styles["vint_retro_scifi"], lyric_songs[0], batch_size=4, duration=25, pulse_react=1.0, motion_react=1.0, use_all_layers=0, use_old_beat = 1, use_song_latent_std = 1)
hallus(styles["vint_retro_scifi"], lyric_songs[0], batch_size=4, duration=25, pulse_react=1.0, motion_react=1.0, use_all_layers=0, use_old_beat = 0, use_song_latent_std = 0)
hallus(styles["vint_retro_scifi"], lyric_songs[0], batch_size=4, duration=25, pulse_react=1.0, motion_react=1.0, use_all_layers=0, use_old_beat = 0, use_song_latent_std = 1)
    
quit()
    
clip_opt_kwargs = {"latent_type": "code_sampling"}
#hallus(None, lyric_songs[8], width=480, height=480, lyrics_path=lyrics[9], batch_size=1, visualize_lyrics=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=0.15, motion_react=0.00, lyrics_iterations=1500, reset_latents_after_phrase=1, model_type="vqgan", use_all_layers=0, no_beat=0, clip_opt_kwargs=clip_opt_kwargs)
clip_opt_kwargs = {"circular": 1}
#hallus(None, lyric_songs[8], width=480, height=480, lyrics_path=lyrics[9], batch_size=1, visualize_lyrics=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=0.5, motion_react=0.00, lyrics_iterations=1500, reset_latents_after_phrase=1, model_type="vqgan", use_all_layers=0, no_beat=0, clip_opt_kwargs=clip_opt_kwargs)
clip_opt_kwargs = None
hallus(None, lyric_songs[6], width=480, height=480, lyrics_path=lyrics[6], batch_size=1, visualize_lyrics=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=1.0, motion_react=0.00, lyrics_iterations=1000, reset_latents_after_phrase=1, model_type="vqgan", use_all_layers=0, no_beat=0, clip_opt_kwargs=clip_opt_kwargs)

hallus(None, lyric_songs[5], width=480, height=480, lyrics_path=lyrics[5], batch_size=1, visualize_lyrics=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=1.0, motion_react=0.00, lyrics_iterations=1000, reset_latents_after_phrase=1, model_type="vqgan", use_all_layers=0, no_beat=0, clip_opt_kwargs=clip_opt_kwargs)
quit()

#hallus(None, lyric_songs[7], width=480, height=480, lyrics_path=lyrics[8], batch_size=2, visualize_lyrics=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=0.15, motion_react=0.00, lyrics_iterations=1000, reset_latents_after_phrase=1, model_type="vqgan", use_all_layers=0, no_beat=0)

#hallus(None, lyric_songs[7], width=480, height=480, lyrics_path=lyrics[7], batch_size=2, visualize_lyrics=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=0.1, motion_react=0.03, lyrics_iterations=2000, reset_latents_after_phrase=1, model_type="vqgan", use_all_layers=0)

    
#cluster_pitches: "spectral","chroma_cqt", "chroma_stft
#hallus(styles["vint_retro_scifi"], songs[1], batch_size=4, duration=30, pulse_react=0.1, motion_react=0.05, use_all_layers=1, cluster_pitches="chroma_stft", start=10)
#hallus(styles["vint_retro_scifi"], songs[1], batch_size=4, duration=30, pulse_react=0.1, motion_react=0.05, use_all_layers=1, cluster_pitches="spectral", start=10)
#hallus(styles["vint_retro_scifi"], songs[1], batch_size=4, duration=30, pulse_react=0.1, motion_react=0.05, use_all_layers=1, cluster_pitches="chroma_cqt", start=10)

    
# stylegan test
#hallus(styles["vint_retro_scifi"], lyric_songs[0], batch_size=4, duration=10, pulse_react=0.1, motion_react=0.05, use_all_layers=0)
#hallus(styles["vint_retro_scifi"], lyric_songs[0], batch_size=4, duration=10, pulse_react=0.1, motion_react=0.05, use_all_layers=1)

# vqgan no lyrics
#hallus(styles["vint_retro_scifi"], lyric_songs[0], batch_size=1, duration=10, pulse_react=0.1, motion_react=0.05, model_type="vqgan")
# with lyrics
#hallus(styles["vint_retro_scifi"], lyric_songs[7], lyrics_path=lyrics[8], batch_size=1, visualize_lyrics=1, no_beat=0, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=0.3, motion_react=0.0, lyrics_iterations=50, reset_latents_after_phrase=1, model_type="vqgan", use_all_layers=0)
quit()

hallus(None, lyric_songs[7], width=480, height=480, lyrics_path=lyrics[7], batch_size=4, visualize_lyrics=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=1.2, motion_react=0.0, lyrics_iterations=2000, reset_latents_after_phrase=1, model_type="vqgan", use_all_layers=0)

#hallus(styles["vint_retro_scifi"], lyric_songs[7], lyrics_path=lyrics[8], batch_size=1, visualize_lyrics=1, no_beat=0, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7.5, ampl_influences_speed=1, pulse_react=0.6, motion_react=0.0, lyrics_iterations=3000, reset_latents_after_phrase=1, model_type="vqgan", use_all_layers=0)

quit()

# 
hallus(styles["vint_retro_scifi"], lyric_songs[7], lyrics_path=lyrics[8], batch_size=4, visualize_lyrics=1, no_beat=0, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=0.3, motion_react=0.0, lyrics_iterations=3000, reset_latents_after_phrase=1)
    
quit()
    
hallus(styles["vint_retro_scifi"], lyric_songs[7], lyrics_path=lyrics[8], batch_size=4, visualize_lyrics=1, no_beat=0, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=0.3, motion_react=0.0, lyrics_iterations=3000, reset_latents_after_phrase=1)

quit()
    
#hallus(styles["vint_retro_scifi"], lyric_songs[0], lyrics_path=lyrics[0], batch_size=4, visualize_lyrics=0, no_beat=0, duration=10, pulse_react=0.2, motion_react=0.1, cluster_pitches=None, use_all_layers=0)

    
hallus(styles["vint_retro_scifi"], lyric_songs[0], lyrics_path=lyrics[0], batch_size=4, visualize_lyrics=0, no_beat=0, duration=10, pulse_react=0.0, motion_react=300.0, cluster_pitches=None, speed_fpm=0)

quit()
    
#hallus(styles["vint_retro_scifi"], lyric_songs[0], lyrics_path=lyrics[0], batch_size=4, visualize_lyrics=0, no_beat=0, duration=60, pulse_react=0.5, motion_react=0.5, cluster_pitches="spectral")

#hallus(styles["vint_retro_scifi"], lyric_songs[0], lyrics_path=lyrics[0], batch_size=4, visualize_lyrics=0, no_beat=0, duration=60, pulse_react=0.5, motion_react=0.5, cluster_pitches="chroma_cqt")

#hallus(styles["vint_retro_scifi"], lyric_songs[0], lyrics_path=lyrics[0], batch_size=4, visualize_lyrics=0, no_beat=0, duration=60, pulse_react=0.5, motion_react=0.5, cluster_pitches="chroma_stft")

hallus(styles["vint_retro_scifi"], lyric_songs[0], lyrics_path=lyrics[0], batch_size=4, visualize_lyrics=0, no_beat=0, duration=30, pulse_react=0.3, motion_react=0.3, cluster_pitches=None)

hallus(styles["vint_retro_scifi"], lyric_songs[0], lyrics_path=lyrics[0], batch_size=4, visualize_lyrics=0, no_beat=0, duration=30, pulse_react=0.3, motion_react=0.1, cluster_pitches=None)

hallus(styles["vint_retro_scifi"], lyric_songs[0], lyrics_path=lyrics[0], batch_size=4, visualize_lyrics=0, no_beat=0, duration=30, pulse_react=0.3, motion_react=0.05, cluster_pitches=None)

hallus(styles["vint_retro_scifi"], lyric_songs[0], lyrics_path=lyrics[0], batch_size=4, visualize_lyrics=0, no_beat=0, duration=30, pulse_react=0.3, motion_react=0.3, cluster_pitches=None, truncation=0.5)

hallus(styles["vint_retro_scifi"], lyric_songs[0], lyrics_path=lyrics[0], batch_size=4, visualize_lyrics=0, no_beat=0, duration=30, pulse_react=0.3, motion_react=0.3, cluster_pitches=None, truncation=0.3)

hallus(styles["vint_retro_scifi"], lyric_songs[0], lyrics_path=lyrics[0], batch_size=4, visualize_lyrics=0, no_beat=0, duration=30, pulse_react=0.3, motion_react=0.3, cluster_pitches=None, truncation=0.1)

    
quit()
    
    
hallus(styles["visionaryart"], lyric_songs[0], lyrics_path=lyrics[0], batch_size=4, visualize_lyrics=1, no_beat=0, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, reset_latents_after_phrase=1, lyrics_iterations=1000, pulse_react=0.3, motion_react=0.0)


hallus(styles["visionaryart"], lyric_songs[0], lyrics_path=lyrics[0], batch_size=4, visualize_lyrics=1, no_beat=0, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, reset_latents_after_phrase=0, lyrics_iterations=1000, pulse_react=0.3, motion_react=0.0)

hallus(styles["visionaryart"], lyric_songs[0], lyrics_path=lyrics[0], batch_size=4, visualize_lyrics=1, no_beat=0, duration=60, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, reset_latents_after_phrase=1, lyrics_iterations=1000, pulse_react=0.3, motion_react=0.1)

hallus(styles["visionaryart"], lyric_songs[0], lyrics_path=lyrics[0], batch_size=4, visualize_lyrics=1, no_beat=0, duration=60, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, reset_latents_after_phrase=1, lyrics_iterations=1000, pulse_react=0.3, motion_react=0.05)

hallus(styles["visionaryart"], lyric_songs[0], lyrics_path=lyrics[0], batch_size=4, visualize_lyrics=1, no_beat=0, duration=60, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, reset_latents_after_phrase=1, lyrics_iterations=1000, pulse_react=0.3, motion_react=0.01)






quit()

hallus(styles["madziowa"], lyric_songs[2], lyrics_path=lyrics[2], batch_size=4, visualize_lyrics=1, no_beat=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, reset_latents_after_phrase=0, lyrics_iterations=1000)

#hallus(styles["vint_retro_scifi"], lyric_songs[7], lyrics_path=lyrics[7], batch_size=2, visualize_lyrics=1, no_beat=0, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=0.0, motion_react=0.0, lyrics_iterations=2000, reset_latents_after_phrase=1)

#hallus(styles["vint_retro_scifi"], lyric_songs[7], lyrics_path=lyrics[7], batch_size=2, visualize_lyrics=1, no_beat=0, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=0.0, motion_react=0.0, lyrics_iterations=2000, concat_phrases=1)


#hallus(styles["madziowa"], lyric_songs[2], lyrics_path=lyrics[2], batch_size=4, visualize_lyrics=1, no_beat=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, reset_latents_after_phrase=0, lyrics_iterations=1000)

hallus(styles["madziowa"], lyric_songs[2], lyrics_path=lyrics[2], batch_size=4, visualize_lyrics=1, no_beat=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, reset_latents_after_phrase=0, lyrics_iterations=1000, concat_phrases=1)


quit()

hallus(styles["vint_retro_scifi"], lyric_songs[6], lyrics_path=lyrics[6], batch_size=4, visualize_lyrics=1, no_beat=0, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=0.4, motion_react=0.0, lyrics_iterations=1000)

hallus(styles["vint_retro_scifi"], lyric_songs[6], lyrics_path=lyrics[6], batch_size=4, visualize_lyrics=1, no_beat=0, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=0.4, motion_react=0.0, reset_latents_after_phrase=0)

hallus(styles["vint_retro_scifi"], lyric_songs[6], lyrics_path=lyrics[6], batch_size=4, visualize_lyrics=1, no_beat=0, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=0.4, motion_react=0.0, concat_phrases=1, reset_latents_after_phrase=0)
    
hallus(styles["madziowa"], lyric_songs[2], lyrics_path=lyrics[2], batch_size=4, visualize_lyrics=1, no_beat=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, reset_latents_after_phrase=0, lyrics_iterations=1000)

hallus(styles["madziowa"], lyric_songs[2], lyrics_path=lyrics[2], batch_size=4, visualize_lyrics=1, no_beat=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, reset_latents_after_phrase=0, lyrics_iterations=1000, concat_phrases=1)
    
quit()
    
hallus(styles["vint_retro_scifi"], lyric_songs[6], lyrics_path=lyrics[6], batch_size=4, visualize_lyrics=1, no_beat=0, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=0.4, motion_react=0.0)
    
quit()
    

hallus(styles["vint_retro_scifi"], lyric_songs[4], lyrics_path=lyrics[4], batch_size=8, visualize_lyrics=1, no_beat=1, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, concat_phrases=1)

hallus(styles["madziowa"], lyric_songs[2], lyrics_path=lyrics[2], batch_size=8, visualize_lyrics=1, no_beat=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1)
hallus(styles["madziowa"], lyric_songs[2], lyrics_path=lyrics[2], batch_size=8, visualize_lyrics=1, no_beat=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, concat_phrases=1)
hallus(styles["madziowa"], lyric_songs[2], lyrics_path=lyrics[2], batch_size=8, visualize_lyrics=1, no_beat=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, lyrics_iterations=1000)
#eminem
hallus(styles["vint_retro_scifi"], lyric_songs[6], lyrics_path=lyrics[6], batch_size=8, visualize_lyrics=1, no_beat=0, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=0.2, motion_react=0.0)
quit()

#hallus(styles["vint_retro_scifi"], lyric_songs[4], lyrics_path=lyrics[4], batch_size=8, visualize_lyrics=1, no_beat=1, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=0, duration=60)
#hallus(styles["vint_retro_scifi"], lyric_songs[4], lyrics_path=lyrics[4], batch_size=8, visualize_lyrics=1, no_beat=1, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, duration=60)
#hallus(styles["vint_retro_scifi"], lyric_songs[4], lyrics_path=lyrics[4], batch_size=8, visualize_lyrics=1, no_beat=0, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1, pulse_react=0.1, motion_react=0.0, pulse_harmonic=1, pulse_percussive=1)

#hallus(styles["astromaniacmag"], lyric_songs[4], lyrics_path=lyrics[4], batch_size=8, visualize_lyrics=1, no_beat=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7)

#hallus(styles["madziowa"], lyric_songs[2], lyrics_path=lyrics[2], batch_size=8, visualize_lyrics=1, no_beat=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=9, ampl_influences_speed=1)
#hallus(styles["therealtheory"], lyric_songs[3], lyrics_path=lyrics[3], batch_size=8, visualize_lyrics=1, no_beat=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1)
#hallus(styles["ffhq"], lyric_songs[5], lyrics_path=lyrics[5], batch_size=8, visualize_lyrics=1, no_beat=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1)
hallus(styles["vint_retro_scifi"], lyric_songs[6], lyrics_path=lyrics[6], batch_size=8, visualize_lyrics=1, no_beat=1, duration=None, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1)
hallus(styles["vint_retro_scifi"], lyric_songs[4], lyrics_path=lyrics[4], batch_size=8, visualize_lyrics=1, no_beat=1, lyrics_sigmoid_transition=1, lyrics_sigmoid_t=7, ampl_influences_speed=1)
    
    
#hallus(styles["vint_retro_scifi"], lyric_songs[4], lyrics_path=lyrics[4], batch_size=8, visualize_lyrics=1, no_beat=1, duration=None, ampl_influences_speed=1)
#hallus(styles["vint_retro_scifi"], lyric_songs[4], lyrics_path=lyrics[4], batch_size=8, visualize_lyrics=1, no_beat=1, duration=90, ampl_influences_speed=1, lyrics_sigmoid_transition=1)
    
    
quit()

# also just a shaky effect...
hallus(styles["vint_retro_scifi"], lyric_songs[4], lyrics_path=lyrics[4], batch_size=8, visualize_lyrics=1, no_beat=0, duration=60, pulse_react=0.5, motion_react=0.0, pulse_harmonic=1, pulse_percussive=0)
hallus(styles["vint_retro_scifi"], lyric_songs[4], lyrics_path=lyrics[4], batch_size=8, visualize_lyrics=1, no_beat=0, duration=60, pulse_react=0.2, motion_react=0.0, pulse_harmonic=1, pulse_percussive=0)

quit()

hallus(styles["vint_retro_scifi"], lyric_songs[4], lyrics_path=lyrics[4], batch_size=8, visualize_lyrics=1, no_beat=0, duration=60, pulse_react=0.5, motion_react=0.0) # baad. song has no percussion, so it just shakes a bit when the voice comes up
hallus(styles["vint_retro_scifi"], lyric_songs[4], lyrics_path=lyrics[4], batch_size=8, visualize_lyrics=1, no_beat=0, duration=60, pulse_react=0.1, motion_react=0.0) # also bad, but nearly no effect
hallus(styles["vint_retro_scifi"], lyric_songs[4], lyrics_path=lyrics[4], batch_size=8, visualize_lyrics=1, no_beat=0, duration=60, pulse_react=0.0, motion_react=0.5)  # really bad. the piano introduces large changes that are completely ood, so we just see bright colors 
hallus(styles["vint_retro_scifi"], lyric_songs[4], lyrics_path=lyrics[4], batch_size=8, visualize_lyrics=1, no_beat=0, duration=60, pulse_react=0.0, motion_react=0.1) # still too strong!
hallus(styles["vint_retro_scifi"], lyric_songs[4], lyrics_path=lyrics[4], batch_size=8, visualize_lyrics=1, no_beat=0, duration=60, pulse_react=0.1, motion_react=0.1)
#hallus(styles["vint_retro_scifi"], lyric_songs[4], lyrics_path=lyrics[4], batch_size=8, visualize_lyrics=1, no_beat=0, duration=70)

    
quit()
    
hallus(styles[1], lyric_songs[2], lyrics_path=lyrics[2], batch_size=4, visualize_lyrics=1)
    
quit()
    
hallus(styles[-1], songs[1], duration=None, batch_size=4, pulse_react=0.55, motion_react=1.0, fps=60, speed_fpm=1, start=7) 

quit()
    
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=0, clmr_softmax_t=0.1, clmr_ema=0.0, batch_size=4, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10, duration=30) 
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=0, clmr_softmax_t=0.1, clmr_ema=0.05, batch_size=8, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10, duration=30) 

quit()

#hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.1, clmr_ema=0.4, batch_size=4, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10, duration=30) 
# old style to see networks
hallus(styles[-1], songs[0], duration=None, batch_size=4, pulse_react=0.55, motion_react=1.0, fps=60, speed_fpm=1, start=0) 
hallus(styles[-1], songs[1], duration=None, batch_size=4, pulse_react=0.55, motion_react=1.0, fps=60, speed_fpm=1, start=7) 


quit()
    
hallus(styles[1], songs[0], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.1, clmr_ema=0.2, duration=60, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=0) 
    
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.5, clmr_ema=0.2, duration=None, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10)    
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.2, clmr_ema=0.2, duration=None, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10)    
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.1, clmr_ema=0.2, duration=None, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10)    
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.05, clmr_ema=0.2, duration=None, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10) 


quit()

hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=0, clmr_softmax_t=0.05, clmr_ema=0.3, duration=30, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10)
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=0, clmr_softmax_t=0.05, clmr_ema=0.6, duration=30, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10)
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=0, clmr_softmax_t=0.05, clmr_ema=0.9, duration=30, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10)
    
    
quit()

#hallus(styles[0], songs[1], use_clmr=1, clmr_softmax=0, duration=20, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=7)
#hallus(styles[0], songs[1], use_clmr=1, clmr_softmax=1, duration=20, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=7)
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.05, clmr_ema=0.3, duration=30, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10)
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.05, clmr_ema=0.6, duration=30, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10)
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.05, clmr_ema=0.9, duration=30, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10)


quit()
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.05, clmr_ema=0.99, duration=30, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10)
hallus(styles[1], songs[1], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.05, clmr_ema=0.999, duration=30, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=10)
hallus(styles[1], songs[0], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.05, clmr_ema=0.9999, duration=30, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=0)
#hallus(styles[0], songs[1], use_clmr=1, clmr_softmax=1, clmr_softmax_t=0.01, duration=20, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=43, speed_fpm=1, start=7)

quit()

#hallus(styles[0], songs[1], duration=None, start=7, batch_size=16, truncation_psi=0.9, pulse_react=1.05, motion_react=0.75, fps=60)
#hallus(styles[1], songs[1], duration=None, start=7, batch_size=16, pulse_react=0.9, motion_react=0.85, fps=60)
#hallus(styles[5], songs[1], duration=None, start=7, batch_size=16, truncation_psi=0.9, pulse_react=1.05, motion_react=0.75, fps=60)
hallus(styles[0], songs[1], duration=None, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=60, speed_fpm=1)
hallus(styles[5], songs[1], duration=None, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=60, speed_fpm=1)

hallus(styles[1], songs[0], duration=None, batch_size=16, pulse_react=0.55, motion_react=1.0, fps=60, speed_fpm=1)
#hallus(styles[1], songs[0], duration=None, batch_size=16, pulse_react=0.7, motion_react=0.9, fps=60, speed_fpm=1)
#hallus(styles[1], songs[0], duration=None, batch_size=16, pulse_react=0.7, motion_react=0.9, fps=60, speed_fpm=12)


quit()

#hallus(styles[1], duration=1)
#hallus(styles[0], duration=1)

#hallus(styles[1])

hallus(styles[-1], duration=60, start=210, speed_fpm=3)
hallus(styles[-1], duration=60, start=210, speed_fpm=24)
#hallus(styles[-1], duration=60, start=210, class_complexity=0.5)
#hallus(styles[-1], duration=60, start=210, class_complexity=0.1)
hallus(styles[-1], duration=60, start=210, fps=24)
hallus(styles[-1], duration=60, start=210, batch_size=4)
hallus(styles[-1], duration=60, start=210, batch_size=16)
hallus(styles[-1], pulse_react=1.2, motion_react=0.7)

hallus(styles["floor_plans"], lyric_songs[-1], duration=10, batch_size=2, pulse_react=0.55, motion_react=1.0, fps=42, speed_fpm=1)
