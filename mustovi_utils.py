from collections import defaultdict

import musicnn
from musicnn.extractor import extractor
from musicnn.tagger import top_tags
import pandas as pd
import numpy as np
import librosa
import torch
import pandas as pd
from clip import tokenize
#remove deprecation warnings
#import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


def get_taggram(resampled_path, input_overlap, input_length):
    #MTT_musicnn', 'MTT_vgg', 'MSD_musicnn', 'MSD_musicnn_big' or 'MSD_vgg'.
    try:
        taggram_msd, tags_msd = extractor(resampled_path, model='MSD_musicnn_big', extract_features=False, 
                                          input_overlap=input_overlap, input_length=input_length)
    except ValueError:
        print("Please install musicnn from source to use MSD_musicnn_big. Defaulting to small model...")
        taggram_msd, tags_msd = extractor(resampled_path, model='MSD_musicnn', extract_features=False, 
                                          input_overlap=input_overlap, input_length=input_length)
    taggram_mtt, tags_mtt = extractor(resampled_path, model='MTT_musicnn', extract_features=False, 
                                      input_overlap=input_overlap, input_length=input_length)
    # clear cuda
    from numba import cuda
    cuda.select_device(0)
    cuda.close()
    # merge taggrams
    tag_df_msd = pd.DataFrame(taggram_msd, columns=tags_msd)
    tag_df_mtt = pd.DataFrame(taggram_mtt, columns=tags_mtt)
    tag_df = pd.concat([tag_df_msd, tag_df_mtt], axis=1)
    # merge beat (before normalization because we merged them before calculating the stds)
    tag_df["Beat"] = (tag_df["beat"] + tag_df["beats"]) / 2
    tag_df = tag_df.drop(columns=["beat", "beats"])
    # identify duplicated columns
    duplicated_cols = set()
    for col in tag_df.columns:
        if isinstance(tag_df[col], pd.DataFrame):
            duplicated_cols.add(col)
    # average duplicated columns
    for duplicated_col in list(duplicated_cols):
        averaged = tag_df[duplicated_col].mean(axis=1)
        del tag_df[duplicated_col]
        tag_df[duplicated_col] = averaged
    # normalize taggram by std on the MTT dataset
    mtt_stds = pd.read_csv("data/mtt/mtt_stds.csv", header=None, index_col=0, squeeze=True).iloc[1:]
    normed_tag_df = tag_df / mtt_stds
    # merge some more columns
    merge_dict = {("male vocal", "male voice", "male vocalists"): "male vocal",
                  ("female vocal", "female voice", "female vocalists"): "female vocal",
                  ("vocal", "vocals", "voice"): "vocal",
                  ("no vocal", "no vocals", "no voice"): "no vocal",
                  ("electro", "electronic"): "electro",
                  ("choir", "choral"): "choral", 
                 }
    print(normed_tag_df.columns)
    for cols_to_merge in merge_dict:
        cols_to_merge_list = list(cols_to_merge)
        merged_col = normed_tag_df[cols_to_merge_list].to_numpy().mean(axis=1)
        normed_tag_df = normed_tag_df.drop(columns=cols_to_merge_list)
        normed_tag_df[merge_dict[cols_to_merge]] = merged_col
    return normed_tag_df


def get_spec_norm(song):
    mel_spec = librosa.feature.melspectrogram(song, sr=16000, S=None, 
                                              n_fft=512, hop_length=256, 
                                              win_length=None, window='hann', center=True, 
                                              pad_mode='reflect', power=2.0)
    # Obtain maximum value per time-frame
    spec_max = np.amax(mel_spec, axis=0)
    #print(spec_max.shape)
    # Normalize all values between 0 and 1
    mel_spec = (mel_spec - np.min(spec_max)) / np.ptp(spec_max)

    #mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
    #sns.heatmap(mel_spec[:20, :])
    return mel_spec.mean(axis=0)


def slerp(low, high, val):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    epsilon = 1e-7
    omega = (low_norm * high_norm).sum(1)
    omega = torch.acos(torch.clamp(omega, -1 + epsilon, 1 - epsilon))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res


def load_gpt_model(name):
    import transformers
    from transformers import GPTNeoForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == "neo1.3":
        gpt_tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        gpt_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", pad_token_id=gpt_tokenizer.eos_token_id)
    elif name == "neo2.7":
        gpt_tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
        gpt_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B", pad_token_id=gpt_tokenizer.eos_token_id)
    elif name == "gpt2":
        gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        gpt_model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=gpt_tokenizer.eos_token_id)
    elif name == "gptj":
        # monkey-patch GPT-J for low precision storage
        from gpt_j_low_prec import GPTJBlock, GPTJForCausalLM
        transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock
        
        #from transformers import GPTJForCausalLM
        
        # load 
        config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
        gpt_tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        gpt_model = GPTJForCausalLM.from_pretrained("hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)

    gpt_model = gpt_model.to(device)
    return gpt_model, gpt_tokenizer


def gen_sent(gpt_model, gpt_tokenizer, clip_model, target_clip_feats, 
             start_text="", p=0.9,
             prefix="", examples=None, prompter=None, target_text=None,
             clip_weight=0.4, 
             clip_temp=0.45, gpt_temp=0.75, out_len=50, v=0, num_beams=20, return_num=1):
    
    context_text = prefix
    if examples is not None:
        # add examples to prompt
        for example_input in examples:
            example_target = examples[example_input]
            context_text += f"{example_input}{prompter}{example_target}\n "
        # add line to prompt actual output
        context_text += f"{target_text}{prompter}"   
    print("Context: ", "\n", context_text)
    
    data_dict = defaultdict(list)
    for _ in range(num_beams):
        sent, scores, clip_scores, gpt_scores = sample_GPT_with_CLIP(gpt_model, gpt_tokenizer, clip_model, target_clip_feats,
                         start_text=start_text,
                         p=p, 
                         context_text=context_text,
                         clip_weight=clip_weight,
                         clip_temp=clip_temp,
                         gpt_temp=gpt_temp,
                         out_len=out_len,
                         v=v,
                        )
        if v > -1:
            #print("Scores:")
            #print(scores.round(2))
            print(clip_scores.round(2))
            #print(gpt_scores.round(2))
            print("During scores:", scores.mean().round(2))
            print("Post scores:", (clip_scores[gpt_scores < 0.95].mean() + gpt_scores[gpt_scores < 0.95].mean()) / 2)
            print("Post score weighted:", (clip_scores[gpt_scores < 0.95].mean()  * (clip_weight)+ gpt_scores[gpt_scores < 0.95].mean() * (1-clip_weight)))
            print(clip_scores.mean().round(2))
            print(gpt_scores.mean().round(2))
            print("Sent:", sent)
            #print(scores.mean() * np.sqrt(len(scores)))
            print()
        data_dict["sent"].append(sent)
        data_dict["during_score"].append(scores.mean())
        data_dict["post_score"].append((clip_scores[gpt_scores < 0.95].mean()  * (clip_weight)+ gpt_scores[gpt_scores < 0.95].mean() * (1-clip_weight)))
        data_dict["clip_score"].append(clip_scores.mean())
        data_dict["gpt_score"].append(gpt_scores.mean())
    
    df = pd.DataFrame(data_dict)
    
    df = add_example_based_metrics(df, clip_model, target_clip_feats, examples=examples)
    
    best_sentence = select_best_beam(df, return_num=return_num)
    
    if v > 0:
        return best_sentence, df
    else:
        return best_sentence


def norm(a):
    return a / a.norm(dim=-1, keepdim=True)


def add_example_based_metrics(df, clip_model, target_clip_feats, examples=None):
    # filter out sentences that just imitate the target text
    out_encodings = clip_model.encode_text(tokenize(list(df["sent"])).to("cuda"))
    similarity_score = torch.cosine_similarity(target_clip_feats, out_encodings)
    df["target_similarity"] = similarity_score.tolist()

    # filter out sentences that copy a given example
    if examples is not None:
        example_target_encodings = clip_model.encode_text(tokenize(list(examples.values())).to("cuda"))
        similarity_score = [torch.cosine_similarity(out_encoding.unsqueeze(0), example_target_encodings).max().item()
                            for out_encoding in out_encodings]
        df["example_similarity"] = similarity_score

        example_input_encodings = clip_model.encode_text(tokenize(list(examples.keys())).to("cuda"))
        # get directions
        directions = norm(norm(example_target_encodings) - norm(example_input_encodings))
        mean_dir = norm(directions.mean(dim=0, keepdim=True))
        # get similarity of predicted directions and mean training direction
        generated_dirs = norm(norm(out_encodings) - norm(target_clip_feats))
        dir_sims = torch.cosine_similarity(generated_dirs, mean_dir).cpu().tolist()
        df["direction_similarity"] = dir_sims
    return df

def select_best_beam(df, return_num=1,
                     target_sim_thresh=0.93, example_sim_thresh=0.89, dir_sim_thresh=0.25):
    
    # apply filters as long as they do not delete all prompts
    df = df[df["target_similarity"] < target_sim_thresh] if len(df[df["target_similarity"] < target_sim_thresh]) > 0 else df
    if "example_similarity" in df.columns:
        df = df[df["example_similarity"] < example_sim_thresh] if len(df[df["example_similarity"] < example_sim_thresh]) > 0 else df
    if "direction_similarity" in df.columns:
        df = df[df["direction_similarity"] > dir_sim_thresh] if len(df[df["direction_similarity"] > dir_sim_thresh * (2 / 3)]) > 0 else df
        df = df[df["direction_similarity"] > dir_sim_thresh] if len(df[df["direction_similarity"] > dir_sim_thresh]) > 0 else df

    best_sentence = df.sort_values("post_score", ascending=False).iloc[:return_num]["sent"].to_list()
    best_sentence = [s.strip() for s in best_sentence]
    return best_sentence


@torch.inference_mode()
def sample_GPT_with_CLIP(model, tokenizer, clip_model, target_clip_feats,
                         start_text="This", 
                         context_text="",
                         p=0.55, # top-p for GPT
                         k=200, # top-k for GPT
                         clip_temp=0.001, 
                         gpt_temp=0.7,
                         clip_weight=1.0,
                         out_len=100,
                         clip_bs=8,
                         max_tokens=77,
                         v=False,
                         forbidden_words=";\/{}()[]|<>*+~#'-~",
                        ):
    
    forbidden_tokens = [tokenizer.encode(w)[0] for w in forbidden_words]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    context_tokens = tokenizer.encode(context_text)
    generated = tokenizer.encode(context_text + start_text) if len(context_tokens) > 0 else [0]
    context = torch.tensor([generated]).to(device)
    past = None

    num_choices_list = []
    losses = []

    scores = []
    clip_scores = []
    gpt_scores = []
    num_tokens = len(tokenizer.encode(start_text)) if len(start_text) > 0 else 0
    for i in range(out_len):
        output = model(context, past_key_values=past, use_cache=True)
        logits = output["logits"].squeeze(0)
        if logits.ndim > 1:
            logits = logits[-1]  # at first step logits for all tokens are calculated
        past = output["past_key_values"]  # to cache computations

        # choose token based on CLIP

        # create sentence options
        # filter out some tokens
        #top_tokens = torch.topk(logits[..., -1, :], k, dim=-1).indices.squeeze()  # top k
        logits = torch.softmax(logits / gpt_temp, dim=-1)
        logits[forbidden_tokens] = 0
        # top-p sampling
        sorted_logits = torch.sort(logits, descending=True)
        sorted_vals, sorted_idcs = sorted_logits.values[:k], sorted_logits.indices[:k]
        cumsum = torch.cumsum(sorted_vals, dim=0)
        mask = cumsum < p
        if mask[0] == 0:
            mask[0] = 1  # include at least one token
        top_tokens = sorted_idcs[mask]
        gpt_top_logits = sorted_vals[mask]


        top_generations = [generated[len(context_tokens):] + [token.item()] for token in top_tokens]
        top_sentences = [tokenizer.decode(gen) for gen in top_generations]

        if v > 2:
            print("Num sentences: ", len(top_sentences))
            print("Some top sentences: ", top_sentences[:10])
        
        # calc CLIP sentence-img similarity scores
        all_feats = []
        for i in range(0, len(top_sentences), clip_bs):
            sentence_batch = top_sentences[i: i + clip_bs]
            clip_encoded = tokenize(sentence_batch).to(device) #torch.cat([tokenize(sentence_batch) for sentence in sentence_batch]).to(device)
            clip_sentence_feats = clip_model.encode_text(clip_encoded)
            all_feats.extend([f for f in clip_sentence_feats])
        all_feats = torch.stack(all_feats)
        clip_similarities = torch.nn.functional.cosine_similarity(all_feats, target_clip_feats, dim=-1)
        std = clip_similarities.std()
        if std > 0:
            normed_clip_similarities = (clip_similarities - clip_similarities.mean()) / std
        else:
            normed_clip_similarities = clip_similarities
        clip_similarities_softmax = torch.softmax(normed_clip_similarities / clip_temp, dim=0)

        #print(torch.topk(clip_similarities, min(10, len(clip_similarities)), dim=-1).values.tolist())  # top k

        # choose best fitting token
        #token = top_tokens[torch.argmax(clip_similarities)]  # greedy
        if v > 2:
            print("First 10 gpt logits. ", np.round(gpt_top_logits[:10].cpu().numpy(), 2))
            print("First 10 clip logits. ", np.round(clip_similarities_softmax[:10].cpu().numpy(), 2))
        weights = gpt_top_logits * (1 - clip_weight) + clip_similarities_softmax * clip_weight
        if v > 2:
            print("First 10 weights. ", weights[:10])
        idx = torch.multinomial(weights, num_samples=1)
        token = top_tokens[idx][0]
        scores.append(weights[idx].item())
        clip_scores.append(clip_similarities[idx].item())
        gpt_scores.append(gpt_top_logits[idx].item())


        #token = torch.argmax(logits[..., -1, :])    
        generated += [token.tolist()]
        context = token.unsqueeze(0)
        
        token_decoded = tokenizer.decode(token.tolist())
        if v > 1:
            print(token_decoded)
            
        num_tokens += 1
        
        if max_tokens is not None and num_tokens >= max_tokens:
            break
        if token_decoded == "\n":
            break

        num_choices_list.append(len(top_tokens))
        losses.append(clip_similarities[idx].detach().cpu().item())
        #pbar.set_description(f"{tokenizer.decode(generated)} - {len(top_tokens)}")

    sequence = tokenizer.decode(generated[len(context_tokens):])
    return sequence, np.array(scores), np.array(clip_scores), np.array(gpt_scores)



import torch

def gpt_create_prompt(model, tokenizer, merged_top_tags, num_beams=6, top_p=0.80, prompt_num=1):
    if prompt_num == 1:
        prompter = ". Full description:"
        gpt_prompt = f"""The following are adjectives describing a song, followed by a description of the corresponding image:
    fast, sad, dark{prompter} A man running through dark woods while crying.
    loud, techno, electronic, abstract{prompter} Dynamic and vibrant colors forming strong geometric shapes that give the impression of a rave.
    slow, soft, beautiful, sad, quiet{prompter} An old woman who is sitting on a chair with her hands folded in front of her. She is looking at you with a sad expression on her face.
    {merged_top_tags}{prompter}"""
    if prompt_num == 2:
        prompter = ". Full description:"
        gpt_prompt = f"""The following are adjectives describing an image, followed by a full description of the corresponding image:
    fast, sad, dark{prompter} A man is running through dark woods while crying.
    loud, techno, electronic, abstract{prompter} Dynamic and vibrant colors forming strong geometric shapes that give the impression of a rave.
    slow, soft, beautiful, sad, quiet{prompter} An old woman is sitting on a chair in a beautiful garden with her hands folded in front of her. She is looking at you with a sad expression on her face.
    {merged_top_tags}{prompter}"""
    
    out = model.generate(tokenizer.encode(gpt_prompt, return_tensors="pt").to(model.device),
                         top_p=top_p, do_sample=True, output_scores=True, 
                         return_dict_in_generate=True, 
                         max_length=220,
                         num_beams=num_beams, 
                         no_repeat_ngram_size=2,)
    out_text = tokenizer.decode(out["sequences"].tolist()[0])
    #print(out_text.split("Corresponding image description:")[3:])
    clip_prompt = out_text.split(prompter)[-1].split("\n")[0].strip().strip('<|endoftext|>')
    if not clip_prompt.endswith("."):
        clip_prompt = ".".join(clip_prompt.split(".")[:-1]) + "."
    return clip_prompt, out_text
