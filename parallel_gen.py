def unflatten_encodings(encodings, enc_idcs):
    return [unflatten_encoding(enc, enc_idcs) for enc in encodings]


def unflatten_encoding(encoding, enc_idcs):
    return [encoding[idcs[0]:idcs[1]] for idcs in enc_idcs]


def get_gpt_stories_and_weights(cluster_gpt_stories, n_start_prompts, dist_to_centers, gpt_story_top_k, idx, t=50):
    if cluster_gpt_stories is not None:
        story_idx = max(idx - n_start_prompts, 0)
        dists = dist_to_centers[story_idx]
        dists = 1 - (dists / dists.max())
        dists = torch.nn.functional.softmax(dists * t, dim=-1)
        top_k = dists.topk(k=gpt_story_top_k, largest=True)
        #story_weights = (1 - (top_k.values / dist_to_centers[story_idx].max())) ** 10
        story_weights = top_k.values
        top_idcs = top_k.indices
        gpt_stories = [cluster_gpt_stories[i] for i in top_idcs]
    else:
        gpt_stories = [""]
        story_weights = [1]
    return gpt_stories, story_weights


def parallel_gen(clip_prompts):
    sub_steps = 100
    
    all_encodings = []
    for i, prompt in enumerate(clip_prompts):
        gpt_stories, story_weights = get_gpt_stories_and_weights(used_gpt_stories, n_start_prompts, 
                                                                 dist_to_centers, gpt_story_top_k, i)
        story_prompt = gpt_stories[0]
        encoding = encode(story_prompt)
        all_encodings.append(encoding)
        
    latent_dict = dict()
    
    enc_lens = [len(enc) for enc in all_encodings[0]]
    enc_idcs = []
    last_idx = 0
    for l in enc_lens:
        enc_idcs.append((last_idx, last_idx + l))
        last_idx += l
    print(enc_lens)
    print(enc_idcs)
    print(torch.cat(all_encodings[0]).shape)
    flat_targets = [torch.cat(enc).float() for enc in all_encodings]
    
    unique_targets = torch.unique(torch.stack(flat_targets), dim=0)
    print("Num unique_targets: ", len(unique_targets))
    assert len(unique_targets) < 100
    
    
    target_dict = dict()
    img_latents = []
    for i, prompt in enumerate(tqdm(clip_prompts)):
        gpt_stories, story_weights = get_gpt_stories_and_weights(used_gpt_stories, n_start_prompts, 
                                                                 dist_to_centers, gpt_story_top_k, i, t=100)
        story_prompt = gpt_stories[0]
        for story_prompt in gpt_stories:
            if story_prompt in target_dict:
                latent = target_dict[story_prompt]
            else:
                print(story_prompt)
                encoding = encode(story_prompt)
    
        #for target in flat_targets:
         #   if target in latent_dict:
         #       latent = latent_dict[target]
         #   else:
                imagine.reset()
                #unflattend_target = unflatten_encoding(target.half(), enc_idcs)
                imagine.set_clip_encoding(encoding=encoding)
                for _ in range(sub_steps):
                    img, loss = imagine.train_step(0, 0)
                latent = imagine.model.model.get_latent().detach().cpu()
                target_dict[story_prompt] = latent
                pil_img = to_pil(img.squeeze())
                display(pil_img)
        
        #if i % 10 == 0:
        #    print(story_weights)
        story_encodings = [target_dict[story_prompt] for story_prompt in gpt_stories]
        clip_encoding = torch.sum(torch.stack([enc * weight for enc, weight in zip(story_encodings, story_weights)]), dim=0) / sum(story_weights)
        img_latents.append(clip_encoding.clone())
    
    return img_latents