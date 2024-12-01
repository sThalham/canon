import torch
from PIL import Image
from torchvision import transforms as th_transforms
from torch import nn
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from PIL import Image

from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.decomposition import PCA
import cv2

import torch.nn.functional as F
from torch import nn, Tensor


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    im_size = (224, 224)
    # model instantiation

    "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"

    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    #model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image")
    #conversation = [
    #    {
    #        "role": "user",
    #        "content": [
    #            {"type": "text", "text": "What are these?"},
    #            {"type": "image"},
    #        ],
    #    },
    #]
    #prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    img_path = "images/query_2.png"
    with Image.open(img_path) as im:
        raw_image = np.array(im)
        o_height, o_width, _ = raw_image.shape
        image = torch.tensor(raw_image, dtype=torch.float32)
        #raw_image = (np.array(im) * (1 / 255))

    #inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(device, torch.float16)

    #output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    # Generate
    generate_ids = model.generate(**inputs, max_new_tokens=15)
    processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(generate_ids)

    #outputs = model(**inputs)

    print("loop through model outputs")
    for idx, whatever in enumerate(outputs):
        print(idx, whatever)

    last_hidden_states = outputs[2]
    print(last_hidden_states.shape)
    #print(processor.decode(output[0][2:], skip_special_tokens=True))

    img_emb = last_hidden_states.detach().cpu()
    img_emb = img_emb[:, 1:, :]
    b, tokens, feat = img_emb.shape
    # img_emb = img_emb.view(b, int(math.sqrt(tokens)), int(math.sqrt(tokens)), feat)
    img_emb = img_emb.detach().cpu()

    # stupid projection to image space
    pca = PCA(n_components=3, svd_solver='full')
    img_viz = img_emb[0, ...]
    img_viz = pca.fit_transform(img_viz)
    # img_viz = pca.transform(img_viz)
    print(img_viz.shape)
    img_viz = np.reshape(img_viz, (int(math.sqrt(tokens)), int(math.sqrt(tokens)), 3))
    print(img_viz.shape)

    img_viz = cv2.resize(img_viz, dsize=(o_height, o_width), interpolation=cv2.INTER_CUBIC)
    img_viz[..., 0] = (img_viz[..., 0] - np.nanmin(img_viz[..., 0])) / np.max(
        img_viz[..., 0] - np.nanmin(img_viz[..., 0]))  # * (255 / np.max(img_viz[..., 0]))
    img_viz[..., 1] = (img_viz[..., 1] - np.nanmin(img_viz[..., 1])) / np.max(
        img_viz[..., 1] - np.nanmin(img_viz[..., 1]))  # * (255 / np.max(img_viz[..., 1]))
    img_viz[..., 2] = (img_viz[..., 2] - np.nanmin(img_viz[..., 2])) / np.max(
        img_viz[..., 2] - np.nanmin(img_viz[..., 2]))  # * (255 / np.max(img_viz[..., 2]))

    print("im_og: ", np.min(im_og), np.max(im_og))
    print("im_emb: ", np.min(img_viz), np.max(img_viz))
    img_comp = np.concatenate([im_og, img_viz], axis=1)
    plt.imshow(img_comp, interpolation='nearest')
    plt.show()

if __name__ == "__main__":
    main()