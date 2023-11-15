
import os, sys
from typing import Union

import torch.distributed
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from PIL import Image
import torch
import time
from sat.model.mixins import CachedAutoregressiveMixin
from sat.mpu import get_model_parallel_world_size

from utils.parser import parse_response
from utils.chat import chat
from models.cogvlm_model import CogVLMModel
from utils.language import llama2_tokenizer, llama2_text_processor_inference
from utils.vision import get_image_processor


RANK = int(os.environ.get('RANK', 0))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', None))


def process_image_without_resize(image_prompt):
    image = Image.open(image_prompt)
    timestamp = int(time.time())
    file_ext = os.path.splitext(image_prompt)[1]
    filename_grounding = f"examples/{timestamp}_grounding{file_ext}"
    return image, filename_grounding


def load_model(args):
    print(f"Loading model from {RANK} with world size {WORLD_SIZE}\n")
    model, model_args = CogVLMModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
            deepspeed=None,
            local_rank=LOCAL_RANK,
            rank=RANK,
            world_size=WORLD_SIZE,
            model_parallel_size=WORLD_SIZE,
            mode='inference',
            skip_init=True,
            use_gpu_initialization=True if torch.cuda.is_available() else False,
            device=f'cuda',
            **vars(args)
        ),
        overwrite_args={'model_parallel_size': WORLD_SIZE} if WORLD_SIZE != 1 else {}
    )
    model = model.eval()
    assert WORLD_SIZE == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"
    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model.image_length)

    print(f"Model loading finished from {RANK} with world size {WORLD_SIZE}\n.")
    return model, image_processor, text_processor_infer


## Parse args and load models
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
parser.add_argument("--english", action='store_true', help='only output English')
parser.add_argument("--version", type=str, default="chat", help='version to interact with')
parser.add_argument("--from_pretrained", type=str, default="cogvlm-chat", help='pretrained ckpt')
parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
parser.add_argument("--no_prompt", action='store_true', help='Sometimes there is no prompt in stage 1')
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--bf16", action="store_true")
parser = CogVLMModel.add_model_specific_args(parser)
args = parser.parse_args()
# Load models
model, image_processor, text_processor_infer = load_model(args)

print(f"model loading done on RANK={RANK}")
##

def call_predict(
        input_text,
        temperature,
        top_p,
        top_k,
        image_prompt,
        result_previous,
        hidden_image,
        is_grounding,
):
    parameters_str = f"""
        input params:
        input_text: {input_text}
        temperature: {temperature}
        top_p: {top_p}
        top_k: {top_k}
        image_prompt: {image_prompt}
        result_previous: {result_previous}
        hidden_image: {hidden_image}
    """

    print(
        parameters_str
    )
    result_text = [(ele[0], ele[1]) for ele in result_previous]
    for i in range(len(result_text) - 1, -1, -1):
        if result_text[i][0] == "" or result_text[i][0] == None:
            del result_text[i]
    print(f"history {result_text}")

    try:
        with torch.no_grad():
            pil_img, image_path_grounding = process_image_without_resize(image_prompt)

            # Pull all necessary data into list so we can broadcast them through `torch.distributed.broadcast_object_list`.
            if RANK == 0:
                image_prompts = [image_prompt]
                input_texts = [input_text]
                pil_imgs = [pil_img]
            else:
                image_prompts = [None]
                input_texts = [None]
                pil_imgs = [None]


            print("result_text: ", result_text)

            if WORLD_SIZE > 1:
                if RANK == 0:
                    torch.distributed.broadcast_object_list(input_texts, src=0)
                    if len(result_text) > 0:
                        torch.distributed.broadcast_object_list(result_text, src=0)
                    torch.distributed.broadcast_object_list(image_prompts, src=0)
                    torch.distributed.broadcast_object_list(pil_imgs, src=0)
                else:
                    print("Waiting for broadcasting from source device")

                print(f"image_prompts: {image_prompts}")
                print("result_texts:", result_text)
                print("input_texts:", input_texts)
                pil_img.save(f"temp_{RANK}.png")

            print(f"Calling chat from rank={RANK}")
            response, _, cache_image = chat(
                image_path=image_prompts[0],
                model=model,
                text_processor=text_processor_infer,
                img_processor=image_processor,
                query=input_texts[0],
                history=result_text,
                image=None,
                max_length=2048,
                top_p=top_p,
                temperature=temperature,
                top_k=top_k,
                invalid_slices=text_processor_infer.invalid_slices if hasattr(text_processor_infer, "invalid_slices") else [],
                no_prompt=False
            )
            print("chat call finished")
    except Exception as e:
        print("error message", e)
        traceback.print_exc()
        result_text.append((input_text, 'Timeout! Please wait a few minutes and retry.'))
        return "", result_text, hidden_image

    answer = response
    if is_grounding:
        parse_response(pil_img, answer, image_path_grounding)
        new_answer = answer.replace(input_text, "")
        result_text.append((input_text, new_answer))
        result_text.append((None, (image_path_grounding,)))
    else:
        result_text.append((input_text, answer))

    print(result_text)
    print('Chat task finished')
    return "", result_text, hidden_image



app = FastAPI()

@app.get("/")
def read_root():
    return {"Message": "Welcome to CogVLM FastAPI."}


@app.get("/inference/{item_id}")
def inference_image(item_id: int, q: Union[str, None] = None):
    is_grounding = 'grounding' in args.from_pretrained
    print(f"This rank: {RANK}, running inference on fastapi")
    input_text, result_text, hidden_image_hash = call_predict(
        input_text="Describe this image",
        temperature=0.7,
        top_p=0.4,
        top_k=10,
        image_prompt=f"./examples/{item_id}.jpg",
        result_previous=[['', 'Hi, What do you want to know about this image?']],
        hidden_image=None,
        is_grounding=is_grounding,
    )

    return {
        "input_text": input_text,
        "result_text": result_text,
    }

