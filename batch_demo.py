import os, sys
import torch.distributed
import traceback

import config

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import json
import torch
import time
from sat.model.mixins import CachedAutoregressiveMixin
from sat.mpu import get_model_parallel_world_size

from utils.parser import parse_response
from utils.chat import chat
from models.cogvlm_model import CogVLMModel
from utils.language import llama2_tokenizer, llama2_text_processor_inference
from utils.vision import get_image_processor


def process_image_without_resize(image_prompt):
    image = Image.open(image_prompt)
    timestamp = int(time.time())
    file_ext = os.path.splitext(image_prompt)[1]
    filename_grounding = f"examples/{timestamp}_grounding{file_ext}"
    return image, filename_grounding


def load_model(args, rank, world_size):
    print(f"Loading model from {rank} with world size {world_size}\n")
    model, model_args = CogVLMModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
            deepspeed=None,
            local_rank=rank,
            rank=rank,
            world_size=world_size,
            model_parallel_size=world_size,
            mode='inference',
            skip_init=True,
            use_gpu_initialization=True if torch.cuda.is_available() else False,
            device=f'cuda',
            **vars(args)
        ),
        overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {}
    )
    model = model.eval()
    assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"
    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model.image_length)

    print(f"Model loading finished from {rank} with world size {world_size}\n.")
    return model, image_processor, text_processor_infer


def run_predict(args,
                model,
                image_processor,
                text_processor_infer,
                rank,
                input_text,
                temperature,
                top_p,
                top_k,
                image_prompt,
                ):
    is_grounding = 'grounding' in args.from_pretrained

    try:
        with torch.no_grad():
            pil_img, image_path_grounding = process_image_without_resize(image_prompt)

            # Pull all necessary data into list so we can broadcast them through `torch.distributed.broadcast_object_list`.
            if rank == 0:
                image_prompts = [image_prompt]
                input_texts = [input_text]
                pil_imgs = [pil_img]
            else:
                image_prompts = [None]
                input_texts = [None]
                pil_imgs = [None]

            if world_size > 1:
                torch.distributed.broadcast_object_list(input_texts, src=0)
                torch.distributed.broadcast_object_list(image_prompts, src=0)
                torch.distributed.broadcast_object_list(pil_imgs, src=0)

                print("image_prompts:", image_prompts)
                print("input_texts:", input_texts)
                pil_img.save(f"temp_{rank}.png")

            print(f"Calling chat from rank={rank}")
            response, _, cache_image = chat(
                image_path=image_prompts[0],
                model=model,
                text_processor=text_processor_infer,
                img_processor=image_processor,
                query=input_texts[0],
                history=[],
                image=None,
                max_length=2048,
                top_p=top_p,
                temperature=temperature,
                top_k=top_k,
                invalid_slices=text_processor_infer.invalid_slices if hasattr(text_processor_infer,
                                                                              "invalid_slices") else [],
                no_prompt=False
            )
            print("chat call finished")
    except Exception as e:
        print("error message", e)
        traceback.print_exc()

    answer = response
    if is_grounding:
        parse_response(pil_img, answer, image_path_grounding)
        new_answer = answer.replace(input_text, "")
    else:
        new_answer = answer

    return new_answer


if __name__ == '__main__':
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
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', None))
    print(f"local_rank: {local_rank}")
    parser = CogVLMModel.add_model_specific_args(parser)
    args = parser.parse_args()
    # Load models
    model, image_processor, text_processor_infer = load_model(args, rank, world_size)

    output_lines = []
    for i in range(1, 35):
        image_prompt = f"product/{i}-bg.png"
        answer = run_predict(args,
                             model,
                             image_processor,
                             text_processor_infer,
                             rank,
                             input_text="Could you describe this image in 300 tokens?",
                             temperature=config.MODEL_TEMP,
                             top_p=config.TOP_P,
                             top_k=config.TOP_K,
                             image_prompt=image_prompt,
                             )
        output_lines.append(dict(
            file_name=image_prompt,
            description=answer
        ))

    if rank == 0:
        with open("output.jsonl") as fp:
            for each in output_lines:
                fp.write(json.dumps(each) + "\n")
