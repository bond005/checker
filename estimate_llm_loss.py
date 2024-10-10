from argparse import ArgumentParser
import codecs
import json
import logging
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


RANDOM_SEED = 42
loss_calculation_logger = logging.getLogger(__name__)


def load_samples(fname: str) -> List[Tuple[List[Dict[str, str]], str]]:
    samples = []
    with codecs.open(fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        curline = fp.readline()
        while len(curline) > 0:
            prepline = curline.strip()
            if len(prepline) > 0:
                try:
                    new_sample = json.loads(prepline)
                    system_prompt = new_sample['system']
                    user_prompt = new_sample['query']
                    history = new_sample['history']
                    true_answer = new_sample['response']
                except Exception as err:
                    loss_calculation_logger.error(str(err))
                    raise
                new_input_message = [
                    {'role': 'system', 'content': system_prompt}
                ]
                if len(history) > 0:
                    for it in history:
                        new_input_message += [
                            {'role': 'user', 'content': it[0]},
                            {'role': 'assistant', 'content': it[1]}
                        ]
                new_input_message.append(
                    {'role': 'user', 'content': user_prompt}
                )
                samples.append((new_input_message, true_answer))
            curline = fp.readline()
    return samples


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not torch.cuda.is_available():
        err_msg = 'CUDA is not available!'
        loss_calculation_logger.error(err_msg)
        raise RuntimeError(err_msg)
    device = torch.device('cuda')
    torch.cuda.manual_seed(RANDOM_SEED)

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The input name of model for loss estimation.')
    parser.add_argument('-i', '--input', dest='input_dataset_name', type=str, required=True,
                        help='The input dataset name.')
    parser.add_argument('-o', '--output', dest='output_dataset_name', type=str, required=True,
                        help='The output dataset name.')
    args = parser.parse_args()

    model_name = os.path.normpath(args.model_name)
    if not os.path.isdir(model_name):
        err_msg = f'The directory "{model_name}" does not exist!'
        loss_calculation_logger.error(err_msg)
        raise IOError(err_msg)

    input_fname = os.path.normpath(args.input_dataset_name)
    if not os.path.isfile(input_fname):
        err_msg = f'The file "{input_fname}" does not exist!'
        loss_calculation_logger.error(err_msg)
        raise IOError(err_msg)

    output_fname = os.path.normpath(args.output_dataset_name)
    if not os.path.isfile(output_fname):
        base_dir = os.path.dirname(output_fname)
        if len(base_dir) > 0:
            if not os.path.isdir(base_dir):
                err_msg = f'The directory "{base_dir}" does not exist!'
                loss_calculation_logger.error(err_msg)
                raise IOError(err_msg)

    loaded_samples = load_samples(input_fname)
    loss_calculation_logger.info(f'There are {len(loaded_samples)} loaded samples in the {input_fname}.')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    ).to(device)

    with codecs.open(output_fname, mode='w', encoding='utf-8', errors='ignore') as fp:
        for input_messages, true_target in tqdm(loaded_samples):
            if len(input_messages) < 2:
                err_msg = (f'The input message sequence is too short! Expected 2 or greater, '
                           f'got {len(input_messages)}. {input_messages}')
                loss_calculation_logger.error(err_msg)
                raise RuntimeError(err_msg)
            if input_messages[0]['role'] != 'system':
                err_msg = f'The input message sequence is incorrect! {input_messages}'
                loss_calculation_logger.error(err_msg)
                raise RuntimeError(err_msg)
            if input_messages[-1]['user'] != 'system':
                err_msg = f'The input message sequence is incorrect! {input_messages}'
                loss_calculation_logger.error(err_msg)
                raise RuntimeError(err_msg)
            history = []
            if len(input_messages) > 2:
                if (len(input_messages) - 1) % 2 != 0:
                    err_msg = f'The input message sequence is incorrect! {input_messages}'
                    loss_calculation_logger.error(err_msg)
                    raise RuntimeError(err_msg)
                for idx in range((len(input_messages) - 1) // 2):
                    if input_messages[1 + idx * 2]['role'] != 'user':
                        err_msg = f'The input message sequence is incorrect! {input_messages}'
                        loss_calculation_logger.error(err_msg)
                        raise RuntimeError(err_msg)
                    if input_messages[2 + idx * 2]['role'] != 'assistant':
                        err_msg = f'The input message sequence is incorrect! {input_messages}'
                        loss_calculation_logger.error(err_msg)
                        raise RuntimeError(err_msg)
                    history.append([input_messages[1 + idx * 2]['content'], input_messages[2 + idx * 2]['content']])
            input_text = tokenizer.apply_chat_template(
                input_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([input_text], return_tensors=None)['input_ids'][0]
            full_text = tokenizer.apply_chat_template(
                input_messages + [{'role': 'assistant', 'content': true_target}],
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs_with_true_answer = tokenizer([full_text], return_tensors=None)['input_ids'][0]
            user_prompt_len = len(model_inputs)
            model_labels = [-100] * user_prompt_len + model_inputs_with_true_answer[user_prompt_len:]
            with torch.no_grad():
                res = model(
                    input_ids=torch.tensor([model_inputs_with_true_answer], dtype=torch.long).to(device),
                    labels=torch.tensor([model_labels], dtype=torch.long).to(device),
                    return_dict=True
                )
            loss_value = float(res.loss.cpu().numpy()[0])
            new_sample = {
                'system': input_messages[0]['content'],
                'query': input_messages[-1]['content'],
                'response': true_target,
                'history': history,
                f'{"_".join(os.path.basename(model_name).split())}_loss': loss_value
            }
            del history
            del model_inputs, model_inputs_with_true_answer
            del model_labels
            fp.write(json.dumps(obj=new_sample, ensure_ascii=False) + '\n')
            del new_sample


if __name__ == '__main__':
    loss_calculation_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    loss_calculation_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('loss_calculation.log')
    file_handler.setFormatter(formatter)
    loss_calculation_logger.addHandler(file_handler)
    main()