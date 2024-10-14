from argparse import ArgumentParser
import codecs
import copy
import json
import os
import random
from typing import Dict, List, Tuple


RANDOM_SEED = 42


def load_samples(fname: str) -> List[Tuple[str, List[Tuple[str, str]]]]:
    samples = []
    with codecs.open(fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        curline = fp.readline()
        while len(curline) > 0:
            prepline = curline.strip()
            if len(prepline) > 0:
                new_sample = json.loads(prepline)
                system_prompt = new_sample['system']
                additional_keys = set(new_sample.keys()) - {'system'}
                additional_data = []
                if len(additional_keys) > 0:
                    for k in additional_keys:
                        additional_data.append((k, new_sample[k]))
                samples.append((system_prompt, additional_data))
                del additional_keys, additional_data
            curline = fp.readline()
    return samples


def load_ontology_for_prompts(fname: str) -> Dict[str, List[str]]:
    with codecs.open(fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        err_msg = f'{fname} contains a wrong data! Expected {type({"a": "b", "c": "d"})}, got {type(data)}.'
        raise IOError(err_msg)
    prompts = sorted(list(data.keys()))
    if len(prompts) < 1:
        err_msg = f'{fname} contains a wrong data! A prompt dictionary is empty!'
        raise IOError(err_msg)
    for cur_prompt in prompts:
        if not isinstance(data[cur_prompt], list):
            err_msg = (f'{fname} contains a wrong data for the prompt "{cur_prompt}"! '
                       f'Expected {type(["a", "b"])}, got {type(data[cur_prompt])}.')
            raise IOError(err_msg)
        if len(data[cur_prompt]) < 1:
            err_msg = f'{fname} contains a wrong data! The variant list for the prompt "{cur_prompt}" is empty!'
            raise IOError(err_msg)
        if len(set(data[cur_prompt])) != len(data[cur_prompt]):
            err_msg = (f'{fname} contains a wrong data! '
                       f'The variant list for the prompt "{cur_prompt}" contains duplicates!')
            raise IOError(err_msg)
    return data


def load_augmentation_config(fname: str) -> Dict[str, float]:
    with codecs.open(fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        err_msg = f'{fname} contains a wrong data! Expected {type({"a": 1.0, "c": 2.0})}, got {type(data)}.'
        raise IOError(err_msg)
    prompts = sorted(list(data.keys()))
    if len(prompts) < 1:
        err_msg = f'{fname} contains a wrong data! A prompt dictionary is empty!'
        raise IOError(err_msg)
    for cur_prompt in prompts:
        if not isinstance(data[cur_prompt], float):
            err_msg = (f'{fname} contains a wrong data for the prompt "{cur_prompt}"! '
                       f'Expected {type(1.3)}, got {type(data[cur_prompt])}.')
            raise IOError(err_msg)
        distortion_factor = data[cur_prompt]
        if distortion_factor <= 0.0:
            err_msg = (f'{fname} contains a wrong distortion factor for the prompt "{cur_prompt}"! '
                       f'Expected a positive value, got {data[cur_prompt]}.')
            raise IOError(err_msg)
    return data


def main():
    random.seed(RANDOM_SEED)

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_dataset_name', type=str, required=True,
                        help='The input dataset name.')
    parser.add_argument('-p', '--prompts', dest='prompts_name', type=str, required=True,
                        help='The JSON file name with prompts and their synonyms.')
    parser.add_argument('-o', '--output', dest='output_dataset_name', type=str, required=True,
                        help='The output dataset name after synonymizing.')
    parser.add_argument('-c', '--config', dest='augmentation_config', type=str, required=False,
                        default=None, help='The JSON file name with augmentation config.')

    args = parser.parse_args()

    input_dataset_fname = os.path.normpath(args.input_dataset_name)
    if not os.path.isfile(input_dataset_fname):
        err_msg = f'The file "{input_dataset_fname}" does not exist!'
        raise IOError(err_msg)

    prompts_fname = os.path.normpath(args.prompts_name)
    if not os.path.isfile(prompts_fname):
        err_msg = f'The file "{prompts_fname}" does not exist!'
        raise IOError(err_msg)

    if args.augmentation_config is None:
        augmentation_fname = ''
    else:
        augmentation_fname = os.path.normpath(args.augmentation_config)
        if not os.path.isfile(augmentation_fname):
            err_msg = f'The file "{augmentation_fname}" does not exist!'
            raise IOError(err_msg)

    output_dataset_fname = os.path.normpath(args.output_dataset_name)
    if not os.path.isfile(output_dataset_fname):
        basedir = os.path.dirname(output_dataset_fname)
        if len(basedir) > 0:
            if not os.path.isdir(basedir):
                err_msg = f'The directory "{basedir}" does not exist!'
                raise IOError(err_msg)

    if os.path.basename(output_dataset_fname) == os.path.basename(input_dataset_fname):
        raise IOError('The input dataset name and the output dataset name are same!')

    prompt_ontology = load_ontology_for_prompts(prompts_fname)
    print(f'There are {len(prompt_ontology)} prompt types in the prompts ontology "{prompts_fname}".')

    dataset = load_samples(input_dataset_fname)
    print(f'There are {len(dataset)} samples in the dataset "{input_dataset_fname}".')

    if len(augmentation_fname) > 0:
        augmentation_config = load_augmentation_config(augmentation_fname)
        max_text_width = max([len(k) for k in augmentation_config])
        print('')
        print('Augmentation config:')
        for it in augmentation_config:
            print('    {0:>{1}}: {2:.4f}'.format(it, max_text_width, augmentation_config[it]))
        print('')
        for it in sorted(list(augmentation_config.keys())):
            if it not in prompt_ontology:
                del augmentation_config[it]
    else:
        augmentation_config = dict()

    dataset_by_prompt_kinds = dict()
    for idx, val in enumerate(dataset):
        system_prompt = val[0]
        found_prompt_type = ''
        if system_prompt in prompt_ontology:
            found_prompt_type = system_prompt
        else:
            for prompt_type in prompt_ontology:
                if system_prompt in set(prompt_ontology[prompt_type]):
                    found_prompt_type = prompt_type
                    break
        if len(found_prompt_type) == 0:
            err_msg = (f'The dataset {input_dataset_fname} contains a wrong sample {idx}. '
                       f'The system prompt "{system_prompt}" has an unknown type!')
            raise ValueError(err_msg)
        if found_prompt_type not in dataset_by_prompt_kinds:
            dataset_by_prompt_kinds[found_prompt_type] = []
        dataset_by_prompt_kinds[found_prompt_type].append(val[1])
    print(f'All samples are grouped by {len(dataset_by_prompt_kinds)} prompt types.')

    with codecs.open(output_dataset_fname, mode='w', encoding='utf-8', errors='ignore') as fp:
        for prompt_type in prompt_ontology:
            old_size = len(dataset_by_prompt_kinds[prompt_type])
            if prompt_type in augmentation_config:
                new_size = max(10, round(old_size * augmentation_config[prompt_type]))
            else:
                new_size = old_size
            if old_size == new_size:
                samples = copy.deepcopy(dataset_by_prompt_kinds[prompt_type])
            elif new_size < old_size:
                samples = random.sample(population=dataset_by_prompt_kinds[prompt_type], k=new_size)
            else:
                samples = copy.deepcopy(dataset_by_prompt_kinds[prompt_type])
                while (new_size - len(samples)) >= old_size:
                    samples += dataset_by_prompt_kinds[prompt_type]
                if (new_size - len(samples)) > 0:
                    samples += random.sample(population=dataset_by_prompt_kinds[prompt_type],
                                             k=new_size - len(samples))
            for old_sample in samples:
                new_sample = [('system', random.choice(prompt_ontology[prompt_type]))] + old_sample
                fp.write(json.dumps(dict(new_sample), ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
