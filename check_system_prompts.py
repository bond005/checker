from argparse import ArgumentParser
import codecs
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


def main():
    random.seed(RANDOM_SEED)

    parser = ArgumentParser()
    parser.add_argument('-d', '--data', dest='dataset_name', type=str, required=True,
                        help='The checked dataset name.')
    parser.add_argument('-p', '--prompts', dest='prompts_name', type=str, required=True,
                        help='The JSON file name with prompts and their synonyms.')
    args = parser.parse_args()

    dataset_fname = os.path.normpath(args.dataset_name)
    if not os.path.isfile(dataset_fname):
        err_msg = f'The file "{dataset_fname}" does not exist!'
        raise IOError(err_msg)

    prompts_fname = os.path.normpath(args.prompts_name)
    if not os.path.isfile(prompts_fname):
        err_msg = f'The file "{prompts_fname}" does not exist!'
        raise IOError(err_msg)

    prompt_ontology = load_ontology_for_prompts(prompts_fname)
    print(f'There are {len(prompt_ontology)} prompt types in the prompts ontology "{prompts_fname}".')

    dataset = load_samples(dataset_fname)
    print(f'There are {len(dataset)} samples in the dataset "{dataset_fname}".')

    frequencies_of_prompt_kinds = dict()
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
            err_msg = (f'The dataset {dataset_fname} contains a wrong sample {idx}. '
                       f'The system prompt "{system_prompt}" has an unknown type!')
            raise ValueError(err_msg)
        if found_prompt_type not in frequencies_of_prompt_kinds:
            frequencies_of_prompt_kinds[found_prompt_type] = 0
        frequencies_of_prompt_kinds[found_prompt_type] += 1
    print('All right!')

    max_text_width = max([len(k) for k in frequencies_of_prompt_kinds])
    max_number_width = max([len(str(frequencies_of_prompt_kinds[k])) for k in frequencies_of_prompt_kinds])
    print('')
    print('Frequencies of prompt kinds:')
    for it in sorted(list(frequencies_of_prompt_kinds.keys()), key=lambda x: -frequencies_of_prompt_kinds[x]):
        print('  {0:>{1}}  {2:>{3}}'.format(it, max_text_width,
                                            frequencies_of_prompt_kinds[it], max_number_width))


if __name__ == '__main__':
    main()
