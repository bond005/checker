from argparse import ArgumentParser
import codecs
import csv
import json
import logging
import random
import sys

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import torch


loss_calculation_logger = logging.getLogger(__name__)


def main():
    pass


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