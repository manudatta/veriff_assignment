import heapq
from heapq import heapify
from io import StringIO
from typing import List, Tuple, Dict

import numpy
import numpy as np

from classifier.models import BirdData
from utils.utils import download_url


def clean_up_raw_data(data: StringIO) -> List[str]:
    bird_labels_lines = [line.decode('utf-8').replace('\n', '') for line in data.readlines()]
    bird_labels_lines.pop(0)  # remove header (id, name)
    return bird_labels_lines


def get_id_and_name(line: str) -> Tuple[int, str]:
    bird_id = int(line.split(',')[0])
    bird_name = line.split(',')[1]
    return (bird_id, bird_name)


def load_and_parse_labels(url: str) -> Dict[int, BirdData]:
    data = download_url(url)
    bird_labels_lines = clean_up_raw_data(data)
    birds = {}
    for bird_line in bird_labels_lines:
        bird_id, bird_name = get_id_and_name(bird_line)
        birds[bird_id] = BirdData(**{'name': bird_name, 'id': bird_id})
    return birds


def order_birds_by_result_score(model_raw_output: numpy.ndarray, bird_labels: List[BirdData]) -> List[BirdData]:
    for index, value in np.ndenumerate(model_raw_output):
        bird_index = index[1]
        bird_labels[bird_index].score = value
    bird_labels_list = list(bird_labels.values())
    heapify(list(bird_labels_list))
    return bird_labels_list


def print_top_3_result_by_score(birds_names_with_results_ordered: List[BirdData]) -> None:
    top_3_birds = heapq.nlargest(3, birds_names_with_results_ordered)
    bird = top_3_birds[0]
    print('Top match: "%s" with score: %s' % (bird.name, bird.score))
    bird = top_3_birds[1]
    print('Second match: "%s" with score: %s' % (bird.name, bird.score))
    bird = top_3_birds[2]
    print('Third match: "%s" with score: %s' % (bird.name, bird.score))
    print('\n')
