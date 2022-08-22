import os
import pdb
import sys
import time
import logging
from typing import List, Dict

from classifier.classifier import BirdClassifier
from classifier.constants import IMAGE_URLS, MODEL_URL, LOG_FORMAT, TF_CPP_MIN_LOG_LEVEL, LABELS_URL
from classifier.models import BirdData
from utils.utils import load_model, download_url
from utils.birds_utils import load_and_parse_labels, order_birds_by_result_score, print_top_3_result_by_score

os.environ[TF_CPP_MIN_LOG_LEVEL] = '3'  # Turn of tensorflow logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger()


def print_results(index: int, sorted_birds_result: List[BirdData]) -> None:
    print('Run: %s' % int(index + 1))
    print_top_3_result_by_score(sorted_birds_result)


def process_image_urls(classifier: BirdClassifier, bird_labels_dict: Dict[int, BirdData], urls: List[str]) -> None:
    for i, url in enumerate(urls):
        logger.info(f"Processing image_url: {url}")
        url_raw_data = download_url(url).read()
        model_raw_result = classifier.process_image(url_raw_data)
        sorted_birds_result = order_birds_by_result_score(model_raw_result, bird_labels_dict)
        print_results(i, sorted_birds_result)


if __name__ == "__main__":
    start_time = time.time()
    logger.info("Started loading model")
    model = load_model(MODEL_URL)
    logger.info("Loading model finished")
    args_len = len(sys.argv)
    images_url = IMAGE_URLS
    if args_len > 1:
        images_url = sys.argv[1:]
    bird_lables_dict = load_and_parse_labels(LABELS_URL)
    classifier = BirdClassifier(model=model)
    process_image_urls(classifier, bird_lables_dict, images_url)
    print('Time spent: %s' % (time.time() - start_time))
