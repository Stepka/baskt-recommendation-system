
from time import time
from datetime import timedelta
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import tensorflow as tf

from recipes.baskt_rs_recipes import BasktRecommendationSystemRecipes

import boto3
from botocore.exceptions import ClientError


def get_products_full_info(item_ids):
  if len(item_ids) > 0:
    try:
      response = client.batch_get_item(
        RequestItems={
          'oc_product_description': {
            'Keys': [{'upc': {'S': id}} for id in item_ids],
            'ConsistentRead': True
          }
        }
      )
      return response['Responses']

    except ClientError as e:
      print(e.response['Error']['Message'])

  return []


if __name__ == '__main__':
  start_main_time = time()

  client = boto3.client('dynamodb', region_name='us-east-1')

  print("open tf session...")
  with tf.Session() as session:
    print("session opened")
    print("create and preparing model...")

    model = BasktRecommendationSystemRecipes()
    model.download_data_from_s3()
    model.prepare('predict', session)

    elapsed_main_time = time() - start_main_time
    print("model created and prepared, with elapsed time {}:".format(timedelta(seconds=elapsed_main_time)))

    try:
      while True:

        product_name = input("Enter product name or seacrh query for recommendations: ")

        start_predict_time = time()

        predicted = model.predict(product_name, 10, predict_type="product_ids")

        predicted_full_info = get_products_full_info(predicted)

        elapsed_predict_time = time() - start_predict_time
        print("model recommends: {}, \nelapsed time {}".format(predicted_full_info, timedelta(seconds=elapsed_predict_time)))

    except KeyboardInterrupt:
      print("\nexitting")