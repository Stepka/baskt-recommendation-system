
from time import time
from datetime import timedelta
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import tensorflow as tf

from orders.baskt_rs_orders import BasktRecommendationSystemOrders

import boto3
from boto3.dynamodb.conditions import Key, Attr
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

  print("create and preparing model...")

  model = BasktRecommendationSystemOrders(verbose=0)
  model.download_data_from_s3()
  model.prepare('predict')

  elapsed_main_time = time() - start_main_time
  print("model created and prepared, with elapsed time {}:".format(timedelta(seconds=elapsed_main_time)))

  try:
    while True:

      product_upc = input("Enter product upc for recommendations: ")

      start_predict_time = time()

      predicted = model.predict(product_upc, 10)

      predicted_full_info = get_products_full_info(predicted)

      elapsed_predict_time = time() - start_predict_time
      print("model recommends: {}, \nelapsed time {}".format(predicted_full_info, timedelta(seconds=elapsed_predict_time)))

  except KeyboardInterrupt:
    print("\nexitting")