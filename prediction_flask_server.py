#!/usr/bin/env python3

from time import time
from datetime import timedelta
import json
import decimal
import os
import sys
import traceback
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import tensorflow as tf

from recipes.baskt_rs_recipes import BasktRecommendationSystemRecipes
from orders.baskt_rs_orders import BasktRecommendationSystemOrders
from favs.baskt_rs_favs import BasktRecommendationSystemFavs

from flask import Flask
from flask import request

import boto3
from botocore.exceptions import ClientError

app = Flask(__name__)


# Helper class to convert a DynamoDB item to JSON.
class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return str(o)
        return super(DecimalEncoder, self).default(o)


@app.route("/")
def hello():
    return "Flask prediction server is running"


@app.route("/get_recommendations")
def get_recommendations():
    try:
        upc = request.args.get('upc')
        num_recommendations = request.args.get('num_recommendations')
        full_info = request.args.get('full_info')

        if num_recommendations is None:
            num_recommendations = 10
        else:
            num_recommendations = int(num_recommendations)

        if full_info is None:
            return_full_info = True
        else:
            return_full_info = str(full_info) == 'true'

        if upc:
            predicted = {'recipes_based': [], 'favs_based': [], 'orders_based': []}
            predicted_full_info = {'recipes_based': [], 'favs_based': [], 'orders_based': []}
            product_name = get_product_name(upc)
            if product_name is not None:
                predicted['recipes_based'] = recipes_model.predict(product_name,
                                                                   num_recommendations,
                                                                   predict_type='product_ids')
            else:
                predicted['recipes_based'] = recipes_model.predict(str(upc),
                                                                   num_recommendations,
                                                                   predict_type='product_ids')
                predicted['error'] = "There are no products for upc '{}' at the db, check if you use valid upc. " \
                                     "Returned values consist of search result for '{}' as product name.".format(upc, upc)
                predicted_full_info['error'] = "There are no products for upc '{}' at the db, " \
                                               "check if you use valid upc. " \
                                               "Returned values consist of search result for '{}' " \
                                               "as product name.".format(upc, upc)
            predicted['favs_based'] = favs_model.predict(upc, num_recommendations)
            predicted['orders_based'] = orders_model.predict(upc, num_recommendations)
            if return_full_info:
                if len(predicted['recipes_based']) > 0:
                    predicted_full_info['recipes_based'] = get_products_full_info(predicted['recipes_based'])
                if len(predicted['favs_based']) > 0:
                    predicted_full_info['favs_based'] = get_products_full_info(predicted['favs_based'])
                if len(predicted['orders_based']) > 0:
                    predicted_full_info['orders_based'] = get_products_full_info(predicted['orders_based'])

                return json.dumps(predicted_full_info)
            else:
                return json.dumps(predicted)
        else:
            return "Request hasn't required param 'upc'"

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback, file=sys.stdout)
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        error = "Exception '{}' caught, details: {}".format(type(ex).__name__, str(ex))
        return json.dumps(error)


@app.route("/get_recommendations/recipes")
def get_recommendations_recipes():
    try:
        product_name = request.args.get('product_name')
        num_recommendations = request.args.get('num_recommendations')
        predict_type = request.args.get('predict_type')

        return_full_info = False

        if num_recommendations is None:
            num_recommendations = 10
        else:
            num_recommendations = int(num_recommendations)

        if predict_type is None:
            predict_type = 'product_ids'
            return_full_info = True

        if product_name:
            predicted = recipes_model.predict(product_name, num_recommendations, predict_type=predict_type)
            if return_full_info:
                if len(predicted) > 0:
                    predicted_full_info = get_products_full_info(predicted)
                    return json.dumps(predicted_full_info)
                else:
                    return json.dumps(predicted)
            else:
                return json.dumps(predicted)
        else:
            return "Request hasn't required param 'product_name'"

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback, file=sys.stdout)
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        error = "Exception '{}' caught, details: {}".format(type(ex).__name__, str(ex))
        return json.dumps(error)


@app.route("/get_recommendations/orders")
def get_recommendations_orders():
    try:
        upc = request.args.get('upc')
        num_recommendations = request.args.get('num_recommendations')
        full_info = request.args.get('full_info')

        if num_recommendations is None:
            num_recommendations = 10
        else:
            num_recommendations = int(num_recommendations)

        if full_info is None:
            return_full_info = True
        else:
            return_full_info = str(full_info) == 'true'

        if upc:
            predicted = orders_model.predict(upc, num_recommendations)
            if return_full_info:
                if len(predicted) > 0:
                    predicted_full_info = get_products_full_info(predicted)
                    return json.dumps(predicted_full_info)
                else:
                    return json.dumps(predicted)
            else:
                return json.dumps(predicted)
        else:
            return "Request hasn't required param 'upc'"

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback, file=sys.stdout)
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        error = "Exception '{}' caught, details: {}".format(type(ex).__name__, str(ex))
        return json.dumps(error)


@app.route("/get_recommendations/favs")
def get_recommendations_favs():
    try:
        upc = request.args.get('upc')
        num_recommendations = request.args.get('num_recommendations')
        full_info = request.args.get('full_info')

        if num_recommendations is None:
            num_recommendations = 10
        else:
            num_recommendations = int(num_recommendations)

        if full_info is None:
            return_full_info = True
        else:
            return_full_info = str(full_info) == 'true'

        if upc:
            predicted = favs_model.predict(upc, num_recommendations)
            if return_full_info:
                if len(predicted) > 0:
                    predicted_full_info = get_products_full_info(predicted)
                    return json.dumps(predicted_full_info)
                else:
                    return json.dumps(predicted)
            else:
                return json.dumps(predicted)
        else:
            return "Request hasn't required param 'upc'"

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback, file=sys.stdout)
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        error = "Exception '{}' caught, details: {}".format(type(ex).__name__, str(ex))
        return json.dumps(error)


def get_products_full_info(item_ids):
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


def get_product_name(upc):
    try:
        response = client.get_item(
            TableName='oc_product_description', Key={'upc': {'S': str(upc)}}
        )
        if 'Item' in response:
            return response['Item']['oc_name']['S']

    except ClientError as e:
        print(e.response['Error']['Message'])

    return None


if __name__ == "__main__":
    start_main_time = time()

    client = boto3.client('dynamodb', region_name='us-east-1')

    print("open tf session...")
    with tf.Session() as session:
        print("session opened")
        print("create and preparing models...")

        recipes_model = BasktRecommendationSystemRecipes()
        recipes_model.download_data_from_s3()
        recipes_model.prepare('predict', session)

        orders_model = BasktRecommendationSystemOrders()
        orders_model.download_data_from_s3()
        orders_model.prepare('predict')

        favs_model = BasktRecommendationSystemFavs()
        favs_model.download_data_from_s3()
        favs_model.prepare('predict')

        elapsed_main_time = time() - start_main_time
        print("models created and prepared, with elapsed time {}:".format(timedelta(seconds=elapsed_main_time)))

        app.run("0.0.0.0", 5001)