#!/usr/bin/python
# -*- coding: utf-8 -*-

###############################
# storage
###############################

import logging
import boto3
from botocore.exceptions import ClientError
import io


def get_object_from_s3(bucket_name, object_name):
    """Retrieve an object from an Amazon S3 bucket

    :param bucket_name: string
    :param object_name: string
    :return: botocore.response.StreamingBody object. If error, return None.
    """

    # Retrieve the object

    s3 = boto3.client('s3')
    try:
        response = s3.get_object(Bucket=bucket_name, Key=object_name)
    except ClientError, e:

        # AllAccessDisabled error == bucket or object not found

        logging.error(e)
        return None

    # Return an open StreamingBody object

    return response['Body']


default_path = 'baskt-recommendation-system-data'

###############################
# dependences
###############################

## Install the latest Tensorflow version.

## Install TF-Hub.

###############################
# imports
###############################

from time import time
from datetime import timedelta
import os

import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics

from demjson import decode

import re
from collections import Counter
from itertools import combinations
from random import randint
from sklearn.externals import joblib

##########################
# Basic idea of the model
#
# 1. Load raw recipes.
# 2. Extract raw ingredients from recipes.
# 3. Convert raw ingredients strings to 512 dimensional float vector (known as embedings). Use Tf-Hub Universal Sentence Encoder for that.
# 4. Clusterize raw ingredients using number representation.
# 5. Construct similarity matrix for ingredient's clusters based on pairing in recipes. It is a core model.
# 6. Define method for looking for cluster for baskt.com product name
# 7. Create dictionary with cluster ids for baskt.com products
# 8. Define method for getting the nearest clusters for given baskt.com product name
# 9. Define method for getting the nearest baskt.com products for given baskt.com product name
#
###########################

###############################
# config (train / predict)
###############################

TRAIN_MODE = 'train'  # @param ["train", "predict"]


def main():
    start_total_time = time()

    # ##############################
    # 1. load recipes and process it
    # ##############################

    def try_json_convert(value):
        try:
            return json.loads(value.replace("'", '"'))
        except Exception, e:

    #     print("error for:", value)

            norm_json = decode(value)

    #     print(norm_json)

            return norm_json

        return json.loads('[]')

    if TRAIN_MODE == 'train':

        recipes_df = pd.read_csv(default_path + 'data/recipes_db.csv')

        print 'Num recipes: {}'.format(len(recipes_df.index))

        recipes_df['ingredients_json'] = recipes_df['ingredients'
                ].map(try_json_convert)

    # ##############################
    # 2. extract ingredients
    # ##############################

    if TRAIN_MODE == 'train':
        ingredients = np.array([])
        recipes = recipes_df['ingredients_json'].values

        for i in range(len(recipes)):
            ingredients = np.append(ingredients, recipes[i])
            ingredients = np.unique(ingredients, axis=0)

            if i % 1000 == 0:
                print 'extracted ingredients from {} recipes from {} total, {} ingredients found '.format(i,
                        len(ingredients))

        print 'extracted {} ingredients'.format(len(ingredients))

    # ##############################
    # 3. extract embedings for ingredients
    # ##############################

    module_url = \
        'https://tfhub.dev/google/universal-sentence-encoder-large/3'  # @param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

    # Import the Universal Sentence Encoder's TF Hub module

    embed = hub.Module(module_url)

    # Reduce logging output.

    tf.logging.set_verbosity(tf.logging.ERROR)

    BATCH_SIZE = 1000

    def get_embedings(strings):
        embeddings = np.array([])

        similarity_input_placeholder = tf.placeholder(tf.string,
                shape=None)
        similarity_message_encodings = \
            embed(similarity_input_placeholder)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())

            start_time = time()
            num_steps = int(len(strings) / BATCH_SIZE) + 1
            for i in range(num_steps):
                if i * BATCH_SIZE < len(strings):
                    embeddings_batch = \
                        session.run(similarity_message_encodings,
                                    feed_dict={similarity_input_placeholder: strings[i
                                    * BATCH_SIZE:(i + 1) * BATCH_SIZE]})

                    if len(embeddings) == 0:
                        embeddings = embeddings_batch
                    else:
                        embeddings = np.vstack((embeddings,
                                embeddings_batch))

                    elapsed_time = time() - start_time
                    print (
                        'step',
                        i,
                        'from',
                        num_steps,
                        ', elapsed time',
                        timedelta(seconds=elapsed_time),
                        )

        return embeddings

    if TRAIN_MODE == 'train':
        ingredients_embeddings = get_embedings(ingredients)

        print ('shape of found embeddings:',
               ingredients_embeddings.shape)

    # ##############################
    # 4. clusterization ingredients based on their embedings [K-Means]
    # ##############################

    if TRAIN_MODE == 'train':
        num_clusters = int(len(ingredients_embeddings) / 4)

        print ('num clusters:', num_clusters)

        start_time = time()
        kmeans = KMeans(n_clusters=num_clusters,
                        verbose=0).fit(ingredients_embeddings)

      # save the model

        joblib.dump(kmeans, default_path + 'model/kmeans.sav')

        ingredients_clusters = kmeans.predict(ingredients_embeddings)

        elapsed_time = time() - start_time
        print ('elapsed time:', timedelta(seconds=elapsed_time))
        print 'Found clusters for {} ingredients'.format(len(ingredients_clusters))

    # ##############################
    # 4.a prepare data from clusters and ingredients
    # ##############################

    if TRAIN_MODE == 'train':
        s1 = pd.Series(ingredients, name='ingredient')
        s2 = pd.Series(ingredients_clusters, name='cluster_id')

        clustered_ingredients_df = pd.concat([s1, s2], axis=1)

        def get_shortes(series):
            shortest = ''

            arr = series.values
            if len(arr) > 0:
                shortest = arr[0]
                for i in arr:
                    if len(shortest) > len(i):
                        shortest = i

            return shortest

        def get_longest(series):

            arr = series.values
            if len(arr) > 0:
                longest = arr[0]
                for i in arr:
                    if len(longest) < len(i):
                        longest = i

            return longest

        aggregations = {'ingredient': [get_shortes, get_longest]}

        cluster_names = clustered_ingredients_df.groupby('cluster_id'
                ).agg(aggregations)['ingredient']['get_shortes'].values

        print 'Found {} cluster names'.format(len(ingredients_clusters))

        np.save(default_path + 'model/cluster_names.npy', cluster_names)

    # ##############################
    # 5. similarity matrix for clusters
    # ##############################

    if TRAIN_MODE == 'train':

        recipes_with_clustered = []

        def convert_to_cluster(item):
            return clustered_ingredients_df.loc[clustered_ingredients_df['ingredient'
                    ] == item]['cluster_id'].values[0]

        print ('num clusters:', num_clusters)

        start_time = time()

        d = Counter()
        step = 0
        for recipe in recipes:

            recipe_w_c = np.array([convert_to_cluster(ingr) for ingr in
                                  recipe])

            recipes_with_clustered.append(recipe_w_c)

            if len(recipe_w_c) < 2:
                continue
            recipe_w_c.sort()
            for comb in combinations(recipe_w_c, 2):
                d[comb] += 1

            step = step + 1

            if step % 100 == 0:
                print 'step {} from {}'.format(step, len(recipes))

        pairs = np.array(d.most_common())

        elapsed_time = time() - start_time
        print ('elapsed time after pairs search:',
               timedelta(seconds=elapsed_time))

        print 'we have {} processed recipes'.format(len(recipes_with_clustered))
        print 'we have {} found pairs'.format(pairs.shape)

      # # pairs[?][0][0] - is X coordinate
      # # pairs[?][0][1] - is Y coordinate
      # # pairs[i][1] is value for matrix

        cluster_ids = kmeans.labels_

        similarity_matrix_for_clusters = np.zeros((len(cluster_ids),
                len(cluster_ids)))

        for i in range(len(pairs)):

      #   x_pos = np.where(cluster_ids == pairs[i][0][0])[0][0]
      #   y_pos = np.where(cluster_ids == pairs[i][0][1])[0][0]

            x_pos = pairs[i][0][0]
            y_pos = pairs[i][0][1]
            similarity_matrix_for_clusters.itemset((x_pos, y_pos),
                    pairs[i][1])
            similarity_matrix_for_clusters.itemset((y_pos, x_pos),
                    pairs[i][1])

        np.save(default_path
                + 'model/similarity_matrix_for_clusters.npy',
                similarity_matrix_for_clusters)

        elapsed_time = time() - start_time
        print ('elapsed time for similarity matrix at all:',
               timedelta(seconds=elapsed_time))

    # ##############################
    # 6. predict cluster for baskt.com product name function
    # ##############################

    def get_cluster(product_name):

        found_embeddings = get_embedings([product_name])

        found_cluster_id = kmeans.predict(found_embeddings)

        print 'for "{}" found cluster "{}" (id: {})'.format(product_name,
                cluster_names[found_cluster_id][0], found_cluster_id[0])

        return (found_cluster_id[0], cluster_names[found_cluster_id][0])

    #   clustered_ingredients_df.loc[clustered_ingredients_df['cluster_id'] == found_cluster_id[0]]

    # ##############################
    # 7. Create dictionary with cluster ids for baskt.com products
    # ##############################

    def clean_product_name(product_name):
        result = np.array([])

        s = product_name
        s = s.replace('-', ' ')
        s = s.replace("'", ' ')
        s = s.replace('"', ' ')
        s = s.replace(',', ' ')
        s = s.replace('.', ' ')
        s = s.replace(':', ' ')
        s = s.replace(';', ' ')
        s = s.replace('/', ' ')
        s = s.replace('%', ' ')
        s = s.replace('(', ' ')
        s = s.replace(')', ' ')
        s = s.replace('&', ' ')
        s = s.replace('#', ' ')
        s = s.replace('*', ' ')
        s = s.replace('!', ' ')
        s = s.replace('+', ' ')
        s = s.replace('?', ' ')
        s = s.replace('0', ' ')
        s = s.replace('1', ' ')
        s = s.replace('2', ' ')
        s = s.replace('3', ' ')
        s = s.replace('4', ' ')
        s = s.replace('5', ' ')
        s = s.replace('6', ' ')
        s = s.replace('7', ' ')
        s = s.replace('8', ' ')
        s = s.replace('9', ' ')

        words = s.split()

        for i in words:
            if len(i) >= 4:
                result = np.append(result, i)

        result = np.unique(result)

        return ' '.join(str(x) for x in result)

    if TRAIN_MODE == 'train':
        baskt_products_df = pd.read_json(default_path
                + 'data/items.json')
        baskt_products_df['category'] = \
            baskt_products_df['category_path'].map(lambda x: x.split('/'
                )[-1])
        baskt_products_df['clean_name'] = baskt_products_df['oc_name'
                ].map(clean_product_name)

        product_embeddings = \
            get_embedings(baskt_products_df['clean_name'].values)
        product_cluster_ids = kmeans.predict(product_embeddings)
        baskt_products_df['cluster_id'] = product_cluster_ids
        baskt_products_df['cluster_name'] = \
            baskt_products_df['cluster_id'].map(lambda x: \
                cluster_names[x])

        pd.DataFrame(baskt_products_df).to_csv(default_path
                + 'model/baskt_products.csv')

      # baskt_products_df.head(20)

    # ##############################
    # END OF TRAIN
    # ##############################

    # ##############################
    # load saved model
    # ##############################

    if TRAIN_MODE == 'predict':

      # load the model from disk

        kmeans = joblib.load(default_path + 'model/kmeans.sav')

        cluster_names = np.load(default_path + 'model/cluster_names.npy'
                                )

        similarity_matrix_for_clusters = np.load(default_path
                + 'model/similarity_matrix_for_clusters.npy')

        baskt_products_df = pd.read_csv(default_path
                + 'model/baskt_products.csv')

    # ###################

    elapsed_total_time = time() - start_total_time
    print ('TOTAL ELAPSED TIME:', timedelta(seconds=elapsed_total_time))

    # ###################

    # ##############################
    # 8. recommendation function for baskt.com product name function
    # ##############################

    def get_recommendations_for_cluster(cluster_id,
            num_recommendations):

        found_ids = np.where(similarity_matrix_for_clusters[cluster_id]
                             > 0)[0]
        found_occurrences = \
            np.take(similarity_matrix_for_clusters[cluster_id],
                    found_ids)

        selected_dict = {}
        for i in range(len(found_ids)):
            selected_dict[found_ids[i]] = found_occurrences[i]

        sorted_selected_dict = sorted(selected_dict.items(),
                key=lambda kv: kv[1], reverse=True)

    #   print(sorted_selected_dict)

        sorted_found_ids = [x[0] for x in sorted_selected_dict]

    #   print(cluster_names[sorted_found_ids].values)

        return (sorted_found_ids[:num_recommendations],
                (cluster_names[sorted_found_ids])[:num_recommendations])

    def get_recommendations_as_clusters(product_name,
            num_recommendations):

        (cluster_id, cluster_name) = get_cluster(product_name)

        print 'looking for recommendations for cluster "{}" (id: {})'.format(cluster_name,
                cluster_id)

        return get_recommendations_for_cluster(cluster_id,
                num_recommendations)

    # ##############################
    # 9. Define method for getting the nearest baskt.com products for given baskt.com product name
    # ##############################

    def get_recommendations(product_name, num_recommendations):

        (cluster_ids, cluster_names) = \
            get_recommendations_as_clusters(product_name,
                num_recommendations)

        print ('found closets clusters:', cluster_names)

        products = []
        result_df = pd.DataFrame([], columns=baskt_products_df.columns)
        for i in range(num_recommendations):
            result_df = \
                result_df.append(baskt_products_df[baskt_products_df['cluster_id'
                                 ] == cluster_ids[i]])
            products_for_cluster = \
                baskt_products_df[baskt_products_df['cluster_id']
                                  == cluster_ids[i]]['oc_name'].values

            if len(products_for_cluster) > 0:
                products.append(products_for_cluster[randint(0,
                                len(products_for_cluster) - 1)])

    #   return result_df['oc_name'].values[:num_recommendations]

        return (products, result_df)

    # product = 'cocoa'
    # product = "Cap'n Crunch Breakfast Cereal, 14 oz Box"

    product = baskt_products_df['oc_name'][randint(0,
            len(baskt_products_df.index) - 1)]

    print 'looking for recommendations for "{}"'.format(product)

    (resommendations, df) = get_recommendations(product, 10)

    print resommendations


if __name__ == '__main__':
    parser = \
        optparse.OptionParser(usage='python baskt_recommendation_system.py -t '
                              )
    parser.add_option('-t', '--type', action='store', dest='port',
                      help='Model call type.')
    (args, _) = parser.parse_args()
    print args

    main()
