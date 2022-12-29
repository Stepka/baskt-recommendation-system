
from time import time
from datetime import timedelta

from baskt_rs_recipes import BasktRecommendationSystem

if __name__ == '__main__':
  start_main_time = time()

  model = BasktRecommendationSystem()

  model.download_data_from_s3()

  elapsed_main_time = time() - start_main_time
  print ("finished with time {}:".format(timedelta(seconds=elapsed_main_time)))