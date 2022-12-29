Recommendation System located at baskt-recommendation-system at the baskt.com ec2 instasnce.

###Content.

- **recipes/baskt_rs_recipes.py** - recommendations engine based on recipes.
- **favs/baskt_rs_favs.py** - recommendations engine based on favorites.
- **orders/baskt_rs_orders.py** - recommendations engine based on orders.


- **train_service_recipes.py** - service that runs train engine based on recipes. Just call this file from command prompt.
- **train_service_favs.py** - service that runs train engine based on favorites. Just call this file from command prompt.
- **train_service_orders.py** - service that runs train engine based on orders. Just call this file from command prompt.


- **prediction_flask_server.py** - example of using Recommendation System as server API.


- **recipes/model** - folder where trained model based on recipes stores.
- **favs/model** - folder where trained model based on favorites stores.
- **orders/model** - folder where trained model based on orders stores.


- **prediction_service_recipes.py** - example of using Recommendation Engine based on recipes interacting via keyboard input.
- **prediction_service_favs.py** - example of using Recommendation Engine based on favorites interacting via keyboard input.
- **prediction_service_orders.py** - example of using Recommendation Engine based on orders interacting via keyboard input.


- **recipes/data/recipes_db.csv** - dataset with scrapped recipes from yammly

###Showcase
###### Flask served Recommendation Engine.

For using Engine as endpoints I made simple Flask server with Recommendation Engine model running on.

To run a server launch *prediction_flask_server.py* 

Flask server will be run on 5001 port. 

Call following url to see recommendations:
http://ec2-18-214-220-221.compute-1.amazonaws.com:5001/get_recommendations?upc=030000560846
http://ec2-18-214-220-221.compute-1.amazonaws.com:5001/get_recommendations/favs?upc=030000560846
http://ec2-18-214-220-221.compute-1.amazonaws.com:5001/get_recommendations/orders?upc=030000560846
http://ec2-18-214-220-221.compute-1.amazonaws.com:5001/get_recommendations/recipes?product_name=potato 

###### Recommendations via command line:
For testing purposes and as example of using I made *prediction_service_recipes.py*, *prediction_service_favs.py*, *prediction_service_orders.py*. Just run it, wait until model be prepared and ask.

###Usage
First of all you need to train model. Run *train_service.py* for that. Looks like it is a good idea to run the script after new products will be added to database (but no more often then 1 time per day).

When Recommendation System training process have ends you can get recommendations. 

###### Case 1
You can create own *BasktRecommendationSystemRecipes*, *BasktRecommendationSystemOrders*, *BasktRecommendationSystemFavs* object and make direct calls to.

For this you need create *BasktRecommendationSystemRecipes*, *BasktRecommendationSystemOrders*, *BasktRecommendationSystemFavs*  object, prepare it and call predict() method. 

**Important!** *BasktRecommendationSystemRecipes* object should be created inside TensorFlow session: 

	with tf.Session() as session:    
		# create BasktRecommendationSystemRecipes object
		model = BasktRecommendationSystemRecipes(verbose=0)	
		# download and prepare model
		model.download_data_from_s3()
		model.prepare('predict', session)

		# get recommendations
		product_name = “product name”
		num_recommendations = 10
		predicted = model.predict(product_name, num_recommendations , predict_type="product_names")

For integration examples see *prediction_flask_server.py* or* prediction_service_recipes.py*, * prediction_service_orders.py*, * prediction_service_favs.py*


###### Case 2
You can use API calls to already written simple server. 

*prediction_flask_server.py* is a simple Flask server with API. Right now server is running on ec2 instance via *supervisor* (and shell script *start_server.sh*). Running on http://ec2-18-214-220-221.compute-1.amazonaws.com:5001. 

Has API methods:
- **/get_recommendations**
	- *upc* -  it should be *upc* of the product.
    - *num_recommendations* - num of products that system will try to return. System may return less products if it did not find the required number of products from the nearest clusters. Default is 10.
- **/get_recommendations/recipes**
	- *product_name* -  it can be product name or any string from search.
    - *num_recommendations* - num of products that system will try to return. System may return less products if it did not find the required number of products from the nearest clusters. Default is 10.
    - *predict_type* - type of return value. Can be one from “*product_ids*”, “*product_names*” or “*cluster_names*” strings. In case “*product_ids*” system will return products ucp, in case “*product_names*” will return full product names, and in case “*cluster_names*” system will return cluster names, extracted from yammly recipes. ***If omitted will be returned full info about products got from dynamoDB.***
- **/get_recommendations/favs**
	- *upc* -  it should be *upc* of the product.
    - *num_recommendations* - num of products that system will try to return. System may return less products if it did not find the required number of products from the nearest clusters. Default is 10.
- **/get_recommendations/orders**
	- *upc* -  it should be *upc* of the product.
    - *num_recommendations* - num of products that system will try to return. System may return less products if it did not find the required number of products from the nearest clusters. Default is 10.

###### Case 3
You can use written Lambda function named
`baskt-api-get-recommendations`

Function located in N. Virginia region. Function written for NodeJS. And can be merged to your NodeJS scripts. 

###Params

###### *BasktRecommendationSystemRecipes.predict()*  
method has the following params:
- ***product_name*** - it can be product name or any string from search.
- ***num_predictions*** - num of products that system will try to return. System may return less products if it did not find the required number of products from the nearest clusters.
- ***predict_type*** - type of return value. Can be one from “*product_ids*”, “*product_names*” or “*cluster_names*” strings. In case “*product_ids*” system will return products ucp, in case “*product_names*” will return full product names, and in case “*cluster_names*” system will return cluster names, extracted from yammly recipes.

###### *BasktRecommendationSystemFavs.predict()*  
method has the following params:
- ***upc*** -  it should be *upc* of the product.
- ***num_predictions*** - num of products that system will try to return. System may return less products if it did not find the required number of products from the nearest clusters.

###### *BasktRecommendationSystemOrders.predict()*  
method has the following params:
- ***upc*** -  it should be *upc* of the product.
- ***num_predictions*** - num of products that system will try to return. System may return less products if it did not find the required number of products from the nearest clusters.







