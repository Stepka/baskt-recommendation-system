from __future__ import print_function # Python 2/3 compatibility
import boto3
import json
import decimal
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError
import pandas as pd

# Helper class to convert a DynamoDB item to JSON.
class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return str(o)
        return super(DecimalEncoder, self).default(o)

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

# get categories
needed_categories = [
	'1030023f-6709-4ecb-8fb6-aa2ef1037a0b', #'Breakfast and Cereal', 
	'1b881ef9-9d69-419b-ab00-830128ebee53', #'Condiments and Sauces', 
	'2cb1a152-d13a-4f59-92ce-236cdab49bc1', #'Baking and Cooking Needs', 
	'2d0b4787-d13f-4226-9c72-0a37ce5820da', #'Canned Goods and Soups', 
	'470550f9-a287-4ac3-b12c-a774e6d50d83', #'Organic', 
	'66642c85-fc66-450a-8007-d7ca5b053795', #'Candy, Cookies and Snacks', 
	'8197a390-8e97-42a0-a740-e72ba7b97442', #'Grains, Pasta, Beans and Seeds', 
	'b7886c67-e050-4297-ada5-ffac3e3e92d5', #'Beverages', 
	'ee80e1e8-a514-4250-a6e8-70646c001b98', #'Bread and Bakery', 
	'f41b4b89-ee8f-41fe-a5aa-ef3f2a86f540', #'Gluten Free'
	]
	
item_ids = []
items = []
try:
	table = dynamodb.Table('oc_product_to_category')
	for category_id in needed_categories:
		response = table.scan(
			FilterExpression=Key('category_id').eq(category_id)
		)

		print("responce", len(response['Items']))
		for i in response['Items']:
			item_ids.append(i['upc'])
		
		print(len(item_ids))
		
	table = dynamodb.Table('oc_product_description')
	step = 0
	for item_id in item_ids:
		response = table.query(
			KeyConditionExpression=Key('upc').eq(item_id)
		)

		#print(item_id, "responce", len(response['Items']))
		try:
			for i in response['Items']:
				items.append({'oc_name': i['oc_name'], 'category_path': i['category_path']})
		except KeyError as key_error:
			print(i)
			
		step = step + 1
		
		if step%100 == 0 :
			print("step", step, "from", len(item_ids))
			
				
	print(len(items))
	df = pd.DataFrame(data = items)
	pd.DataFrame(df).to_json('items.json')
			
except ClientError as e:
    print(e.response['Error']['Message'])
	
	
