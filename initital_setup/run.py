from asyncio.windows_events import NULL
import imp

from nfu_server import *
# import module  
# import requests
# import json
from random import TfidfRecommender
import grpc
import newsRecommendationSystem_pb2 as pb2
import newsRecommendationSystem_pb2_grpc as pb2_grpc

class NRSService(pb2_grpc.NRSServiceServicer):

  def __init__(self, title, category):
    self.title = title
    self.category = category

  def send(title, category):
        # var title = request.title
        # var category = request.category
        # X = Engine(title, request)
        # return newsRecommendationSystem_pb2.NRSReply(X)
        
      return pb2.NRSReply(title = f'Title: {title}', category = f'Category: {category}')
  # objecs = send.__getattribute__()
    # def returns(self, ):

  def recommendation_function(self):
    pass  



object1 = NRSService.send("E-Kantipur", "Politics")
print(object1)

# request_uri = 
     
def recommendation_function(self):
  pass

