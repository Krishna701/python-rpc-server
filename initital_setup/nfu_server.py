# python -m grpc_tools.protoc -I ./protos --python_out=. --grpc_python_out=. ./protos/newsRecommendationSystem.proto

from concurrent import futures
import logging
import re

from transformers import ProphetNetConfig

import grpc
import newsRecommendationSystem_pb2 as pb2
import newsRecommendationSystem_pb2_grpc as pb2_grpc
#import run 
class NRSService(pb2_grpc.NRSServiceServicer):
    
    def send(self, request, context):
         # var title = request.title
         # var category = request.category
         # X = Engine(title, category)
         # return newsRecommendationSystem_pb2.NRSReply(X)
        print(request.title)
        print(request.category)
        return pb2.NRSReply(title = f'Title: {request.title}', category = f'Category: {request.category}')
        

#     # def returns(self):

def main():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers = 10)) #max_workers defines the total number of concurrent requests at a time
    pb2_grpc.add_NRSServiceServicer_to_server(NRSService(), server)
    print("Server Started! Listening on port: 50059")
    server.add_insecure_port('0.0.0.0:50059')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    main()

