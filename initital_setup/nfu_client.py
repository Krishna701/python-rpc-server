from __future__ import print_function
from ctypes import addressof
import logging

import grpc
import newsRecommendationSystem_pb2
import newsRecommendationSystem_pb2_grpc

def run():
    with grpc.insecure_channel('localhost:50059') as channel:
        stub = newsRecommendationSystem_pb2_grpc.NRSServiceStub(channel)
        response = stub.send(newsRecommendationSystem_pb2.NRSRequest(name = 'Krishna', address = "50059"))
        print("Client: " + response.message)

if __name__ == '__main__':
    logging.basicConfig()
    run()