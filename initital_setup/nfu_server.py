from concurrent import futures
#from email import message
import logging

import grpc
import newsRecommendationSystem_pb2
import newsRecommendationSystem_pb2_grpc

class NRSService(newsRecommendationSystem_pb2_grpc.NRSServiceServicer):
    
    def send(self, request, context):
        return newsRecommendationSystem_pb2.NRSReply(message = f'Hello, {request.name}, received from address {request.address}')
  
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=100)) #max_workers defines the total number of concurrent requests at a time
    newsRecommendationSystem_pb2_grpc.add_NRSServiceServicer_to_server(NRSService(),server)
    print("Server Started! Listening on port: 50059")
    server.add_insecure_port('0.0.0.0:50059')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()

