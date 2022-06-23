# FROM nginx:latest as builder
# RUN mkdir /usr/src/app
# WORKDIR /usr/src/app
# ENV PATH /usr/src/app/news_for_you/.bin:$PATH
# COPY . /usr/src/app/
# RUN  apt-get update -y
# RUN  apt-get install -y python3
# RUN apt-get -y install python3-pip
# RUN pip install -r requirements.txt 
# # RUN npm install react-scripts@1.1.1 -g --silent
# # COPY . /usr/src/app
# EXPOSE 50055
# CMD ["python","test_server"]

FROM python:3.9.7 as builder
RUN mkdir /usr/src/app
WORKDIR /usr/src/app
# ENV PATH /usr/src/app/server_setup/.bin:$PATH
COPY . /usr/src/app/
RUN pip install -r requirement.txt 
# COPY . /usr/src/app
EXPOSE 50059

CMD ["python","/usr/src/app/nfu_server.py"]

