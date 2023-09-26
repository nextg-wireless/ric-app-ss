# ==================================================================================
#       Copyright (c) 2018-2020 China Mobile Technology (USA) Inc. Intellectual Property.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# ==================================================================================
#FROM frolvlad/alpine-miniconda3:python3.7
FROM continuumio/miniconda3

# RMR setup

# RUN mkdir -p /opt/route/
# copy rmr files from builder image in lieu of an Alpine package
# COPY --from=nexus3.o-ran-sc.org:10002/o-ran-sc/bldr-alpine3-rmr:4.1.2 /usr/local/lib64/librmr* /usr/local/lib64/
# rmr_probe replaced health_ck
# COPY --from=nexus3.o-ran-sc.org:10002/o-ran-sc/bldr-alpine3-rmr:4.1.2 /usr/local/bin/rmr* /usr/local/bin/

# sdl needs gcc
#RUN apk update && apk add gcc musl-dev g++ jpeg-dev zlib-dev mesa-gl glib wget dpkg
RUN apt-get update && apt-get -y install build-essential musl-dev libjpeg-dev zlib1g-dev libgl1-mesa-dev wget dpkg

# RMR
ARG RMRVERSION=4.8.0
ARG RMRLIBURL=https://packagecloud.io/o-ran-sc/release/packages/debian/stretch/rmr_${RMRVERSION}_amd64.deb/download.deb
ARG RMRDEVURL=https://packagecloud.io/o-ran-sc/release/packages/debian/stretch/rmr-dev_${RMRVERSION}_amd64.deb/download.deb
RUN wget --content-disposition ${RMRLIBURL} && dpkg --force-architecture -i rmr_${RMRVERSION}_amd64.deb
RUN wget --content-disposition ${RMRDEVURL} && dpkg --force-architecture -i rmr-dev_${RMRVERSION}_amd64.deb
RUN rm -f rmr_${RMRVERSION}_amd64.deb rmr-dev_${RMRVERSION}_amd64.deb
ENV LD_LIBRARY_PATH /usr/local/lib/:/usr/local/lib64

# Install
COPY setup.py /tmp
COPY LICENSE.txt /tmp/
COPY ss/model/requirements.txt /tmp/ss/model/requirements.txt
#COPY init /tmp/init
# RUN unzip /tmp/ss/model.zip -d /tmp/ss && \
RUN pip install --upgrade pip && \ 
pip install /tmp && \
pip install -r /tmp/ss/model/requirements.txt

# Copy the code afterwards so we don't have to keep reinstalling pip libraries after every code change
COPY ss /tmp/ss
RUN pip install /tmp

RUN mkdir -p /opt/ric/config && chmod -R 755 /opt/ric/config
COPY init/ /opt/ric/config
ENV CONFIG_FILE=/opt/ric/config/config-file.json
#RUN sed -i 's/SERVICE_{}_{}_RMR_PORT/SERVICE_{}_{}_RMR_PORT_4560_TCP/g' /opt/conda/lib/python3.7/site-packages/ricxappframe/util/constants.py
#RUN sed -i 's/SERVICE_{}_{}_HTTP_PORT/SERVICE_{}_{}_RMR_PORT_8080_TCP/g' /opt/conda/lib/python3.7/site-packages/ricxappframe/util/constants.py
#ENV CONFIG_FILE config-file.json
#ENV CONFIG_FILE_PATH /tmp/init/

# TO DO: ADD RUN COMMANDS 
ENV PYTHONUNBUFFERED 1
CMD start-ss.py
