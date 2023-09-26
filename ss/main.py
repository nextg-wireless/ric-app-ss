# ==================================================================================
#       Copyright (c) 2020 China Mobile Technology (USA) Inc. Intellectual Property.
#       Copyright (c) 2022 NextG Wireless Lab Intellectual Property.
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
"""
ss entrypoint module
"""

ENABLE_E2_LIKE = True
PROTOCOL = 'SCTP'
ENABLE_DEBUG = False
OBJECT_SCORE_THRESHOLD = 1e-3
SWITCH_FREQUENCY = False

import schedule
#from zipfile import ZipFile
import json
from os import getenv
if not ENABLE_E2_LIKE:
    from ricxappframe.xapp_frame import RMRXapp, rmr, Xapp

import os, sys
import time
import torch

import sctp, socket
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # backend required for headless usage
import numpy as np

from PIL import Image
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

if __name__ != '__main__':
    sys.path.append('/tmp/ss')
    sys.path.append(os.getcwd()+'/ss')

from log import log_debug, log_info, log_warning, log_error

if not ENABLE_E2_LIKE:
    from constants import Constants
    from .manager import *
    from .handler import *

from model.models.common import DetectMultiBackend
from model.utils.general import check_img_size
from model.utils.datasets import LoadImages

xapp = None
pos = 0
cell_data = None
rmr_xapp = None
ai_model = None
sub_mgr = None
server = None

SPEC_SIZE = 614400

cmds = {
    'BASE_STATION_ON': b'y',
    'BASE_STATION_OFF': b'n',
    'BASE_STATION_SWITCH_FREQ': b'f',
    'BASE_STATION_REVERT_FREQ': b'r'
}

def post_init(self):
    global sub_mgr, server
    """
    Function that runs when xapp initialization is complete
    """
    #self.def_hand_called = 0
    #self.traffic_steering_requests = 0

    log_debug(self, "post_init")

    if not ENABLE_E2_LIKE:
        self.def_hand_called = 0
        self.traffic_steering_requests = 0
    
        sdl_mgr = SdlManager(self)
        sdl_mgr.sdlGetGnbList()
        a1_mgr = A1PolicyManager(self)
        a1_mgr.startup()
        sub_mgr = SubscriptionManager(self)
        
        enb_list = sub_mgr.get_enb_list()
        log_debug(self, f"enb list: {enb_list}")
        for enb in enb_list:
            log_debug(self, f"result: {sub_mgr.send_subscription_request(enb.inventory_name)}")
        
        gnb_list = sub_mgr.get_gnb_list()
        log_debug(self, f"gnb list: {gnb_list}")
        for gnb in gnb_list:
            sub_mgr.send_subscription_request(gnb.inventory_name)
        
        metric_mgr = MetricManager(self)
        metric_mgr.send_metric()

    else:
        ip_addr = socket.gethostbyname(socket.gethostname())
        #ip_addr = "169.254.32.111"
        #ip_addr = "0.0.0.0"
        log_info(self, f"E2-like enabled, connecting using {PROTOCOL} on {ip_addr}")
        if PROTOCOL == 'SCTP':
            server = sctp.sctpsocket_tcp(socket.AF_INET)
        elif PROTOCOL == 'TCP':
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((ip_addr, 5000))
        server.listen()

        log_info(self, 'Server started')


def entry(self):
    global current_iq_data, server
    """  Read from interface in an infinite loop and run prediction every second
      TODO: do training as needed in the future
    """
    post_init(self)
    if not ENABLE_E2_LIKE:
        schedule.every(1).seconds.do(run_prediction, self)
        while True:
            schedule.run_pending()
    else:
        while True:
            try:
                conn, addr = server.accept()
                #conn.setblocking(0)
                log_info(self, f'Connected by {addr}')
                initial = time.time()
                while True:
                    if initial - time.time() < 1.0:
                        conn.send(f"E2-like request from {PROTOCOL} server at {datetime.now().strftime('%H:%M:%S')}".encode('utf-8'))
                        log_info(self, "Sent E2-like request")
                    #data = conn.recv(SPEC_SIZE + 100)
                    data = conn.recv(12000)
                    if data:
                        log_info(self, f"Receiving I/Q data...")
                        while len(data) < SPEC_SIZE:
                            #print(f"Received I/Q data with size {len(data)}")
                            data += conn.recv(12000)
                    
                        recv_ts = time.time()
                        log_info(self, f"Received buffer size {len(data)} with ts {data[0:data.find(b'______')].decode() if data.find(b'______') >= 0 else 'not found'}, received at ts {recv_ts}")
                        log_info(self, f"Finished receiving message, processing")
                        
                        current_iq_data = data
                        result = run_prediction(self)

                        # every 10 seconds
                        if recv_ts % 10.0 < 0.4:
                            result = 'Radar'
                        elif recv_ts % 5.0 < 0.4:  # every 5 seconds in between
                            result = 'LTE'
                        else:
                            result = None

                        if result == 'Radar':
                            log_info(self, "Radar signal detected, sending control message to turn nodeB off")
                            conn.send(cmds['BASE_STATION_OFF'] if not SWITCH_FREQUENCY else cmds['BASE_STATION_SWITCH_FREQ'])
                        elif result in ('5G', 'LTE'):
                            log_info(self, "Radar signal no longer detected, sending control message to turn nodeB on")
                            conn.send(cmds['BASE_STATION_ON'] if not SWITCH_FREQUENCY else cmds['BASE_STATION_REVERT_FREQ'])

            except OSError as e:
                log_error(self, e)
        

def iq_to_spectrogram(iq_data, sampling_rate=5000) -> np.ndarray:
    """Convert I/Q data in 1-dimensional array into a spectrogram image as a numpy array
    """
    # The I/Q data is in [I,Q,I,Q,...] format
    # Each one is a 32-bit float so we can combine them easily by reading the array
    # as complex64 (made of two 32-bit floats)
    #complex_data = iq_data.view(np.complex64)
    complex_data = np.frombuffer(iq_data, dtype=np.complex64)
    # print(complex_data)
    
    # Create new matplotlib figure
    fig = plt.figure()

    # Create spectrogram from data
    plt.specgram(complex_data, Fs=sampling_rate)
    
    # Manually update the canvas
    fig.canvas.draw()

    w, h = [int(i) for i in fig.canvas.get_renderer().get_canvas_width_height()]
    #print(fig.canvas.tostring_rgb()[2000:3000])
    # Convert image to bytes, then read as a PIL image and return
    return np.array(Image.frombytes('RGB', (w, h), fig.canvas.tostring_rgb()))

def run_prediction(self):
    global sub_mgr, current_iq_data
    """Read the latest set of I/Q samples and run it by the model inference
    """
    # get i/q sample data
    start_time = time.perf_counter()
    sample = iq_to_spectrogram(current_iq_data)
    if ENABLE_DEBUG:
        log_debug(self, f"Total time for I/Q data conversion: {time.perf_counter() - start_time}")
    
    start_time = time.perf_counter()
    result = predict(self, sample)
    #log_debug(self,sub_mgr)
    if ENABLE_DEBUG:
        log_debug(self, f"Total time for prediction: {time.perf_counter() - start_time}")

    return result

def predict(self, data) -> str:
    inferences = predict_unseen_data(ai_model, data)
    if ENABLE_DEBUG:
        log_debug(self, f"data: {data}")
    
    # expected size is torch.Size([1, 63, 8])

    # Structure should be this, according these sources:
    # https://github.com/opencv/opencv/blob/4.x/samples/dnn/object_detection.cpp#L376-L447
    # https://towardsdatascience.com/yolo2-walkthrough-with-examples-e40452ca265f
    # rows are detected objects
    # column indices 0-3: center X, center Y, width, height
    # column index 4: object score (total confidence)
    # column index 5-7: confidence for each category (Radar, 5G, LTE)

    count = {key: [0, 0] for key in ai_model.names}

    for result in inferences:
        for row in result[0]:
            center_x, center_y, width, height, object_score = row[:5]
            confidences = {name: row[5+i] for i, name in enumerate(ai_model.names)}
            most_likely_classifier = max(confidences, key=lambda x: confidences[x])

            if object_score > 0.1:
                if ENABLE_DEBUG:
                    log_debug(self, f"Detected object with {object_score:.6f} confidence @ ({int(center_x)},{int(center_y)}), size {int(width)}x{int(height)}")
                    log_debug(self, f"Confidences: {confidences}")
                    log_debug(self, f"Highest confidence: {most_likely_classifier}")

                # detection count
                count[most_likely_classifier][0] += 1
                # sum of confidences
                count[most_likely_classifier][1] += confidences[most_likely_classifier]

    mean_confidence = {key: (count[key][1]/count[key][0] if count[key][0] else 0) for key in count}
    log_info(self, f"Mean confidence values: {mean_confidence}")

    most_likely_classifier = max(mean_confidence, key=lambda x: mean_confidence[x])

    return most_likely_classifier if mean_confidence[most_likely_classifier] > OBJECT_SCORE_THRESHOLD else None

def load_model_parameter():
    weights = '/tmp/ss/model/trained-on-dummy-data.pt'
    try:
        m = DetectMultiBackend(weights, device=torch.device('cpu'), dnn=False)
    except:
        weights = os.getcwd()+'/ss/model/trained-on-dummy-data.pt'
        m = DetectMultiBackend(weights, device=torch.device('cpu'), dnn=False)
    return m

def predict_unseen_data(model: DetectMultiBackend, unseen_data: np.ndarray) -> torch.Tensor:
    imgsz = check_img_size(640, s=model.stride)
    # dataset = LoadImages(unseen_data, img_size=imgsz, stride=model.stride, auto=model.pt and not model.jit)

    inferences = []

    #for path, im, im0s, vid_cap, s in dataset:

        # tensor = torch.Tensor(1, 3, 32, 32)

    im = torch.from_numpy(unseen_data).to(model.device)
    im = im.float() / 255
    # (640, 480, 3) -> (3, 640, 480)
    im = im.permute((2, 0, 1))

    # usually done in batches where batch is the first dimension,
    # if there is no batch, to conform to this tensor[None] will return
    # tensor with size of (1, 3, 32, 32)
    if len(im.shape) == 3:
        inference = model(im[None])
    else:
        inference = model(im)

    inferences.append(inference)

    # # unseen_data size is (32, 32, 3)
    # # expected input size is (3, 32, 32), so we need to transpose the axes
    # tensor = torch.from_numpy(unseen_data.transpose(2, 0, 1)).to(model.device).float()

    # # usually done in batches where batch is the first dimension,
    # # to conform to this tensor[None] will return tensor with size of (1, 3, 32, 32)
    # inference = model(tensor[None])
    
    return inferences


def start(thread=False):
    """
    This is a convenience function that allows this xapp to run in Docker
    for "real" (no thread, real SDL), but also easily modified for unit testing
    (e.g., use_fake_sdl). The defaults for this function are for the Dockerized xapp.
    """
    global xapp, ai_model
    print("XAPP START")
    fake_sdl = getenv("USE_FAKE_SDL", None)
    
    ai_model = load_model_parameter()
    if not ENABLE_E2_LIKE:
        xapp = Xapp(entrypoint=entry, rmr_port=4560, use_fake_sdl=fake_sdl)
        xapp.run()
    else:
        entry(None)


def stop():
    """
    can only be called if thread=True when started
    """
    xapp.stop()


if __name__ == '__main__':
    ai_model = load_model_parameter()
    entry(None)
