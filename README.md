# ric-app-ss

This is the spectrum sensing xApp used in *SenseORAN: O-RAN based Radar Detection in the CBRS Band*.

The xApp is based on the Load Predictor xApp ([ric-app-lp](https://github.com/o-ran-sc/ric-app-lp)) provided by O-RAN Software Community.

The code is under the Apache 2.0 license, with the exception of the code contained in ss/model which uses YOLOv3, and thus is under a GPLv3 license.

This code must be used with a fork of srsRAN that supports the E2-like interface, which can be found here: https://github.com/openaicellular/srsRAN-e2/tree/e2like_support

Information on how to use E2-like xApps is provided on the Open AI Cellular website: https://openaicellular.github.io/oaic/xapp_python.html
