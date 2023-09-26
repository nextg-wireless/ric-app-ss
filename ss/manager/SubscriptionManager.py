# ==================================================================================
#
#       Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved.
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
#
# ==================================================================================
"""

"""

import requests
from ricxappframe.xapp_frame import RMRXapp
#import ricxappframe.xapp_subscribe as subscribe
#from ricxappframe.subsclient.models.subscription_params import SubscriptionParams
import json
from ..constants import Constants
from ._BaseManager import _BaseManager

class SubscriptionManager(_BaseManager):

    __namespace = "e2Manager"

    def __init__(self, rmr_xapp: RMRXapp):
        super().__init__(rmr_xapp)
        self.url = Constants.SUBSCRIPTION_PATH.format(Constants.PLT_NAMESPACE,
                                                 Constants.SUBSCRIPTION_SERVICE,
                                                 Constants.SUBSCRIPTION_PORT)
        self.subscriber = NewSubscriber(url)

    def get_gnb_list(self):
        gnblist = self._rmr_xapp.get_list_gnb_ids()
        self.logger.info("SubscriptionManager.getGnbList:: Processed request: {}".format(gnblist))
        return gnblist

    def get_enb_list(self):
        enblist = self._rmr_xapp.get_list_enb_ids()
        self.logger.info("SubscriptionManager.sdlGetGnbList:: Handler processed request: {}".format(enblist))
        return enblist

    def send_subscription_request(self,xnb_id):
        #subscription_request = {"xnb_id": xnb_id, "action_type": Constants.ACTION_TYPE}
#        try:
        #json_object = json.dumps(subscription_request,indent=4)
        #print(json_object)
#        except TypeError:
#            print("Unable to serialize the object")
        
        # setup the subscription data
        # taken from xapp_subscribe example and idk how these values are chosen

        # host, http_port, rmr_port
        subEndpoint = subscriber.SubscriptionParamsClientEndpoint("localhost", 8080, 4061)
        # e2_timeout_timer_value, e2_retry_count, rmr_routing_needed
        subsDirective = subscriber.SubscriptionParamsE2SubscriptionDirectives(10, 2, False)
        # subsequent_action_type ("continue" or "wait"), time_to_wait ("zero", "w1ms", ... , "w60s")
        subsequentAction = subscriber.SubsequentAction("continue", "w10ms")
        # action_id (0-255), action_type ("insert", "policy" or "report"), action_definition, subsequent_action
        actionDefinitionList = subscriber.ActionToBeSetup(1, "policy", (11,12,13,14,15), subsequentAction)
        # xapp_event_instance_id (0-65535), event_triggers, action_to_be_setup_list
        subsDetail = subscriber.SubscriptionDetail(12110, (1,2,3,4,5), actionDefinitionList)

        sub_params = SubscriptionParams(
            subscription_id="sub727",
            client_endpoint=subEndpoint,
            meid=xnb_id,
            ran_function_id=33,  # for REPORT, though example uses 1231 for policy??
            e2_subscription_directives=subsDirective,
            subscription_details=subsDetail
        )

        #url = Constants.SUBSCRIPTION_PATH.format(Constants.PLT_NAMESPACE,
        #                                         Constants.SUBSCRIPTION_SERVICE,
        #                                         Constants.SUBSCRIPTION_PORT)
        print(self.url)
        data, reason, status = self.subscriber.Subscribe(sub_params)
        
        if status != 200:
            self.logger.error(f"Subscription failed: {data}, {reason}, {status}")

        return data, reason, status

        #try:
        #    response = requests.post(url , json=json_object)
        #3    print(url)
        #    print(response)
        #    print(response.status_code)
        #    response.raise_for_status()
        #except requests.exceptions.HTTPError as err_h:
        #    return "An Http Error occurred:" + repr(err_h)
        #except requests.exceptions.ConnectionError as err_c:
        #    return "An Error Connecting to the API occurred:" + repr(err_c)
        #except requests.exceptions.Timeout as err_t:
        #    return "A Timeout Error occurred:" + repr(err_t)
        #except requests.exceptions.RequestException as err:
            



