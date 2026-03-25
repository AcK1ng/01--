
import my_graph_capture
import time

def callback__for_req_issuing(t, contextlen, generated):
    print("At time {}, issue request with contextlen {} and generate {}".format(t, contextlen, generated))


dataset = my_graph_capture.AzureDataTrace("AzurePublicDataset/data/AzureLLMInferenceTrace_conv.csv",
                                          callback__for_req_issuing,
                                          2)

dataset.run()