############################################################################
# SCRIPT DEVELOPED FOLLOWING THE LINK BELLOW:
# https://colab.research.google.com/github/tinyMLx/colabs/blob/master/3-4-3-PTQ.ipynb#scrollTo=WToIvvhzWhh2
############################################################################

import sys
sys.path.append("../../../tflite/")
import Model

def CamelCaseToSnakeCase(camel_case_input):
  """Converts an identifier in CamelCase to snake_case."""
  s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_case_input)
  return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

def FlatbufferToDict(fb, attribute_name=None):
  """Converts a hierarchy of FB objects into a nested dict."""
  if hasattr(fb, "__dict__"):
    result = {}
    for attribute_name in dir(fb):
      attribute = fb.__getattribute__(attribute_name)
      if not callable(attribute) and attribute_name[0] != "_":
        snake_name = CamelCaseToSnakeCase(attribute_name)
        result[snake_name] = FlatbufferToDict(attribute, snake_name)
    return result
  elif isinstance(fb, str):
    return fb
  elif attribute_name == "name" and fb is not None:
    result = ""
    for entry in fb:
      result += chr(FlatbufferToDict(entry))
    return result
  elif hasattr(fb, "__len__"):
    result = []
    for entry in fb:
      result.append(FlatbufferToDict(entry))
    return result
  else:
    return fb

def CreateDictFromFlatbuffer(buffer_data):
  model_obj = Model.Model.GetRootAsModel(buffer_data, 0)
  model = Model.ModelT.InitFromObj(model_obj)
  return FlatbufferToDict(model)

MODEL_FILE_NAME = './models/inception_v3_2015_2017_11_10/inceptionv3_non_slim_2015.tflite'

def main():
  file = open(MODEL_FILE_NAME, 'rb')
  model_data = file.read()
  file.close()

  model_dict = CreateDictFromFlatbuffer(model_data)
  pprint.pprint(model_dict['subgraphs'][0]['tensors'])

main()

#### end of file ####
