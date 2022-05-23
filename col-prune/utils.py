import re

from fuzzywuzzy import fuzz

NUM_MAPPING = {
  'half': 0.5,
  'one': 1,
  'two': 2,
  'three': 3,
  'four': 4,
  'five': 5,
  'six': 6,
  'seven': 7,
  'eight': 8,
  'nine': 9,
  'ten': 10,
  'eleven': 11,
  'twelve': 12,
  'twenty': 20,
  'thirty': 30,
  'once': 1,
  'twice': 2,
  'first': 1,
  'second': 2,
  'third': 3,
  'fourth': 4,
  'fifth': 5,
  'sixth': 6,
  'seventh': 7,
  'eighth': 8,
  'ninth': 9,
  'tenth': 10,
  'hundred': 100,
  'thousand': 1000,
  'million': 1000000,
  'jan': 1,
  'feb': 2,
  'mar': 3,
  'apr': 4,
  'may': 5,
  'jun': 6,
  'jul': 7,
  'aug': 8,
  'sep': 9,
  'oct': 10,
  'nov': 11,
  'dec': 12,
  'january': 1,
  'february': 2,
  'march': 3,
  'april': 4,
  'june': 6,
  'july': 7,
  'august': 8,
  'september': 9,
  'october': 10,
  'november': 11,
  'december': 12,
}

def parse_number(s):
  if s in NUM_MAPPING:
    return NUM_MAPPING[s]
  s = s.replace(',', '')
  # https://stackoverflow.com/questions/4289331/python-extract-numbers-from-a-string
  ret = re.findall(r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', s)
  if len(ret) > 0:
    return ret[0]
  return None


def get_cells(table):
  ret = set()
  for content in table["contents"][2:]:
    for col in content:
      if col["type"] == "TEXT":
        for x in col["data"]:
          ret.add((col["col"], str(x)))
      elif col["type"] == "LIST TEXT":
        for lst in col["data"]:
          for x in lst:
            ret.add((col["col"], str(x)))
  return ret

def best_match(candidates, query, col=None):
  return max(candidates, key=lambda x: (fuzz.ratio(x[1], query), col==x[0]))
