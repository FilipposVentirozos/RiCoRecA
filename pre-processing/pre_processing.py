import json
import xmltodict
import os
import spacy
import re
import copy
import random
# import paths
import pprint
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
from spacy.util import compile_infix_regex
from spacy.lang.en import English
from collections import defaultdict

nlp = spacy.load("en_core_web_trf")
nlp_sent = English()
nlp_sent.add_pipe("sentencizer")

# def update_all_revs():
#     """ Update all reviews under one file
#
#     :return:
#     """
#     raw = os.path.join(paths.data, "raw")
#     all_revs = dict()
#     for f in os.listdir(raw):
#         if os.path.isfile(os.path.join(raw, f)) and ".json" in f:
#             with open(os.path.join(raw, f), "r") as fp:
#                 reviews = json.load(fp)
#                 all_revs[reviews[0]["asin"]["original"]] = reviews
#                 # pprint.pprint(reviews)
#
#     intermediate = os.path.join(paths.data, "intermediate")
#     with open(os.path.join(intermediate, "revs.json"), "w") as fp:
#         json.dump(all_revs, fp, indent=4)

def fractions_2_float(text):
    """ Convert fractions into string floats. For instance '1/2' will be '0.5'.
    Should be used after split_mixed().
    Update: Added to take quotes into account.
    Update: Added to take some punctuation into account.
    Update: Consider start and ending of string.
    !!! To be executed after the split_mixed()
    Unit Test
    text = "2/3 hello32 21gew fewfw 2/3 32/2 2/43 1/2 /23 :23/2 232/ 2/2/3 321 3/16\" 2/12' ,3/4  12/3d 23/3"
    text = 0.667 hello32 21gew fewfw 0.667 16.000 0.047 0.500 /23 :11.500 232/ 2/2/3 321 0.188" 0.167'
    ,0.750  12/3d 7.667
     ,0.7504  12/3d 7.6673"
    :param text:
    :return:
    """
    pattern = re.compile(r'(^|[\s,.:;])\d+/\d+(?=[\s\'",.:;\)]|$)', re.UNICODE)
    match = re.search(pattern, text)
    while match:
        try:
            out = '%.3f' % (int(match.group().split('/')[0][1:]) / int(match.group().split('/')[1]))
            text = text[:(match.start() + 1)] + out + text[match.end():]
        except ValueError:  # ValueError: invalid literal for int() with base 10: '' if start of String
            out = '%.3f' % (int(match.group().split('/')[0]) / int(match.group().split('/')[1]))
            text = out + text[match.end():]
        except ZeroDivisionError:
            out = str(int(match.group().split('/')[0][1:]))
            text = text[:(match.start() + 1)] + out + text[match.end():]
        match = re.search(pattern, text)
    return text


def custom_tokenizer(nlp):
    infixes = tuple([r"\)", r"\("]) + tuple(nlp.Defaults.infixes)
    inf = list(infixes)               # Default infixes
    inf.remove(r"(?<=[0-9])[+\-\*^](?=[0-9-])")    # Remove the generic op between numbers or between a number and a -
    inf = tuple(inf)                               # Convert inf to tuple
    # Add the removed rule after subtracting (?<=[0-9])-(?=[0-9]) pattern
    infixes = inf + tuple([r"(?<=[0-9])[+*^](?=[0-9-])", r"(?<=[0-9])-(?=-)"])
    infixes = [x for x in infixes if '-|–|—|--|---|——|~' not in x]  # Remove - between letters rule
    infixes = [x for x in infixes if '/' not in x]  # Remove - between letters rule

    infix_re = compile_infix_regex(infixes)

    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                suffix_search=nlp.tokenizer.suffix_search,
                                infix_finditer=infix_re.finditer,
                                token_match=nlp.tokenizer.token_match,
                                rules=nlp.Defaults.tokenizer_exceptions)


def print_for_Prodigy():
    with open("/home/chroner/PhD_remote/FoodBase/data/Intermediate/FoodBase/FoodBase_curated_new.xml", "r") as fp:
        d = xmltodict.parse(fp.read())
    d_out = list()
    for recipe in d["collection"]["document"]:
        text = recipe["infon"][1]["#text"]
        # I omit the below three since now they are labelled as one
        text = re.sub(r"(?<=\d)-(?=[[a-zA-Z])", ' ', text)  # 9x13-inch will be 9x13 inch, to separate Unit and Value
        text = re.sub(r"(?<=\d)(?=[[FC])", ' ', text)  # Split the 425F to 425 F, to separate Unit and Value
        text = fractions_2_float(text)  # Downstream tokenisation
        # We keep the above tokenisation for better downstream modelling

        # text = re.sub(r"(?<=[a-zA-Z])-(?=[[a-zA-Z])", '', text)  # Join hyphen phrases
        # text = re.sub(r"(?<=\d)-(?=\d)", '', text)  # Join hyphen phrases
        # text = re.sub(r"(?<=[a-zA-Z])/(?=[[a-zA-Z])", '', text)
        # text = re.sub(r"(?<=\S)\((?=\S)", ' ( ', text)  # spaCy not tokenizing parenthesis
        # text = re.sub(r"(?<=\S)\)(?=\S)", ' ) ', text)
        # text = text.replace("/", "")  # Replace this with numeric

        # text = text.replace("'", " ")  # Removed recently
        # It seems that lone degrees gets confusion
        # text = re.sub(r"\s[C,F,c,f][\s.,!;:)(]", '', text)
        # text = text.replace("'", "")
        # doc = re.findall(r"[\w']+|[.,!?;:)(]", text)

        doc = nlp(text)
        d_entry = {"text": doc.text}
        doc_sent = nlp_sent(doc.text)
        sentences = list()
        for sent in doc_sent.sents:
            sentences.append(sent.text)
        spans_d = list()
        meta_anno = list()
        # Verb and ADV annotation
        for token in doc:
            if token.pos_ == "VERB":
                d_anno = {"start": token.idx, "end": token.idx+len(token.text),  # "token_start": token_start, "token_end": token_end,
                          "label": "ACTION"}  # annotation["infon"]["#text"]
                spans_d.append(d_anno)
            elif token.pos_ == "ADV":
                d_anno = {"start": token.idx, "end": token.idx+len(token.text),  # "token_start": token_start, "token_end": token_end,
                          "label": "HOW"}  # annotation["infon"]["#text"]
                spans_d.append(d_anno)
        try:
            for annotation in recipe["annotation"]:
                for i in [0, 1, 2]:
                    try:
                        id_ = int(annotation["location"]["@offset"]) - 1
                    except TypeError:
                        # Catch Parsing error when there is only one ingredient
                        annotation = recipe["annotation"]
                        id_ = int(annotation["location"]["@offset"]) - 1
                    id_ += i
                    length = len(annotation["text"])
                    span = ""
                    # token_start = copy.copy(id_)
                    start_id = copy.copy(doc[id_].idx)
                    while (length-1) > 0:  # Insert the labels as tokens
                        span += doc[id_].text
                        if doc[id_].whitespace_ == " ":
                            span += " "
                        length -= doc[id_].__len__() + 1
                        id_ += 1
                    id_ -= 1
                    # token_end = copy.copy(id_)
                    end_id = copy.copy(doc[id_].idx + len(doc[id_]))
                    d_anno = {"start": start_id, "end": end_id,  # "token_start": token_start, "token_end": token_end,
                              "label": "INGR"}  # annotation["infon"]["#text"]
                    spans_d.append(d_anno)
                    span = span.rstrip()  # Remove possible trailing whitespace
                    assert span == doc.text[start_id:end_id]
                    meta_anno.append({"start": start_id, "end": end_id, "span": span, "anno": annotation["infon"]["#text"]})
                    try:
                        compare = annotation["text"]
                        if annotation["text"] == "confectioners ' sugar":
                            compare = "confectioners' sugar"
                        elif annotation["text"] == "confectioners ' coating":
                            compare = "confectioners' coating"
                        elif annotation["text"] == "confectioner 's sugar":
                            compare = "confectioner's sugar"
                        elif annotation["text"] == "mushroom 's liquid":
                            compare = "mushroom's liquid"
                        else:
                            compare = compare.replace("'", "")
                        if annotation["text"] != compare:
                            print("_ Fixed token mismatch! _")
                            print(annotation["text"])
                            print(span)
                            print()
                        assert span == compare
                        break
                    except AssertionError:
                        if i == 2:
                            print(span)
                            print(annotation["text"])
                            print(int(annotation["location"]["@offset"]))
                            print(int(annotation["location"]["@length"]))
                            print(recipe["id"])
                            print()

            d_entry["spans"] = spans_d
            meta = {"id": recipe["id"], "category": recipe["infon"][0]["#text"], "annos": meta_anno,
                    "sentences": sentences}
            d_entry["meta"] = meta
        except KeyError:
            # Entry without ingredients
            meta = {"id": recipe["id"], "category": recipe["infon"][0]["#text"], "sentences": sentences}
            d_entry["meta"] = meta

        d_out.append(d_entry)

    # Shuffle the records for representative annotation
    random.shuffle(d_out)
    with open("/data/Intermediate/Stable/Updated/Source/recipes_all_v1.json", "w") as fp:
        json.dump(d_out, fp, indent=4)

    # Split files by category
    datasets = defaultdict(list)
    for recipe in d_out:
        datasets[recipe["meta"]["category"]].append(recipe)

    file_ = "/home/chroner/PhD_remote/FoodBase/data/Intermediate/Stable/recipes_"
    for category, recipes in datasets.items():
        with open(file_ + category + "_v1.json", "w") as fp:
            json.dump(recipes, fp, indent=4)


if __name__ == '__main__':
    nlp.tokenizer = custom_tokenizer(nlp)
    print_for_Prodigy()
