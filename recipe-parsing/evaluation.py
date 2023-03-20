import copy
import sys
from pyvis.network import Network
from process import process
from os import listdir, makedirs
from os.path import isfile, join, isdir, normpath
import json
import re
from pprint import pprint
import spacy
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import rapidfuzz as fz
import static_variables
from collections import OrderedDict, defaultdict
import datetime
import itertools
# from Bio.Align import PairwiseAligner  # For aligning output with input using DP
import spacy_alignments
import logging
log = logging.getLogger(__name__)

# https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
# We approximate the SemEval Type NER Metric
scores_NER = {"Correct": defaultdict(int), "Hypothesised": defaultdict(int), "Missed": defaultdict(int), "Incorrect": defaultdict(int)}
scores_Membership = {"Correct": 0, "Hypothesised": 0, "Missed": 0} #  "Incorrect": 0
# Only for entities, if predicted another label then it will be as Hypothesised or Missed
scores_Primes = {"Correct": 0, "Hypothesised": 0, "Missed": 0} # "Incorrect_Percentage": 0
# The correct Entity is applied if a Couple is wrong but at least the Entity Type is correct (e.g., `Modifier`)
# Then the Correct_Percentage_Entity is the remaining percentage x defaultdict(int).5 if correct
# I can calculate a more strict version if I only consider Correct_Percentage_Couple == 1 (1defaultdict(int)0%)
scores_RE = {"Correct": defaultdict(int), "Hypothesised": defaultdict(int), "Missed": defaultdict(int)}  #  "Correct_Percentage_Entity": 0,
nlp = None

def get_recipes(annotator, meal_type, version="1"):
    recipes = dict()
    # Start with recipes
    # Prodigy Annotationsrecipe_vis
    with open("/home/chroner/PhD_remote/FoodBase/data/Processed/DB/Final/" + meal_type + "/" + annotator + "/prodigy_" +
              version + ".json", "r") \
            as fdrs:
        d_out = json.load(fdrs)
    # PSVs
    print("Total recipes")
    print(len(d_out))
    psvs = "/home/chroner/PhD_remote/FoodBase/data/Processed/PSV_2.0_" + annotator + "/"
    onlyfiles = [f for f in listdir(psvs) if isfile(join(psvs, f))]
    # omit = ["4recipe353", "4recipe641", "4recipe338", "4recipe618", "4recipe735", "4recipe611", "4recipe15", "4recipe382",
    #         "4recipe212", "4recipe406", "4recipe285", "4recipe247"]
    omit = []
    not_passed = []
    for recipe in d_out:
        log.debug(recipe["meta"]["id"])
        # Skip erroring recipes
        # if recipe["meta"]["id"] in omit:
        #     continue  # 4recipe512, 4recipe338, 4recipe530
        for file in onlyfiles:
            if file.split(".")[0] == recipe["meta"]["id"]:
                # if recipe["meta"]["id"] == "4recipe382":  # todo remove me
                try:
                    recipes[recipe["meta"]["id"]] = process(recipe, join(psvs, file))
                except:#  OSError:  # todo remove me
                    not_passed.append(recipe["meta"]["id"])
                break
        else:
            omit.append(recipe["meta"]["id"])
            log.debug("not found")
    log.debug("Not passed")
    pprint(not_passed)
    log.debug("Not Found CSV")
    pprint(omit)
    return recipes

def parser():
    pass

def get_values(tok, idx=0):
    try:
        if tok.count('|') == 4:
            tok = re.sub(r'[\[\]]', "", tok)
            return tok.split("|")[idx].strip()
    except AttributeError:
        return None
def assign_values(func):
    def match_function(tok):
        match func:
            case x if "token" in func.__name__:
                return func(get_values(tok, idx=0))
            case x if "prime" in func.__name__:
                return func(get_values(tok, idx=1))
            case x if "mem" in func.__name__:
                return func(get_values(tok, idx=2))
            case x if "ner" in func.__name__:
                return func(get_values(tok, idx=3))
            case x if "re" in func.__name__:
                return func(get_values(tok, idx=4))
    return match_function

def return_tok(tok):
    # process val
    if not tok:
        return "O"
    else:
        return tok

@assign_values
def get_token(tok):
    return return_tok(tok)

@assign_values
def get_mem(tok):
    return return_tok(tok)

@assign_values
def get_primes(tok):
    return return_tok(tok)

@assign_values
def get_ner(tok):
    return  return_tok(tok)
@assign_values
def get_re(tok):
    return return_tok(tok)

def recipe_vis(out, fn_ext, primes=True):

    def get_closest_cand(k, token):
        token = token.strip()
        nonlocal nodes
        for i in range(1, 1_000):
            # See Forwards
            try:
                if nodes[k + i]["title"].strip() == token.strip():
                    # try: # Get also the last token from span, for matching relations
                    #     if get_token(out[k + i]).strip() == get_token(out[k + i + 1]).strip():
                    #         continue
                    # except KeyError:
                    #     pass
                    return k + i
            except KeyError:
                pass
            # See Backwards
            try:
                if nodes[k - i]["title"].strip() == token.strip():
                    return k - i
            except KeyError:
                pass

    # Parse predicates
    nodes = dict()
    edges  = list()
    out = {int(k): v for k,v in out.items()}
    out = OrderedDict(sorted(out.items(), reverse=True))
    mem = None
    counter_id = 1_000
    additional_nodes = defaultdict(list)
    for k, tok in out.items():
        if tok == mem:
            continue
        # if get_ner(tok).lower() not in static_variables.code:
        #     continue
        # Maybe add the primes, instead for the Entities
        if get_ner(tok).lower().strip() not in static_variables.entities:
            nodes[k] = {"title": get_token(tok), "label": get_ner(tok)}
        else:
            # Alternative format
            for prime in get_primes(tok).split(','):
                if prime == "O":
                    break
                prime = prime.strip()
                counter_id += 1
                additional_nodes[k].append({"id": counter_id, "title": prime, "label": get_ner(tok)})
        mem = tok
    for k, tok in out.items():
        if tok == mem:
            continue
        ## Get Memberships
        for member in get_mem(tok).split(','):
            member = member.strip()
            # It's a Predicate with no upper belonging
            if get_ner(tok).strip().lower() in static_variables.code and get_mem(tok).strip() == get_token(tok).strip():
                break
            # Else add Membership
            else:
                if additional_nodes[k]:
                    for l in additional_nodes[k]:
                        relation = {"head": l["id"], "child": get_closest_cand(k, member), "label": "Member"}
                        edges.append(relation)
                else:
                    for member_ in member.split(","):
                        member_  = member_.strip()
                        relation = {"head": k, "child": get_closest_cand(k, member_), "label": "Member"}
                        edges.append(relation)
        # Add the rest of the relations
        for arc in get_re(tok).split(','):
            if arc == "O":
                break
            label = arc.split("=")[0].strip()
            child = arc.split("=")[1].strip()
            if label == "Dependency":
                relation = {"head": get_closest_cand(k, child), "child": k, "label": label}
                edges.append(relation)
            else:
                if  additional_nodes[k]:
                    for l in additional_nodes[k]:
                        relation = {"head": l["id"], "child": get_closest_cand(k, child), "label": label}
                        edges.append(relation)
                else:
                    relation = {"head": k, "child": get_closest_cand(k, child), "label": label}
                    edges.append(relation)
        # Avoid duplicates from span
        mem = tok

    # Populate the Graph
    g = Network('890px', '1900px', directed=True)
    node_ids, node_value, node_title, node_label, node_color, node_size, node_shape = list(), list(), list(), list(), list(), list(), list()
    for k, node in nodes.items():
        node_ids.append(k)  # Token end seems to be the identifier for the relations
        node_title.append(node["title"])
        label = node["label"]
        node_label.append(label)
        if label.lower() in static_variables.code:
            node_color.append("#E15233")
            node_size.append(20)
            node_shape.append("ellipse")
        elif label.lower() in static_variables.entities:
            if "ingr" in label.lower():
                node_color.append("#EFAE2C")
            else:
                node_color.append("#12A5E9")
            node_size.append(14)
            node_shape.append("dot")
        elif label.lower() in static_variables.states:
            if "ingr" in label.lower():
                node_color.append("#EFAE2C")
            else:
                node_color.append("#12A5E9")
            node_size.append(13)
            node_shape.append("triangle")
        elif label.lower() in static_variables.entities_aux:
            node_color.append("#3BE8AF")
            node_size.append(15)
            node_shape.append("box")
        else:
            node_color.append("#82A5B6")
            node_size.append(10)
            node_shape.append("hexagon")
    for l in additional_nodes.values():
        for node in l:
            node_ids.append(node["id"])  # Token end seems to be the identifier for the relations
            node_title.append(node["title"])
            label = node["label"]
            node_label.append(label)
            if label.lower() in static_variables.code:
                node_color.append("#E15233")
                node_size.append(20)
                node_shape.append("ellipse")
            elif label.lower() in static_variables.entities:
                if "ingr" in label.lower():
                    node_color.append("#EFAE2C")
                else:
                    node_color.append("#12A5E9")
                node_size.append(14)
                node_shape.append("dot")
            elif label.lower() in static_variables.states:
                if "ingr" in label.lower():
                    node_color.append("#EFAE2C")
                else:
                    node_color.append("#12A5E9")
                node_size.append(13)
                node_shape.append("triangle")
            elif label.lower() in static_variables.entities_aux:
                node_color.append("#3BE8AF")
                node_size.append(15)
                node_shape.append("box")
            else:
                node_color.append("#82A5B6")
                node_size.append(10)
                node_shape.append("hexagon")
    g.add_nodes(node_ids, title=node_title, label=node_label, value=node_size, color=node_color, shape=node_shape)
    for edge in edges:
        label = edge["label"]
        if label.lower() == "modifier":
            width = 2
            color = "#65C762"
        elif label.lower() == "member":
            width = 4
            color = "#62C7B5"
        elif label.lower() == "dependency":
            width = 5
            color = "#E38527"
        else:
            width = 1
            color = "#82A5B6"
        # if relation["head_span"]["label"] == "ACTION" and relation["child_span"]["label"] == "ACTION":
        #     g.add_edge(relation["child"], relation["head"], title=label, width=width, color=color)
        # else:
        try:
            g.add_edge(edge["head"], edge["child"], title=label, width=width, color=color)
        except AssertionError:
            continue
    try:
        g.show("temp/" + fn_ext + ".html")
        # g.save_graph("temp/" + fn_ext + ".html")
        # g.to_json
    except FileNotFoundError:
        pass

def get_score(gs, pr, **kwargs):
    global scores_NER
    # pattern = re.compile(r"[\[\]]")
    # pattern.split(gs)
    log.debug(gs["text_in"])
    tokens_gs = dict()
    for token_id in range(len(gs["tokens"])):
        try:
            tokens_gs[token_id] = gs["tokens_label"][token_id]
        except KeyError:
            try:
                tokens_gs[token_id] = gs["tokens_label"][str(token_id)]
            except KeyError:
                tokens_gs[token_id] = gs["tokens"][token_id]

    tokens_pr = dict()
    for token_id in range(len(pr["tokens"])):
        try:
            tokens_pr[token_id] = pr["tokens_label"][token_id]
        except KeyError:
            try:
                tokens_pr[token_id] = pr["tokens_label"][str(token_id)]
            except KeyError:
                tokens_pr[token_id] = pr["tokens"][token_id]
    # recipe_vis(gs["tokens_label"], fn_ext=kwargs["key"] + "_gt") # todo uncomment
    # recipe_vis(pr["tokens_label"], fn_ext=kwargs["key"] + "_pr")
    # NER
    def ner_score(gs_, pr_):
        global scores_NER
        match gs_, pr_:
            case x if gs_ == pr_ :
                scores_NER["Correct"][gs_] += 1
            case x if gs_ == "O" and pr_ != "O":
                scores_NER["Hypothesised"][pr_] += 1
            case x if gs_ != "O" and pr_ == "O":
                scores_NER["Missed"][gs_] += 1
            case _:
                scores_NER["Incorrect"][frozenset({gs_, pr_})] += 1
    # Membership
    def mem_score(gs_, pr_):
        global scores_Membership

        gs_s = {tok.strip() for tok in gs_.split(',')}
        pr_s = {tok.strip() for tok in pr_.split(',')}
        scores_Membership["Correct"] += len(gs_s.intersection(pr_s))  # Stands for TP, TN together
        scores_Membership["Missed"] += len(gs_s - pr_s)
        scores_Membership["Hypothesised"] += len(pr_s - gs_s)
        # scores_Membership["Incorrect"] += len(gs_s.symmetric_difference(pr_s))
    # Primes
    def primes_score(gs_, pr_):
        global scores_Primes
        # If the token is not an Entity skip
        if gs_.strip() == "" and pr_.strip() == "":
            return
        # Skip if it's not entity related
        if gs_ == pr_ == 'O':
            return
        gs_s = {tok.strip() for tok in gs_.split(',')}
        pr_s = {tok.strip() for tok in pr_.split(',')}
        scores_Primes["Correct"] += len(gs_s.intersection(pr_s))
        scores_Primes["Missed"] += len(gs_s - pr_s)
        scores_Primes["Hypothesised"] += len(pr_s - gs_s)
    # Relations

    def split_re(tok):
        """ In some cases there is a comma in the span, hence, it's not suggested to just split by it
        """
        l = tok.split(',')
        merge = list()
        for idx, re_ in enumerate(l[1:], start=1):
            if re_.strip()[0].islower():
                merge.append((idx-1, idx))
        for m in reversed(merge):
            l.append(l[m[0]] + l[m[1]])
        for m in merge:
            del l[m[1]]
            del l[m[0]]
        return {i.strip() for i in l}

    def re_score(gs_, pr_):
        global scores_RE
        # If the token has no Entities then skip
        if gs_.strip() == "" and pr_.strip() == "":
            return
        gs_s = split_re(gs_)
        pr_s = split_re(pr_)
        # Currently is strict, has to match the
        for re_ in gs_s.intersection(pr_s):
            scores_RE["Correct"][re_.split("=")[0].strip()] += 1
        for re_ in gs_s.intersection(gs_s - pr_s):
            scores_RE["Missed"][re_.split("=")[0].strip()] += 1
        for re_ in gs_s.intersection(pr_s - gs_s):
            scores_RE["Hypothesised"][re_.split("=")[0].strip()] += 1
        # scores_RE["Correct"] += len(gs_s.intersection(pr_s))
        # scores_RE["Missed"] += len(gs_s - pr_s)
        # scores_RE["Hypothesised"] += len(pr_s - gs_s)

    for (k_gs, tok_gs), (k_pr, tok_pr) in zip(tokens_gs.items(), tokens_pr.items()):
        # Parsing Error
        if k_gs != k_pr:
            raise IndexError
        # If both are 'O' skip
        if isinstance(tok_gs, dict) and isinstance(tok_pr, dict):
            continue
        ner_score(get_ner(tok_gs), get_ner(tok_pr))
        mem_score(get_mem(tok_gs), get_mem(tok_pr))
        primes_score(get_primes(tok_gs), get_primes(tok_pr))
        if get_ner(tok_gs).lower() in static_variables.entities and get_primes(tok_gs) != get_primes(tok_pr):
            log.debug((tok_gs, tok_pr))
            print()
        re_score(get_re(tok_gs), get_re(tok_pr))
    # pprint(scores_Primes)
    # pprint(scores_Membership)
    # pprint(scores_NER)
    # pprint(scores_RE)

def fname(fp, dataset_name, rnd_num="none", fold=""):
    td = datetime.date.today()
    return join(fp, dataset_name + "_" + str(td.year) + "_" + str(td.month) + "_" + str(td.day) + "_" +
                str(rnd_num) + "_f" + str(fold) + ".csv")

def generate_cross_val_predict_files(d_out, fp, rnd_num=None):
    X = [i["text_in"] for i in d_out.values()]
    Y = [i["text_out"] for i in d_out.values()]
    k = list(d_out.keys())
    kf = KFold(n_splits=5, random_state=rnd_num, shuffle=True)
    for i, (train, test) in enumerate(kf.split(X, y=Y)):
        val, test = train_test_split(test, test_size=0.66, random_state=rnd_num)
        X_train = [X[idx] for idx in train]
        k_train = [k[idx] for idx in train]
        Y_train = [Y[idx] for idx in train]
        X_val = [X[idx] for idx in val]
        k_val = [k[idx] for idx in val]
        Y_val = [Y[idx] for idx in val]
        X_test = [X[idx] for idx in test]
        k_test = [k[idx] for idx in test]
        Y_test = [Y[idx] for idx in test]

        pd.DataFrame({"id":k_train, "recipe": X_train, "out": Y_train}).to_csv(fname(fp, "train", rnd_num, i), index=False)
        pd.DataFrame({"id":k_val,"recipe": X_val, "out": Y_val}).to_csv(fname(fp, "validation", rnd_num, i), index=False)
        pd.DataFrame({"id":k_test,"recipe": X_test, "out": Y_test}).to_csv(fname(fp, "test", rnd_num, i), index=False)

def generate_predict_files(d_out, rnd_num, fp=None):
    X = [i["text_in"] for i in d_out.values()]
    Y = [i["text_out"] for i in d_out.values()]
    # 0.8 , 0.1 , 0.1
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.15, random_state=rnd_num)
    K_train, K_test = train_test_split(list(d_out.keys()), test_size=0.15, random_state=rnd_num)
    X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.40, random_state=rnd_num)
    K_test, K_val = train_test_split(K_test, test_size=0.40, random_state=rnd_num)

    # 0.7 0.1 0.2
    # Assertion Check
    for k, txt in zip(K_test, X_test):
        assert d_out[k]["text_in"] == txt
    # breakpoint()
    if fp:
        # todo Print meta files with the keys
        pd.DataFrame({"recipe": X_train, "out": Y_train}).to_csv(fname(fp, "train"), index=False)
        pd.DataFrame({"recipe": X_val, "out": Y_val}).to_csv(fname(fp, "validation"), index=False)
        pd.DataFrame({"recipe": X_test, "out": Y_test}).to_csv(fname(fp, "test"), index=False)

    # breakpoint()
    return K_train, K_val, K_test


def alignment(pred, gt_d):
    # r'(?<=\[)[^\s]+'
    # '(? <= \[). *?(?=\])'
    # '(? <= \[). *?(?=\])'
    tokens_label_raw = re.findall('\[.*?\]', pred)
    repl = "placeholder_x70x6c"
    temp = re.sub(r'\[.*?\]', repl, pred, 0, re.IGNORECASE | re.MULTILINE)
    # tokenise from spaCy
    nlp.tokenizer(temp)

    # Retrieve original text input
    tokens_label_raw_in = [tok.split('[')[1].split('|')[0].strip() for tok in tokens_label_raw]

    text_in_pred = ""
    par_temp = copy.copy(temp)
    for tok in tokens_label_raw_in:
        idx = par_temp.index(repl)
        par_temp = par_temp.replace(repl, tok, 1)
        text_in_pred += par_temp[:idx + len(tok)]
        par_temp = par_temp[idx + len(tok):]
    text_in_pred += par_temp
    text_in_pred = text_in_pred.rstrip()

    # Mapping dictionary between text_in_pred and tokens_label_raw_in
    # First get the tokenised labelled output
    mapping_token_labels_raw_in = dict()
    tokenised_labels_raw_in = list()
    counter = 0
    for idx, tok in enumerate(tokens_label_raw_in):
        for tok_ in nlp(tok):
            mapping_token_labels_raw_in[counter] = idx
            tokenised_labels_raw_in.append(tok_.text)
            counter += 1

    pred_mapping = dict()
    counter = 0
    for idx, tok in enumerate(nlp(text_in_pred)): # Using same tokeniser
        try:
            if tok.text == tokenised_labels_raw_in[counter]:
                pred_mapping[idx] = counter
                counter +=1
        except IndexError: # Reached end of detected labelled tokens
            break

    # Align
    seq_gt = [tok["text"] for tok in gt_d["tokens"]]
    seq_pred = [tok.text for tok in nlp(text_in_pred)]

    # Align the found items only
    mapping_gt2pred = {}
    a2b, b2a = spacy_alignments.get_alignments(seq_gt, seq_pred)
    for i in range(len(seq_gt)):
        # print(seq_gt[i])
        for j in a2b[i]:
            # print("    ", seq_pred[j])
            mapping_gt2pred[i] = j

    tokens_label_pred = {}
    for idx in range(len(seq_gt)):
        try:
            tokens_label_pred[idx] = tokens_label_raw[mapping_token_labels_raw_in[pred_mapping[mapping_gt2pred[idx]]]]
        except KeyError:
            continue

    out_dict = {"text_in": text_in_pred, "text_out": pred, "tokens": gt_d["tokens"],
                "tokens_label": tokens_label_pred}
    return out_dict

def cross_val_scoring(gs_d, in_, out_):
    global nlp
    nlp = spacy.blank("en")
    # Get the Gold Standard
    with open(gs_d, "r") as fd:
        gs_d = json.load(fd)
    # Get predictions
    predictions = list()
    with open(out_, 'r') as f:
        for line in f:
            predictions.append(line)
    # Get the keys
    K_test = pd.read_csv(in_, header=0, index_col=False)["id"]

    pred_d = {}
    # Do the mapping and create dict
    for pred, k in zip(predictions, K_test):
        # if k == "1recipe1167":  # todo remove me
        pred_d[k] = alignment(pred, gs_d[k])

    # Get Ground Truth dict
    for k in K_test:
        if k not in gs_d:
            del gs_d[k]

    return fill_scores(gs_d, pred_d)

def predict_files_score(d_out, fp, rnd_num):
    global nlp
    nlp = spacy.blank("en")
    with open(d_out, "r") as fd:
        d_out = json.load(fd)
    predictions = list()
    with open(fp, 'r') as f:
        for line in f:
            predictions.append(line)
    # todo get keys from metadata
    # Get tokens for mapping, deprecated
    # _, _, K_test = generate_predict_files(d_out, rnd_num=rnd_num)
    # Get the keys by similarity parsing
    K_test = list()
    for idx, pred in enumerate(predictions):
        highest_score = 0
        k_highest = None
        for k, rec in d_out.items():
            score = fz.fuzz.ratio(rec["text_out"], pred)
            if score > highest_score:
                highest_score = score
                k_highest = k
        # Sanity check
        # print(idx)
        # print(pred)
        # print(d_out[k_highest]["text_out"])
        # print("\n\n--")
        K_test.append(k_highest)
    # remove spurious entries, manually
    remove_idx = [2, 5, 6, 8]
    for idx in sorted(remove_idx, reverse=True):
        del K_test[idx]
        del predictions[idx]

    pred_d = {}
    # Do the mapping and create dict
    for pred, k in zip(predictions, K_test):
        pred_d[k] = alignment(pred, d_out[k])

    # Get Ground Truth dict
    for k in K_test:
        if k not in d_out:
            del d_out[k]

    fill_scores(d_out, pred_d)

def fill_scores(Gold_standard, Prediction):
    for k, v in Gold_standard.items():
        try:
            get_score(v, Prediction[k], key=k)
        except KeyError:
            continue
    results = {}
    # Primes
    primes = {}
    precision = scores_Primes["Correct"] / (scores_Primes["Correct"] + scores_Primes["Hypothesised"])
    recall = scores_Primes["Correct"] / (scores_Primes["Correct"] + scores_Primes["Missed"])
    f_score =  (2 * precision * recall) / (precision + recall)
    print("Primes F-Score: ", round(f_score, 3), end='\t')
    primes["fscore"] = f_score
    print("Precision: ", round(precision, 3), end='\t')
    primes["precision"] = precision
    print("Recall: ", round(recall, 3))
    primes["recall"] = recall
    results['primes'] = primes


    # Membership
    memberships = {}
    precision = scores_Membership["Correct"] / (scores_Membership["Correct"] + scores_Membership["Hypothesised"])
    recall = scores_Membership["Correct"] / (scores_Membership["Correct"] + scores_Membership["Missed"])
    f_score =  (2 * precision * recall) / (precision + recall)
    print("Membership F-Score: ", round(f_score, 3), end='\t')
    memberships["fscore"] = f_score
    print("Precision: ", round(precision, 3), end='\t')
    memberships["precision"] = precision
    print("Recall: ", round(recall, 3))
    memberships["recall"] = recall
    results['memberships'] = memberships

    # NER total. I think is micro
    ners = {}
    precision = sum(scores_NER["Correct"].values())  / (sum(scores_NER["Correct"].values()) +
                                                        sum(scores_NER["Incorrect"].values())  +
                                                        sum(scores_NER["Hypothesised"].values()) )
    recall = sum(scores_NER["Correct"].values())  / (sum(scores_NER["Correct"].values())  +
                                                     sum(scores_NER["Incorrect"].values())  + sum(scores_NER["Missed"].values()))
    f_score =  (2 * precision * recall) / (precision + recall)
    print("NER Micro F-Score: ", round(f_score, 3), end='\t')
    ners["fscore"] = f_score
    print("Precision: ", round(precision, 3), end='\t')
    ners["precision"] = precision
    print("Recall: ", round(recall, 3))
    ners["recall"] = recall
    results['ners'] = ners

    # NER for each Class/Tag
    for k in scores_NER["Correct"].keys():
        precision = scores_NER["Correct"][k] / (scores_NER["Correct"][k] +
                                                           sum([scores_NER["Incorrect"][k_] for k_ in scores_NER["Incorrect"].keys() if k in k_]) +
                                                           scores_NER["Hypothesised"][k])
        recall = scores_NER["Correct"][k] / (scores_NER["Correct"][k] +
                                                           sum([scores_NER["Incorrect"][k_] for k_ in scores_NER["Incorrect"].keys() if k in k_]) +
                                                           scores_NER["Missed"][k])
        f_score = (2 * precision * recall) / (precision + recall)
        print("\t" + k.ljust(10) + " NER F-Score: ", round(f_score, 3), end='\t')
        print("Precision: ", round(precision, 3), end='\t')
        print("Recall: ", round(recall, 3), end='\t')
        print("Support: ", scores_NER["Correct"][k] +
              sum([scores_NER["Incorrect"][k_] for k_ in scores_NER["Incorrect"].keys() if k in k_]) + scores_NER["Missed"][k])

    # RE total
    res = {}
    del scores_RE["Correct"]['O']  # I may want to remove this
    precision = sum(scores_RE["Correct"].values()) / (sum(scores_RE["Correct"].values()) + sum(scores_RE["Hypothesised"].values()))
    recall = sum(scores_RE["Correct"].values()) / (sum(scores_RE["Correct"].values()) + sum(scores_RE["Missed"].values()))
    f_score =  (2 * precision * recall) / (precision + recall)
    print("RE Micro F-Score: ", round(f_score, 3), end='\t')
    res["fscore"] = f_score
    print("Precision: ", round(precision, 3), end='\t')
    res["precision"] = precision
    print("Recall: ", round(recall, 3))
    res["recall"] = recall
    results['res'] = res
    # RE for each Class/Tag
    for k in scores_RE["Correct"].keys():
        precision = scores_RE["Correct"][k] / (scores_RE["Correct"][k] + scores_RE["Hypothesised"][k])
        recall = scores_RE["Correct"][k] / (scores_RE["Correct"][k] + scores_RE["Missed"][k])
        f_score = (2 * precision * recall) / (precision + recall)
        print("\t" + k.ljust(10)  + " RE F-Score: ", round(f_score, 3), end='\t')
        print("Precision: ", round(precision, 3), end='\t')
        print("Recall: ", round(recall, 3), end='\t')
        print("Support: ", scores_RE["Correct"][k] + scores_RE["Missed"][k])

    return results
    # with open(fname("train"), 'a') as ff:
    #     ff.write("in, out \n")
    #     for in_, out in zip(X_train, Y_train):
    #         ff.write('"' + in_.replace('"', "'") + '", "' +  out.replace('"', "'") + '"' + '\n')
    # with open(fname("validation"), 'a') as ff:
    #     ff.write("in, out \n")
    #     for in_, out in zip(X_val, Y_val):
    #         ff.write('"' + in_.replace('"', "'") + '", "' +  out.replace('"', "'") + '"' + '\n')
    # with open(fname("test"), 'a') as ff:
    #     ff.write("in, out \n")
    #     for in_, out in zip(X_test, Y_test):
    #         ff.write('"' + in_.replace('"', "'") + '", "' +  out.replace('"', "'") + '"' + '\n')

def get_prediction():
    pass

if __name__ == '__main__':
    # Read from Mem

    # with open("/home/chroner/PhD_remote/FoodBase/recipe_wf/temp/Prediction.json", "r") as fd:
    #     Prediction = json.load(fd)

    meal_type = "Dinner"
    version = "4"
    # Get two dictionaries and measure the common recipes
    # Gold_standard = get_recipes("Phil", meal_type, version=version)
    # meal_type = "Lunch"
    # version = "3"
    # Gold_standard_lunch = get_recipes("Phil", meal_type, version=version)
    # # Join
    # Gold_standard = Gold_standard | Gold_standard_lunch
    # # Measure max length
    # lens_out = list()
    # lens_in = list()
    # nlp = spacy.load("en_core_web_sm")
    # for v in Gold_standard.values():
    #     lens_out.append(len(nlp(v["text_out"])))
    #     lens_in.append(len(nlp(v["text_in"])))
    # print("Max Lengths: ")
    # print(max(lens_in))
    # print(max(lens_out))
    # # Save to file
    # print("Gold Standard Length")
    # print(len(Gold_standard))
    #
    # with open("/home/chroner/PhD_remote/FoodBase/recipe_wf/temp/phil_gs_4_3.json", "w") as fd:
    #     json.dump(Gold_standard, fd, indent=4)


    with open("/home/chroner/PhD_remote/FoodBase/recipe_wf/temp/phil_gs_4_3.json", "r") as fd:
        gt = json.load(fd)
    # meal_type = "Dinner"
    # version = "1"
    # # Get two dictionaries and measure the common recipes
    # # haifa = get_recipes("Haifa", meal_type, version=version)
    with open("/home/chroner/PhD_remote/FoodBase/recipe_wf/temp/haifa_1.json", "r") as fd:
        # json.dump(haifa, fd, indent=4)
        haifa = json.load(fd)
    # # sys.exit(0)
    # print("len(haifa)")
    # print(len(haifa))
    fill_scores(gt, haifa)
    #
    # meal_type = "Dinner"
    # version = "1"
    # # Get two dictionaries and measure the common recipes
    # mau = get_recipes("Mau", meal_type, version=version)
    # print("len(mau)")
    # print(len(mau))
    # fill_scores(gt, mau)



    # fp = "/home/chroner/PhD_remote/transformers/examples/pytorch/summarization/data/cross_val"
    # generate_cross_val_predict_files(Gold_standard, fp, rnd_num=3245)
    sys.exit(0)
    ## Cross Validation Results on Pegasus
    results = list()
    for i in [0, 1, 2, 3, 4]:
        gs_d = "/home/chroner/PhD_remote/FoodBase/recipe_wf/temp/phil_gs_4_3.json"
        in_ = "/home/chroner/PhD_remote/ricoreca-repro-remote-2/data/test_2023_2_22_3245_f" + str(i) + ".csv"
        out_ = "/home/chroner/PhD_remote/FoodBase/data/Predictions/CrossVal/predict/fold" + str(i) + "_pegasus/generated_predictions.txt"
        # out_ = "/home/chroner/PhD_remote/FoodBase/data/Predictions/CrossVal/predict/alternate_fold0_pegasus/generated_predictions.txt"
        print("__ Fold: " + str(i) + " results__")
        results.append(cross_val_scoring(gs_d, in_, out_))

    bibi = defaultdict(list)
    for i in results:
        for k in i.keys():
            for k_, v_ in i[k].items():
                bibi[k,k_].append(v_)

    print("__PEGASUS Averaged results__")
    for k_, v_ in bibi.items():
        print(k_[0], " - ", k_[1], round(sum(v_) / 5, 3))
    sys.exit(0)
    ## Cross Validation Results on Pegasus
    results = list()
    for i in [0, 1, 2, 3, 4]:
        gs_d = "/home/chroner/PhD_remote/FoodBase/recipe_wf/temp/phil_gs_4_3.json"
        in_ = "/home/chroner/PhD_remote/ricoreca-repro-remote-2/data/test_2023_2_22_3245_f" + str(i) + ".csv"
        out_ = "/home/chroner/PhD_remote/FoodBase/data/Predictions/CrossVal/predict/fold" + str(i) + "_t5/generated_predictions.txt"
        print("__ Fold: " + str(i)+ " results__")
        results.append(cross_val_scoring(gs_d, in_, out_))

    bibi = defaultdict(list)
    for i in results:
        for k in i.keys():
            for k_, v_ in i[k].items():
                bibi[k,k_].append(v_)
    print("__T5 Averaged results__")
    for k_, v_ in bibi.items():
        print(k_[0], " - ", k_[1] , round(sum(v_) / 5, 3))


    # generate_predict_files(Gold_standard, rnd_num=5333, fp=fp)  # fp=fp
    ## Send the predictions to the cloud for training and inference
    ## Read the Predictions and evaluate accuracy
    # gs_d = "/home/chroner/PhD_remote/FoodBase/recipe_wf/temp/phil_gs_4_3.json"
    # pred_fp = "/home/chroner/PhD_remote/FoodBase/data/Predictions/generated_predictions_2.txt"
    # predict_files_score(gs_d, fp=pred_fp, rnd_num=5333)

    # # # Compare NER between Phil & Mau
    # version = "1"
    # Prediction = get_recipes("Haifa", meal_type, version=version)

    # fill_scores(Gold_standard, Prediction)
    # # breakpoint()
    # for k, v in Gold_standard.items():
    #     try:
    #         get_score(v, Prediction[k])
    #     except KeyError:
    #         continue
    # # NER
    # precision = scores_NER["Correct"] / (scores_NER["Correct"] +  scores_NER["Incorrect"] + scores_NER["Hypothesised"])
    # recall = scores_NER["Correct"] / (scores_NER["Correct"] +  scores_NER["Incorrect"] + scores_NER["Missed"])
    # f_score =  (2 * precision * recall) / (precision + recall)
    # print(f_score)



