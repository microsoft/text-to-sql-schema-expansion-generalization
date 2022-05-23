from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, TokenIndexer, Field, Instance
from allennlp.data.fields import TextField, ListField, IndexField, MetadataField
from allennlp.data.fields import (
    TextField,
    ListField,
    IndexField,
    MetadataField,
    ArrayField,
)

import anytree
from anytree.search import *
from collections import defaultdict
from overrides import overrides
from time import time
from typing import Dict
from smbop.utils import moz_sql_parser as msp

import smbop.utils.node_util as node_util
import smbop.utils.hashing as hashing
import smbop.utils.ra_preproc as ra_preproc
from anytree import Node, LevelOrderGroupIter
import dill
import itertools
from collections import defaultdict, OrderedDict
import json
import logging
import numpy as np
import os
from smbop.utils.replacer import Replacer
import time
from smbop.dataset_readers.enc_preproc import *
import smbop.dataset_readers.disamb_sql as disamb_sql
from smbop.utils.cache import TensorCache

logger = logging.getLogger(__name__)
all_list = list()

@DatasetReader.register("smbop")
class SmbopSpiderDatasetReader(DatasetReader):
    def __init__(
        self,
        lazy: bool = True,
        question_token_indexers: Dict[str, TokenIndexer] = None,
        keep_if_unparsable: bool = True,
        tables_file: str = None,
        dataset_path: str = "dataset/database",
        cache_directory: str = "cache/train",
        include_table_name_in_column=True,
        fix_issue_16_primary_keys=False,
        qq_max_dist=2,
        cc_max_dist=2,
        tt_max_dist=2,
        max_instances=10000000,
        decoder_timesteps=9,
        limit_instances=-1,
        value_pred=True,
        use_longdb=True,
    ):
        super().__init__(
            # lazy=lazy,
            # cache_directory=cache_directory,
            # max_instances=max_instances,
            #  manual_distributed_sharding=True,
            # manual_multi_process_sharding=True,
        )
        self.cache_directory = cache_directory
        self.cache = TensorCache(cache_directory)
        self.value_pred = value_pred
        self._decoder_timesteps = decoder_timesteps
        self._max_instances = max_instances
        self.limit_instances = limit_instances
        self.load_less = limit_instances!=-1

        self._utterance_token_indexers = question_token_indexers

        self._tokenizer = self._utterance_token_indexers["tokens"]._allennlp_tokenizer
        self.cls_token = self._tokenizer.tokenize("a")[0]
        self.eos_token = self._tokenizer.tokenize("a")[-1]
        self._keep_if_unparsable = keep_if_unparsable

        self._tables_file = tables_file
        self._dataset_path = dataset_path

        # ratsql
        self.enc_preproc = EncPreproc(
            tables_file,
            dataset_path,
            include_table_name_in_column,
            fix_issue_16_primary_keys,
            qq_max_dist,
            cc_max_dist,
            tt_max_dist,
            use_longdb,
        )
        self._create_action_dicts()
        self.replacer = Replacer(tables_file)

    def _create_action_dicts(self):
        unary_ops = [
            "keep",
            "min",
            "count",
            "max",
            "avg",
            "sum",
            "Subquery",
            "distinct",
            "literal",
            "abs",
            "length",
            "SGLProject",
            "julianday",
        ]

        binary_ops = [
            "eq",
            "like",
            "nlike",
            "add",
            "sub",
            "mul",
            "div",
            "nin",
            "lte",
            "lt",
            "neq",
            "in",
            "gte",
            "gt",
            "And",
            "Or",
            "except",
            "union",
            "intersect",
            "Product",
            "Val_list",
            "Orderby_desc",
            "Orderby_asc",
            "Project",
            "Selection",
            "Limit",
            "Groupby",
            "Is",
            "Not",     
        ]
        self.binary_op_count = len(binary_ops)
        self.unary_op_count = len(unary_ops)
        self._op_names = [
            k for k in itertools.chain(binary_ops, unary_ops, ["nan", "Table", "Value"])
        ]
        self._type_dict = OrderedDict({k: i for i, k in enumerate(self._op_names)})
        self.keep_id = self._type_dict["keep"]
        self._ACTIONS = {k: 1 for k in unary_ops}
        self._ACTIONS.update({k: 2 for k in binary_ops})
        self._ACTIONS = OrderedDict(self._ACTIONS)
        self.hasher = hashing.Hasher("cpu")

    def _init_fields(self, tree_obj):
        tree_obj = node_util.add_max_depth_att(tree_obj)
        tree_obj = node_util.tree2maxdepth(tree_obj)
        tree_obj = self.hasher.add_hash_att(tree_obj, self._type_dict)
        hash_gold_tree = tree_obj.hash
        hash_gold_levelorder = []
        for tree_list in LevelOrderGroupIter(tree_obj):
            hash_gold_levelorder.append([tree.hash for tree in tree_list])

        pad_el = hash_gold_levelorder[0]
        for i in range(self._decoder_timesteps - len(hash_gold_levelorder) + 2):
            hash_gold_levelorder.insert(0, pad_el)
        hash_gold_levelorder = hash_gold_levelorder[::-1]
        max_size = max(len(level) for level in hash_gold_levelorder)
        for level in hash_gold_levelorder:
            level.extend([-1] * (max_size - len(level)))
        hash_gold_levelorder = np.array(hash_gold_levelorder)
        return (
            hash_gold_levelorder,
            hash_gold_tree,
        )

    def process_instance(self, instance: Instance, index: int):
        return instance

    @overrides
    def _read(self, file_path: str):
        if file_path.endswith(".json"):
            yield from self._read_examples_file(file_path)
        else:
            raise ConfigurationError(f"Don't know how to read filetype of {file_path}")

    def _read_examples_file(self, file_path: str):
        # cache_dir = os.path.join("cache", file_path.split("/")[-1])

        cnt = 0
        cache_buffer = []
        cont_flag = True
        sent_set = set()
        for total_cnt,ins in self.cache:
            if cnt >= self._max_instances:
                break
            if ins is not None:
                yield ins
                cnt += 1
            sent_set.add(total_cnt)
            if self.load_less and len(sent_set) > self.limit_instances:
                cont_flag = False
                break

        if cont_flag:
            with open(file_path, "r") as data_file:
                json_obj = json.load(data_file)
                
                for total_cnt, ex in enumerate(json_obj):
                    if cnt >= self._max_instances:
                        break
                    if len(cache_buffer) > 50:
                        self.cache.write(cache_buffer)
                        cache_buffer = []
                    if total_cnt in sent_set:
                        continue
                    else: 
                        #print(ex)
                        #input()
                        ins = self.create_instance(ex)
                        cache_buffer.append([total_cnt, ins])
                    if ins is not None:
                        yield ins
                        cnt +=1
                visited = list()
                for ele in all_list:
                    if ele not in visited:
                        #print(ele)
                        visited.append(ele)
            self.cache.write(cache_buffer)


    def process_instance(self, instance: Instance, index: int):
        return instance

    def create_instance(self,ex):
        sql = None
        sql_with_values = None
        if "query_toks" in ex:
            try:
                ex = disamb_sql.fix_number_value(ex)
                sql = ' '.join(ex["query_toks_no_value"])
                sql_with_values = ex["query"]
                #sql = disamb_sql.disambiguate_items(
                #    ex["db_id"],
                #    ex["query_toks_no_value"],
                #    self._tables_file,
                #    allow_aliases=False,
                #)
                #print(sql)
                #print(sql_with_values)
                #sql_with_values = disamb_sql.sanitize(ex["query"])
                #print(ex['query'])
                #print(sql_with_values)
                #input()
            except Exception as e:
                # there are two examples in the train set that are wrongly formatted, skip them
                print(f"error with {ex['query']}")
                return None

        ins = self.text_to_instance(
            utterance=ex["question"],
            db_id=ex["db_id"],
            sql=sql,
            sql_with_values=sql_with_values,
            raw_col=ex['raw_columns']
        )
        return ins


    def text_to_instance(
        self, utterance: str, db_id: str, sql=None, sql_with_values=None, raw_col=None):
        fields: Dict[str, Field] = {
            "db_id": MetadataField(db_id),
        }
        #print(utterance)

        tokenized_utterance = self._tokenizer.tokenize(utterance)
        #(tokenized_utterance)
        has_gold = sql is not None
        #print(tokenized_utterance)
        #print(sql_with_values)
        RULES_values = """[["Or", ["neq", "neq"]], ["Orderby_desc", ["max", "Groupby"]], ["Or", ["eq", "lt"]], ["And", ["eq", "neq"]], ["And", ["lt", "neq"]], ["Selection", ["like", "Table"]], ["And", ["gte", "lte"]], ["And", ["eq", "And"]], ["Val_list", ["sum", "Value"]], ["Project", ["min", "Table"]], ["Or", ["lt", "gt"]], ["Selection", ["gte", "Table"]], ["Selection", ["lt", "Product"]], ["And", ["gte", "gte"]], ["lte", ["Value", "literal"]], ["Project", ["distinct", "Table"]], ["Subquery", ["intersect"]], ["And", ["And", "And"]], ["count", ["Value"]], ["Orderby_desc", ["length", "Project"]],["Orderby_desc", ["Value", "Project"]], ["And", ["eq", "neq"]], ["Or", ["like", "like"]], ["Limit", ["Value", "Orderby_asc"]], ["gt", ["Value", "Subquery"]], ["Val_list", ["max", "max"]], ["Or", ["eq", "gt"]], ["Val_list", ["min", "min"]], ["Val_list", ["Val_list", "Value"]], ["sum", ["Value"]], ["Selection", ["eq", "Product"]], ["Project", ["Value", "Selection"]], ["Project", ["sum", "Selection"]], ["Val_list", ["count", "Value"]], ["neq", ["Value", "literal"]], 
        ["Orderby_asc", ["avg", "Groupby"]], ["Val_list", ["min", "Value"]], ["min", ["Value"]], ["Or", ["gt", "lt"]], ["eq", ["Value", "Subquery"]], ["lt", ["Value", "Subquery"]], ["Val_list", ["count", "max"]], ["Selection", ["And", "Product"]], ["gte", ["avg", "Value"]], ["Val_list", ["Val_list", "count"]], ["Project", ["Val_list", "Selection"]], ["lte", ["Value", "Value"]], ["Val_list", ["sum", "min"]], ["Or", ["gt", "gt"]], ["Val_list", ["max", "min"]], ["gt", ["count", "Value"]], ["Product", ["Table", "Table"]], ["neq", ["Value", "Value"]], ["And", ["lt", "eq"]], ["And", ["eq", "nin"]], ["Orderby_asc", ["Val_list", "Project"]], ["Groupby", ["Val_list", "Project"]], ["Val_list", ["Val_list", "min"]], ["gte", ["Value", "literal"]], ["gt", ["avg", "Value"]], ["eq", ["count", "Value"]], ["Project", ["avg", "Table"]], ["lt", ["count", "Value"]], ["Orderby_desc", ["avg", "Groupby"]], ["Val_list", ["count", "sum"]], ["And", ["eq", "eq"]], ["lt", ["min", "Value"]], ["Selection", ["Or", "Product"]], ["And", ["gt", "in"]], ["Or", ["gt", "eq"]], ["Val_list", ["sum", "avg"]], ["lt", ["avg", "Value"]], 
        ["Project", ["max", "Selection"]], ["Val_list", ["sum", "sum"]], ["And", ["And", "lt"]], ["Limit", ["Value", "Orderby_desc"]], ["Selection", ["eq", "Table"]], ["gt", ["max", "Value"]], ["Orderby_asc", ["Value", "Groupby"]], ["Project", ["max", "Table"]], ["And", ["eq", "gt"]], ["literal", ["Value"]], ["Val_list", ["avg", "Value"]], ["gt", ["Value", "literal"]], ["gte", ["Value", "Value"]], ["Selection", ["lte", "Table"]], ["Selection", ["And", "Table"]], ["Project", ["count", "Selection"]], ["Val_list", ["Val_list", "sum"]], ["And", ["gte", "gt"]], ["And", ["gt", "lt"]], ["And", ["in", "in"]], ["Val_list", ["Value", "max"]], ["in", ["Value", "Subquery"]], ["lte", ["sum", "Value"]], ["Selection", ["neq", "Table"]], ["lt", ["Value", "literal"]], ["And", ["And", "lte"]], ["Val_list", ["avg", "count"]], ["Project", ["avg", "Selection"]], ["Val_list", ["Value", "count"]], ["Val_list", ["max", "Value"]], ["union", ["Subquery", "Subquery"]], ["Selection", ["gt", "Table"]], 
        ["Val_list", ["sum", "max"]], ["except", ["Subquery", "Subquery"]], ["Subquery", ["Project"]], ["And", ["neq", "eq"]], ["And", ["gt", "gt"]], ["Project", ["count", "Table"]], ["Val_list", ["Value", "avg"]], ["gt", ["Value", "Value"]], ["And", ["eq", "Or"]], ["Project", ["Value", "Table"]], ["like", ["Value", "literal"]], ["Orderby_desc", ["Value", "Groupby"]], ["And", ["gt", "lte"]], ["Val_list", ["distinct", "Value"]], ["Val_list", ["Value", "sum"]], ["Selection", ["lt", "Table"]], ["And", ["eq", "lt"]], ["And", ["gt", "gte"]], ["Orderby_asc", ["Value", "Project"]], ["Val_list", ["avg", "min"]], ["eq", ["Value", "Value"]], ["And", ["And", "Or"]], ["Val_list", ["avg", "max"]], ["Subquery", ["union"]], ["Orderby_asc", ["count", "Groupby"]], ["lt", ["Value", "Value"]], ["Subquery", ["Groupby"]], ["Project", ["Val_list", "Product"]], 
        ["Val_list", ["min", "max"]], ["Selection", ["eq", "Table"]], ["Selection", ["in", "Table"]], ["And", ["like", "neq"]], ["And", ["lt", "lte"]],["And", ["gte", "eq"]], ["count", ["distinct"]], ["Project", ["distinct", "Selection"]], ["lte", ["Value", "Subquery"]], ["Subquery", ["Limit"]], ["Or", ["gte", "gte"]], ["Val_list", ["Value", "Value"]], ["Orderby_asc", ["sum", "Groupby"]], ["And", ["eq", "lte"]], ["max", ["Value"]], ["Selection", ["nlike", "Table"]], ["Or", ["eq", "eq"]], ["gte", ["sum", "Value"]], ["And", ["eq", "gte"]], ["Product", ["Product", "Table"]], ["Val_list", ["min", "avg"]], ["eq", ["Value", "literal"]], ["nlike", ["Value", "literal"]], ["Selection", ["nin", "Table"]], ["Val_list", ["count", "count"]], ["neq", ["Value", "Subquery"]], 
        ["Val_list", ["avg", "avg"]], ["Project", ["add", "Selection"]], ["add", ["Project", "Project"]], ["eq", ["Value", "Value"]], ["sub", ["Value", "Value"]], ["add", ["count", "count"]], ["add", ["Value", "Value"]], ["sub", ["Project", "Project"]], ["gt", ["Project", "Value"]], ["gt", ["Project", "Project"]], ["gt", ["avg", "Subquery"]], ["Project", ["count", "Selection"]], ["eq", ["Value", "add"]], ["Project", ["sub", "Value"]], ["add", ["Project", "Value"]], ["sub", ["Project", "Value"]], ["Project", ["Value", "Selection"]],["Val_list", ["avg", "sum"]], ["And", ["And", "gte"]], ["And", ["eq", "like"]], ["Orderby_desc", ["count", "Groupby"]], ["distinct", ["Value"]], ["gte", ["count", "Value"]], ["lte", ["count", "Value"]], ["And", ["And", "neq"]], ["And", ["And", "like"]], ["And", ["And", "eq"]], ["Val_list", ["Val_list", "max"]], ["gt", ["sum", "Value"]], ["Val_list", ["max", "avg"]], ["Orderby_desc", 
        ["sum", "Groupby"]],["abs",["sub"]], ["SGLProject",["abs"]], ["SGLProject",["eq"]], ["length",["Value"]], ["SGLProject",["add"]], ["SGLProject",["sub"]],["SGLProject",["gt"]],["SGLProject",["lt"]],["SGLProject",["neq"]],["Orderby_desc", ["add", "Project"]],["Project", ["add", "Selection"]],["Project", ["sum", "Table"]], ["Project", ["sub", "Table"]], ["Groupby", ["Value", "Project"]], ["eq", ["Value", "sub"]], ["Selection", ["Or", "Table"]], ["Val_list", ["max", "sum"]], ["Table", ["Subquery"]], ["avg", ["Value"]], ["intersect", ["Subquery", "Subquery"]], ["gte", ["Value", "Subquery"]], ["And", ["gt", "neq"]], ["nin", ["Value", "Subquery"]], ["Val_list", ["Val_list", "avg"]], ["And", ["gt", "eq"]], ["And", ["And", "gt"]], ["Project", ["Val_list", "Table"]], ["Val_list", ["count", "avg"]], ["Project", ["min", "Selection"]],["add", ["Limit", "Value"]],["And", ["neq", "neq"]],["Project", ["sub", "Selection"]],
        ["gt", ["abs", "Value"]],
        ["And", ["lte", "eq"]],
        ["SGLProject", ["gte"]],
        ["gte", ["Project", "Value"]],
        ["eq", ["Project", "Value"]],
        ["Project", ["add", "Table"]],
        ["eq", ["sub", "Value"]],
        ["And", ["neq", "lt"]],
        ["sub", ["max", "min"]],
        ["And", ["lt", "gt"]],
        ["And", ["neq", "gt"]],
        ["Limit", ["Value", "Project"]],
        ["lt", ["Project", "Project"]],
        ["gte", ["sub", "Value"]],
        ["sub", ["min", "Value"]],
        ["And", ["neq", "gte"]],
        ["eq", ["count", "Subquery"]],
        ["sum", ["add"]],
        ["Project", ["abs", "Selection"]],
        ["eq", ["Value", "div"]],
        ["div", ["Project", "Value"]],
        ["Project", ["abs", "Table"]],
        ["Project", ["gt", "Selection"]],
        ["gt", ["count", "Subquery"]],
        ["Or", ["lt", "And"]],
        ["eq", ["Project", "Project"]],
        ["Or", ["lte", "lte"]],
        ["sum", ["sub"]],
        ["sub", ["max", "add"]],
        ["add", ["min", "Value"]],
        ["Orderby_desc", ["abs", "Project"]],
        ["lt", ["Project", "Value"]],
        ["lte", ["abs", "Value"]],
        ["sub", ["sum", "sum"]],
        ["sub", ["Project", "Limit"]],
        ["Orderby_desc", ["div", "Project"]],
        ["div", ["Value", "Value"]],
        ["sub", ["max", "max"]],
        ["max", ["sub"]],
        ["Orderby_desc", ["sub", "Project"]],
        ["eq", ["add", "Value"]],
        ["sub", ["Limit", "Limit"]],
        ["SGLProject", ["lte"]],
        ["lte", ["Project", "Value"]],
        ["max", ["abs"]],
        ["lt", ["add", "Value"]],
        ["And", ["gte", "lt"]],
        ["min", ["add"]],
        ["max", ["add"]],
        ["gt", ["Limit", "Limit"]],
        ["eq", ["abs", "Value"]],
        ["gt", ["add", "Value"]],
        ["SGLProject", ["And"]],
        ["And", ["SGLProject", "SGLProject"]],
        ["count", ["gt"]],
        ["gte", ["add", "Value"]],
        ["Project", ["div", "Selection"]],
        ["sub", ["Value", "Project"]],
        ["Project", ["eq", "Selection"]],
        ["gt", ["sub", "Value"]],
        ["gt", ["Value", "Project"]],
        ["sub", ["Value", "min"]],
        ["min", ["abs"]],
        ["And", ["lt", "gte"]],
        ["eq", ["Value", "mul"]],
        ["mul", ["Project", "Value"]],
        ["abs", ["Value"]],
        ["Project", ["length", "Selection"]],
        ["gt", ["min", "Value"]],
        ["And", ["neq", "lte"]],
        ["And", ["lte", "lte"]],
        ["gte", ["abs", "Value"]],
        ["lt", ["count", "Subquery"]],
        ["neq", ["Project", "Project"]],
        ["Or", ["And", "And"]],
        ["eq", ["sub", "Subquery"]],
        ["And", ["lte", "gte"]],
        ["Or", ["lt", "lt"]],
        ["Orderby_asc", ["sub", "Project"]],
        ["lt", ["abs", "Value"]],
        ["gte", ["Value", "mul"]],
        ["mul", ["Value", "Value"]],
        ["gte", ["Value", "sub"]],
        ["add", ["sum", "sum"]],
        ["sub", ["Value", "add"]],
        ["lt", ["sub", "Value"]],
        ["avg", ["add"]],
        ["eq", ["min", "Value"]],
        ["eq", ["max", "Value"]],
        ["julianday", ["Value"]],
        ["sub", ["julianday","julianday"]],
        ["Project", ["julianday","Selection"]],
        ["min", ["sub"]],
        ["Or", ["Or", "eq"]],
        ["And", ["Or", "neq"]],
        ["sum", ["distinct"]],
        ["And", ["Or", "gt"]],
        ["And", ["Or", "lte"]],
        ["And", ["gt", "Or"]],
        ["eq", ["julianday", "sub"]],
        ["Selection", ["Is", "Table"]],
        ["Is", ["Value", "Value"]],
        ["And", ["eq", "Not"]],
        ["Not", ["Value", "Value"]],
        ["And", ["Or", "eq"]],
        ["And", ["And", "Is"]],
        ["And", ["Not", "Not"]],
        ["Selection", ["Not", "Table"]],
        ["And", ["Not", "Is"]],
        ["And", ["Not", "neq"]],
        ["And", ["Is", "lt"]],
        ["And", ["Not", "gt"]],
        ["lte", ["add", "Value"]]]
        """
        rules = json.loads(RULES_values)
        
        #for r in rules:
        #    print(r)


        if has_gold:
            try:
                #tree_dict = msp.parse(sql)
                tree_dict_values = msp.parse(sql_with_values)
                #tree_dict_values =
            except msp.ParseException as e:
                print(f"could'nt create AST for:  {sql_with_values}")
                return None
            #print(sql)
            #print(tree_dict["query"])
            #print(sql_with_values)
            tree_obj = ra_preproc.ast_to_ra(tree_dict_values["query"])
            tree_obj_values = ra_preproc.ast_to_ra(tree_dict_values["query"])
            #print(tree_obj)
            #print(tree_dict_values["query"])
            #print(node_util.print_tree(tree_obj))
            for pre, fill, node in anytree.RenderTree(tree_obj):
                ll = list()
                for child in anytree.PreOrderIter(node, maxlevel=2):
                    ll.append(child.name)
                new_l = list()
                if len(ll) > 1:
                    new_l.append(ll[0])
                    new_l.append(ll[1:])
                    #if 'Not' in new_l:
                    #    print(new_l)
                    #    print(node_util.print_tree(tree_obj))
                    #    input()
                    if new_l not in rules:
                        all_list.append(new_l)
                        
                        #print(new_l)
                        #print(node_util.print_tree(tree_obj))
                    #input()
            

            #arit_list = anytree.search.findall(
            #    tree_obj, filter_=lambda x: x.name in ["sub", "add"]
            #)  # TODO: fixme
            #haslist_list = anytree.search.findall(
            #    tree_obj,
            #    filter_=lambda x: hasattr(x, "val") and isinstance(x.val, list),
            #)

            #print(arit_list)
            #print(haslist_list)


            #if arit_list or haslist_list:
            #    print(f"could'nt create RA for:  {sql}")
            #    input()
            #    return None
            
            if self.value_pred:
                for a, b in zip(tree_obj_values.leaves, tree_obj.leaves):
                    if b.name == "Table" or b.name== "sub" or ("." in str(b.val)):
                        continue
                    b.val = a.val
                    if (
                        isinstance(a.val, int) or isinstance(a.val, float)
                    ) and b.parent.name == "literal":
                        parent_node = b.parent
                        parent_node.children = []
                        parent_node.name = "Value"
                        parent_node.val = b.val
            
            for leaf in tree_obj.leaves:
                if type(leaf.val) == str and  leaf.val != '' and leaf.val[0] == 'c' and leaf.val.split('_')[0][1:].isnumeric():
                    idx = int(leaf.val.split('_')[0][1:]) - 1
                    col_name = raw_col[idx].replace(' ', '_').replace('\n', '_').lower()
                    tps = '.'.join(leaf.val.split('_')[1:])
                    if tps != '':
                        leaf.val = 'w.' + col_name + '.' + tps
                    else:
                        leaf.val = 'w.' + col_name
                if not self.value_pred and node_util.is_number(leaf.val):
                    leaf.val = "value"

            leafs = list(set(node_util.get_leafs(tree_obj)))
            hash_gold_levelorder, hash_gold_tree = self._init_fields(tree_obj)

            fields.update(
                {
                    "hash_gold_levelorder": ArrayField(
                        hash_gold_levelorder, padding_value=-1, dtype=np.int64
                    ),
                    "hash_gold_tree": ArrayField(
                        np.array(hash_gold_tree), padding_value=-1, dtype=np.int64
                    ),
                    "gold_sql": MetadataField(sql_with_values),
                    "tree_obj": MetadataField(tree_obj),
                }
            )

        desc = self.enc_preproc.get_desc(tokenized_utterance, db_id)
        #print(desc)

        entities, added_values, relation = self.extract_relation(desc)


        question_concated = [[x] for x in tokenized_utterance[1:-1]]
        schema_tokens_pre, schema_tokens_pre_mask = table_text_encoding(
            entities[len(added_values) :]
        )

        schema_size = len(entities)
        schema_tokens_pre = added_values  + schema_tokens_pre

        schema_tokens = [
            [y for y in x if y.text not in ["_"]]
            for x in [self._tokenizer.tokenize(x)[1:-1] for x in schema_tokens_pre]
        ]

        entities_as_leafs = [x.split(":")[0] for x in entities[len(added_values):]]
        entities_as_leafs = added_values  + entities_as_leafs


        orig_entities = [x for x in entities_as_leafs]
        entities_as_leafs_hash, entities_as_leafs_types = self.hash_schema(
            entities_as_leafs, added_values
        )

        fields.update(
            {
                "relation": ArrayField(relation, padding_value=-1, dtype=np.int32),
                "entities": MetadataField(entities_as_leafs),
                 "orig_entities": MetadataField(orig_entities),
                 "leaf_hash": ArrayField(
                    entities_as_leafs_hash, padding_value=-1, dtype=np.int64
                ),
                "leaf_types": ArrayField(
                    entities_as_leafs_types,
                    padding_value=self._type_dict["nan"],
                    dtype=np.int32,
                )
            })

        if has_gold:
            leaf_indices, is_gold_leaf, depth = self.is_gold_leafs(
                tree_obj, leafs, schema_size, entities_as_leafs, raw_col
            )

            fields.update(
                {
                    "is_gold_leaf": ArrayField(
                        is_gold_leaf, padding_value=0, dtype=np.int32
                    ),
                    "leaf_indices": ArrayField(
                        leaf_indices, padding_value=-1, dtype=np.int32
                    ),
                    "depth": ArrayField(depth, padding_value=0, dtype=np.int32),
                }
            )

        utt_len = len(tokenized_utterance[1:-1])
        if self.value_pred:
            span_hash_array = self.hash_spans(tokenized_utterance)
            fields["span_hash"] = ArrayField(
                span_hash_array, padding_value=-1, dtype=np.int64
            )

        if has_gold and self.value_pred:
            value_list = np.array(
                [self.hash_text(x) for x in node_util.get_literals(tree_obj)],
                dtype=np.int64,
            )
            #print(sql_with_values)
            #print(node_util.get_literals(tree_obj))
            #print(value_list)
            #print(tokenized_utterance)
            #print(span_hash_array)
            is_gold_span = np.isin(span_hash_array.reshape([-1]), value_list).reshape(
                [utt_len, utt_len]
            )
            #print(is_gold_span)
            #input()
            fields["is_gold_span"] = ArrayField(
                is_gold_span, padding_value=False, dtype=np.bool
            )

        enc_field_list = []
        offsets = []
        mask_list = (
            [False]
            + ([True] * len(question_concated))
            + [False]
            + ([True] * len(added_values))
            + schema_tokens_pre_mask
            + [False]
        )
        for mask, x in zip(
            mask_list,
            [[self.cls_token]]
            + question_concated
            + [[self.eos_token]]
            + schema_tokens
            + [[self.eos_token]],
        ):
            start_offset = len(enc_field_list)
            enc_field_list.extend(x)
            if mask:
                offsets.append([start_offset, len(enc_field_list) - 1])

        fields["lengths"] = ArrayField(
            np.array(
                [
                    [0, len(question_concated) - 1],
                    [len(question_concated), len(question_concated) + schema_size - 1],
                ]
            ),
            dtype=np.int32,
        )
        fields["offsets"] = ArrayField(
            np.array(offsets), padding_value=0, dtype=np.int32
        )
        fields["enc"] = TextField(enc_field_list)

        ins = Instance(fields)
        return ins

    def extract_relation(self, desc):
        def parse_col(col_list):
            #print(col_list)
            col_type = col_list[0]
            col_name = "_".join(col_list[1:]).lower()
            table = 'w'
            return f'{table}.{col_name}:{col_type.replace("<type: ","")[:-1]}'

        question_concated = [x for x in desc["question"]]
        col_concated = [parse_col(x) for x in desc["columns"]]
        table_concated = ["_".join(x).lower() for x in desc["tables"]]
        enc = question_concated + col_concated + table_concated
        relation = self.enc_preproc.compute_relations(
            desc,
            len(enc),
            len(question_concated),
            len(col_concated),
            range(len(col_concated) + 1),
            range(len(table_concated) + 1),
        )
        unsorted_entities = col_concated + table_concated
        rel_dict = defaultdict(dict)
        # can do this with one loop
        for i, x in enumerate(list(range(len(question_concated))) + unsorted_entities):
            for j, y in enumerate(
                list(range(len(question_concated))) + unsorted_entities
            ):
                rel_dict[x][y] = relation[i, j]
        #entities_sorted = sorted(list(enumerate(unsorted_entities)), key=lambda x: x[1])
        entities = [x for x in unsorted_entities]
        if self.value_pred:
            added_values = [
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "*",
                "id",
                "null",
            ]
        else:
            added_values = ["value"]
        entities = added_values + entities
        new_enc = list(range(len(question_concated))) + entities
        new_relation = np.zeros([len(new_enc), len(new_enc)])
        for i, x in enumerate(new_enc):
            for j, y in enumerate(new_enc):
                if y in added_values or x in added_values:
                    continue
                new_relation[i][j] = rel_dict[x][y]
        return entities, added_values, new_relation

    def is_gold_leafs(self, tree_obj, leafs, schema_size, entities_as_leafs, raw_col):
        enitities_leaf_dict = {ent: i for i, ent in enumerate(entities_as_leafs)}
        indices = []
        #print(leafs, enitities_leaf_dict)
        for leaf in leafs:
            leaf = str(leaf).lower()
            if leaf in enitities_leaf_dict:
                indices.append(enitities_leaf_dict[leaf])
            else:
                if leaf[:2] == 'w.':
                    print('????????????')
                    print(leaf)
                    print(leafs)
                    print(enitities_leaf_dict)
                    #input()
        is_gold_leaf = np.array(
            [1 if (i in indices) else 0 for i in range(schema_size)]
        )
        indices = np.array(indices)
        depth = np.array([1] * max([leaf.depth for leaf in tree_obj.leaves]))
        return indices, is_gold_leaf, depth

    def hash_schema(self, leaf_text, added_values=None):
        beam_hash = []
        beam_types = []

        for leaf in leaf_text:
            leaf = leaf.strip()
            if leaf == 'w':
                leaf_node = Node("Table", val=leaf)
                type_ = self._type_dict["Table"]
            else:
            #if (len(leaf.split(".")) == 2) or ("*" == leaf) or leaf in added_values:
                leaf_node = Node("Value", val=leaf)
                type_ = self._type_dict["Value"]
            #else:
            #    leaf_node = Node("Table", val=leaf)
            #    type_ = self._type_dict["Table"]
            leaf_node = self.hasher.add_hash_att(leaf_node, self._type_dict)
            beam_hash.append(leaf_node.hash)
            beam_types.append(type_)
        beam_hash = np.array(beam_hash, dtype=np.int64)
        beam_types = np.array(beam_types, dtype=np.int32)
        return beam_hash, beam_types

    def hash_text(self, text):
        return self.hasher.set_hash([self._type_dict["Value"], hashing.dethash(text)])

    def hash_spans(self, tokenized_utterance):
        utt_idx = [x.text_id for x in tokenized_utterance[1:-1]]
        utt_len = len(utt_idx)
        span_hash_array = -np.ones([utt_len, utt_len], dtype=int)
        for i_ in range(utt_len):
            for j_ in range(utt_len):
                if i_ <= j_:
                    span_text = self._tokenizer.tokenizer.decode(utt_idx[i_ : j_ + 1])
                    span_hash_array[i_, j_] = self.hash_text(span_text)
        return span_hash_array

    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["enc"].token_indexers = self._utterance_token_indexers


def table_text_encoding(entity_text_list):
    token_list = []
    mask_list = []
    for i, curr in enumerate(entity_text_list):
        if ":" in curr:  # col
            token_list.append(curr)
            if (i + 1) < len(entity_text_list) and ":" in entity_text_list[i + 1]:
                token_list.append(",")
            else:
                token_list.append(")\n")
            mask_list.extend([True, False])
        else:
            token_list.append(curr)
            token_list.append("(")
            mask_list.extend([True, False])

    return token_list, mask_list
