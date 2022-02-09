#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 10:19:58 2021

@author: asep

reference from
[1] https://github.com/nju-websoft/ESBM/tree/master/v1.2
[2] https://github.com/nju-websoft/DeepLENS/blob/master/code/train_test.py
"""

import math

class NDCG:
    """NDCG stand for Normalized Discounted Cumulative Gain"""
    def get_score(self, gold_summaries, triples_rank):
        """ the score is to measure the quality of a set of search results"""
        grade_list, triple_grade = self.set_gold_triples_dict(gold_summaries)
        dcg = 0
        idcg = 0
        max_rank_pos = len(triples_rank)
        max_ideal_pos = len(grade_list)
        for pos in range(1, max_rank_pos+1):
            t_pos = triples_rank[pos-1]
            rel = 0
            try:
                rel = triple_grade[t_pos]
            except Exception:
                pass
            dcg_item = rel/math.log(pos + 1, 2)
            dcg += dcg_item
            if pos <= max_ideal_pos:
                ideal_rel = grade_list[pos-1]
                idcg += ideal_rel/math.log(pos + 1, 2)
        return dcg/idcg
    @staticmethod
    def set_gold_triples_dict(gold_summaries):
        """To create gold triples dictionary"""
        triple_grade = {}
        for triple_gold_sum in gold_summaries:
            for triple in triple_gold_sum:
                if triple not in triple_grade:
                    triple_grade[triple] = 1
                else:
                    triple_grade[triple] = triple_grade[triple]+1
        grade_list = list(triple_grade.values())
        grade_list.sort(reverse=True)
        return grade_list, triple_grade
    def __repr__(self):
        return self.__class__.__name__
    