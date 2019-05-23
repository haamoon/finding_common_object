# coding: utf-8
import tensorflow as tf
from unary_module import UnaryModule
from cosine_similarity import CosineSimilarity
from relation_module import RelationModule


r = RelationModule()
cs = CosineSimilarity()
m = 4
l = 5
mbs = 2
k = 4
d = 10
u = UnaryModule(r, cs, topk=4)
fea0 = tf.random.uniform((mbs*k, m, d))
neg_fea = tf.random.uniform((mbs, l, d))
s = u([fea0, neg_fea])
from IPython import embed;embed()
