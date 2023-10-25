import small_text.query_strategies.strategies

dict_module = small_text.query_strategies.strategies.__dict__
queries = list(dict_module.keys())[17:]
queries = {k:v for k,v in dict_module.items() if k in queries}
# QueryStrategy
# RandomSampling
# ConfidenceBasedQueryStrategy
# BreakingTies
# LeastConfidence
# PredictionEntropy
# SubsamplingQueryStrategy
# EmbeddingBasedQueryStrategy
# EmbeddingKMeans
# ContrastiveActiveLearning
# DiscriminativeActiveLearning
# SEALS