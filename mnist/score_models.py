import score
from mymodels import knn
from mymodels import mlp_notf
from mymodels import mlp
from mymodels import cnn

#score.score(knn.knn, k_max=10)
#score.score(mlp_notf.mlp_notf, batch_size=50)
#score.score(mlp.mlp, batch_size=50)
score.score(cnn.cnn)
