import bayes
import numpy as np

listOposts, listCalsses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOposts)

trainMat = []
for postinDoc in listOposts:
    trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
p0V, p1V, pAb = bayes.trainNB0(trainMat, listCalsses)
# print(p0V)
# print(p1V)
print(np.log(0.5))
bayes.testingNB()

