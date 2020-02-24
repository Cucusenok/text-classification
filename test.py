from classes.helpers.File import File
from classes.nlp.WordTokenaizer import RussianWordTokenaizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
print('=========================================')


text1 = File("data/russian/posts/oop.txt").getContent()
text2 = File("data/russian/posts/ci_di.txt").getContent()
text3 = File("data/russian/posts/yandex.txt").getContent()

# определим датасет
some_texts = [text1, text2, text3]

testText = File("data/russian/posts/oop_test.txt").getContent()



textLabels = ['oop', 'ci/di', 'yandex']


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(some_texts)]
model = Doc2Vec(documents, vector_size=300, workers=4, epochs=3)

# сохранение модели для дальнейшего использования
model.save("my_doc2vec_model")
# загрузка модели
model = Doc2Vec.load("my_doc2vec_model")
# нахождение наиболее похожего документа
vector_to_search = model.infer_vector([testText])
# три наиболее похожих
similar_documents = model.docvecs.most_similar([vector_to_search], topn=3)

    
    
from sklearn.ensemble import RandomForestClassifier
 
clf = RandomForestClassifier(n_estimators=100)
clf.fit([model.infer_vector([x.words]) for x in documents], textLabels)
  
russianTextTokenaizer = RussianWordTokenaizer(testText).make()

modelInfir = model.infer_vector(russianTextTokenaizer)
modelInfir2 = model.infer_vector(russianTextTokenaizer)

#print(modelInfir)

res = clf.predict([modelInfir]) #получение предсказанного класса
resProba = clf.predict_proba([modelInfir]) #вероятности по классам
print('Точность модели:', 
      classification_report(['oop'], clf.predict([modelInfir])),
      )

print(accuracy_score(['oop'], clf.predict([modelInfir])))

print('Предсказанный класс: ', res) 
print('Вероятности по классам: ', resProba) 

