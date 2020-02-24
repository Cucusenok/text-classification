# определим датасет
some_texts = [
   'внедрение зависимости', 
   'хорошо, компьютеры это круто'
]
 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(some_texts)]
 
model = Doc2Vec(documents, vector_size=5, workers=4, epochs=3)


# сохранение модели для дальнейшего использования
model.save("my_doc2vec_model")
# загрузка модели
model = Doc2Vec.load("my_doc2vec_model")
# нахождение наиболее похожего документа
vector_to_search = model.infer_vector(["ищем", "похожий", "текст"])
# три наиболее похожих
similar_documents = model.docvecs.most_similar([vector_to_search], topn=3)
for s in similar_documents:
    print(some_texts[s[0]])
    
    
from sklearn.ensemble import RandomForestClassifier

textLabels = [ 'зависимость', 'компьютеры' ] 
clf = RandomForestClassifier()
clf.fit([model.infer_vector([x.words]) for x in documents], textLabels)
  
res = clf.predict([model.infer_vector(['компьютер это круто'])])
print(res)