from django.shortcuts import render
import numpy as np
from joblib import load
from scipy.sparse import csr_matrix
# from svmproupdated import predictor, preprocess_model, svm_model, vectorizer_model, stemming



preprocess_model = load('./rabin.joblib')

svm_model = load('./anukul.joblib')

vectorizer_model = load('./sujata.joblib')


def predictor(request):
    if request.method == 'POST':
        cmnt = request.POST['comment_box']
        print("Comment:", cmnt)
        cmnt_preprocess = preprocess_model(cmnt)
        print("Preprocessed comment:", cmnt_preprocess)
       # cmnt_preprocess = ' '.join(cmnt_preprocess) # convert list to string
        cmnt_vectorize = vectorizer_model.transform([cmnt_preprocess])
       # cmnt_vectorize = csr_matrix(cmnt_vectorize)
        print("Vectorized comment:", cmnt_vectorize)
        cmnt_predict = svm_model.predict(cmnt_vectorize)
        print("Prediction:", cmnt_predict)
        if cmnt_predict == 0:
            return render(request, 'home.html', {'result': 'Your comment is acceptable. Keep it up!'})
        else:
            return render(request, 'home.html', {'result':  'This comment seems to be toxic. Please refrain from posting harmful content' })
    return render(request, 'home.html')


# def predictor(request):
#     result = None
#     if request.method == 'POST':
#         cmnt = request.POST['comment_box']
#         print(cmnt)
#         cmnt_preprocess = preprocess_model(cmnt)
#         print(cmnt_preprocess)
#         cmnt_preprocess = ' '.join(cmnt_preprocess) # convert list to string
        
#         cmnt_vectorize = vectorizer_model.transform([cmnt_preprocess])
       
#         cmnt_vectorize = csr_matrix(cmnt_vectorize)
        
#         cmnt_predict = svm_model.predict(cmnt_vectorize)
#         print(cmnt_predict)
#         if cmnt_predict == 0:
#             result = 'Your comment is acceptable. Keep it up!'
#         else:
#             result = 'This comment seems to be toxic. Please refrain from posting harmful content'
#     return render(request, 'home.html', {'result': result})
