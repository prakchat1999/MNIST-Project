# matplotlib is a lib which allows us to display images
import matplotlib.pyplot as plt

# sklearn.datasets imports many known datasets
from sklearn import datasets
# sklearn.svm (support vector machine) a form of machine learning
from sklearn import svm

digits = datasets.load_digits()
#  digits.data= a collection of already learned data for that index of the image
#  digits.target = what that specific data corresponds to for that index
#  digits.images[0] = the image pixels of that specific index in array form

clf = svm.SVC(gamma = 0.001 , C = 100)
#clf is classifier
x,y = digits.data[:-1], digits.target[:-1] #test set
#storing data and answers in x and y respectively

clf.fit(x,y) #MAIN LEARN METHOD-----will learn and put it inside clf
#will learn until the last element and the learn is used to predict the last element

print("Prediction:", clf.predict(digits.data[[-1]]))
#clf.predict will predict what the answer is
plt.imshow(digits.images[-1], cmap = plt.cm.gray_r)#, interpolation = "nearest")
#changing the colours to gray so that it looks better
plt.show()

#RISHAB IS A DONKEY---------
