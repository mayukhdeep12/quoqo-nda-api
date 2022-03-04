data = pd.read_csv("ham_spam.csv")

data.columns = ["raw_documents"]
data


cv = pickle.load(open('cv.pkl', 'rb'))
clf = pickle.load(open('spam_ham.pkl', 'rb'))


def vect(text):
    print(text)
    data_vect = cv.transform(text)
    return data_vect

data_vect = data.apply(vect)

data_predict = clf.predict(data_vector)

def pred(vect):
    print(vect)
    pred_text = clf.predict(vect)
    return pred_text

data_predict = data_vect.apply(pred)



data["predict"] = pd.DataFrame(data_predict["raw_documents"])
