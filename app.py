from flask import Flask, make_response, request
import io
from io import StringIO
import csv
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer



app = Flask(__name__)

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")


@app.route('/')
def form():
    return """
        <html>
            <body>
                <form action="/transform" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" class="btn btn-block"/>
                    </br>
                    </br>
                    <button type="submit" class="btn btn-primary btn-block btn-large">Predict</button>
                </form>
            </body>
        </html>
    """
@app.route('/transform', methods=["POST"])
def transform_view():
    file_path = request.files['data_file']
    if not file_path:
        return "No file"

    stream = io.StringIO(file_path.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    
    print(csv_input)
    for row in csv_input:
        print(row)

    stream.seek(0)
    result = transform(stream.read())

    data = pd.read_csv(StringIO(result))
    data.columns = ["raw_documents"]
    
    cv = pickle.load(open('cv.pkl', 'rb'))
    clf = pickle.load(open('spam_ham.pkl', 'rb'))
    def vect(text):
        print(text)
        data_vect = cv.transform(text)
        return data_vect

    data_vect = data.apply(vect)

    def pred(vect):
        print(vect)
        pred_text = clf.predict(vect)
        return pred_text

    data_predict = data_vect.apply(pred)

    data["predict"] = pd.DataFrame(data_predict["raw_documents"])
    
    data.to_json('results.json', orient = 'split', compression = 'infer', index = 'true')

    response = make_response(data.to_csv())
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    return response

if __name__ == "__main__":
    app.run()