
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
from keras.models import load_model

from flask import Flask
from flask_restful import Api, Resource, reqparse

app =Flask(__name__)
api =Api(app)

model = load_model('gru_phishing_emails_data.h5')

post_args = reqparse.RequestParser()
post_args.add_argument('email', type =str, help='email is required', required=True)

# preprocess the email 
def preprocessEmail(email):
    max_len = 150

    tokenizer_file = "tokenizer.pkl"
    with open(tokenizer_file, 'rb') as handle:
        tok = pickle.load(handle)

    email_sequences = tok.texts_to_sequences([email])
    email_sequences_matrix = pad_sequences(email_sequences,maxlen=max_len)

    return email_sequences_matrix

class Predict(Resource):
    def post(self):
        args =post_args.parse_args()

        # preprocess the email content 
        preprocessedEmail = preprocessEmail(args['email'])

        # Make predictions
        pred_prob = model.predict(preprocessedEmail)
        pred = (pred_prob > 0.5).astype(int)  

        predicted_label = "Phishing Email" if pred[0][0] == 0 else "Safe Email"

        return {'prediction is':f'{predicted_label}'}


api.add_resource(Predict,'/predict')


if __name__ =='__main__':
    app.run(debug=False,host='0.0.0.0')