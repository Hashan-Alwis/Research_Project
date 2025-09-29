from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = SentenceTransformer("saved_transformer_model")  

def sentence_similarity(sent1, sent2):
    embeddings = model.encode([sent1, sent2])  
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()  
    percentage = round(similarity * 100, 2)  
    return percentage

@app.route('/similarity', methods=['POST'])
def get_similarity():
    data = request.json
    correct_answer = data.get("correct_answer", "")
    given_answer = data.get("given_answer", "")
    
    if not correct_answer or not given_answer:
        return jsonify({"error": "Both correct_answer and given_answer are required."}), 400
    
    score = sentence_similarity(correct_answer, given_answer)
    return jsonify({"similarity_score": score})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
