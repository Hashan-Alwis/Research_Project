from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
from check_attention import generate_frames
# from sentence_transformers import SentenceTransformer, util


# model = SentenceTransformer("saved_transformer_model")  # Load a pre-trained sentence transformer model

# def sentence_similarity(sent1, sent2):
#     embeddings = model.encode([sent1, sent2])  # Convert sentences to embeddings
#     similarity = util.cos_sim(embeddings[0], embeddings[1]).item()  # Compute cosine similarity
#     percentage = round(similarity * 100, 2)  # Convert to percentage
#     return percentage


app = Flask(__name__)
CORS(app)

# @app.route('/')
# def index():
#     return render_template('test.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='text/event-stream')


# @app.route('/similarity', methods=['POST'])
# def get_similarity():
#     data = request.json
#     correct_answer = data.get("correct_answer", "")
#     given_answer = data.get("given_answer", "")
    
#     if not correct_answer or not given_answer:
#         return jsonify({"error": "Both correct_answer and given_answer are required."}), 400
    
#     score = sentence_similarity(correct_answer, given_answer)
#     return jsonify({"similarity_score": score})







if __name__ == "__main__":
    app.run(debug=True, port=5000)