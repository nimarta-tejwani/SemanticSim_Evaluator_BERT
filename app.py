from flask import Flask, render_template, request
from semantic import calculate_semantic_similarity, load_word_embeddings

app = Flask(__name__)


# Home page
@app.route('/', methods=['GET', 'POST'])
def home():
    similarity_result = None

    if request.method == 'POST':
        text1 = request.form['text1']
        text2 = request.form['text2']
        sentence_embedding_function = load_word_embeddings()

        similarity_percentage = calculate_semantic_similarity(text1, text2, sentence_embedding_function)
        formatted_similarity = "{:.2f}".format(similarity_percentage)
        similarity_result = f"Similarity Percentage: {formatted_similarity}%"

    return render_template('index.html', similarity_result=similarity_result)

if __name__ == '__main__':
    app.run(debug=True)
