"""Flask interface: GET / and POST /recommend."""
from pathlib import Path
from flask import Flask, render_template, request
from recommender import load_model, recommend_by_id

ROOT = Path(__file__).resolve().parent


def create_app(model_dir=ROOT):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024
    movies, similarity = load_model(model_dir)
    choices = movies.sort_values(['title', 'year', 'id']).to_dict('records')

    def page(results=None, selected=None, error=None, status=200):
        return render_template('index.html', movies=choices, results=results,
                               selected=selected, error=error), status

    @app.get('/')
    def index():
        return page()

    @app.route('/recommend', methods=['GET', 'POST'])
    def recommend_route():
        if request.method == 'GET':
            return page()
        try:
            movie_id = int(request.form.get('movie_id', ''))
            n = int(request.form.get('count', '5'))
            results = recommend_by_id(movie_id, movies, similarity, n)
            selected = movies.loc[movies.id.eq(movie_id)].iloc[0].to_dict()
        except (ValueError, TypeError):
            return page(error='Choose a listed movie and a recommendation count from 1 to 20.', status=400)
        return page(results=results.to_dict('records'), selected=selected)

    return app


app = create_app()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
