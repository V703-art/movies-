"""Reusable preprocessing, model building and content recommendation functions."""
from pathlib import Path
from collections import Counter
import json
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(data_dir):
    """Retain all valid movies; never merge by potentially ambiguous title."""
    data_dir = Path(data_dir)
    movies = pd.read_csv(data_dir / 'tmdb_5000_movies.csv', dtype=str, keep_default_na=False)
    credits = pd.read_csv(data_dir / 'tmdb_5000_credits.csv', dtype=str,
                          keep_default_na=False, usecols=['movie_id', 'title', 'cast', 'crew'])
    required = {'id', 'title', 'genres', 'keywords', 'overview', 'release_date'}
    if required - set(movies):
        raise ValueError(f'Missing movie columns: {sorted(required - set(movies))}')
    report = {'movie_rows_raw': len(movies), 'credit_rows_raw': len(credits)}
    for frame, key in [(movies, 'id'), (credits, 'movie_id')]:
        ids = pd.to_numeric(frame[key], errors='coerce')
        valid = ids.notna() & (ids % 1 == 0) & (ids > 0)
        frame.drop(index=frame.index[~valid], inplace=True)
        frame[key] = ids.loc[valid].astype('int64')
    movies = movies.loc[movies.title.str.strip().ne('')].copy()
    if movies.id.duplicated().any() or credits.movie_id.duplicated().any():
        raise ValueError('Duplicate movie IDs found; resolve these before merging.')
    report['valid_credit_rows'] = len(credits)
    report['discarded_credit_rows'] = report['credit_rows_raw'] - len(credits)
    merged = movies.merge(credits[['movie_id', 'cast', 'crew']], how='left',
                          left_on='id', right_on='movie_id', validate='one_to_one', indicator=True)
    report['movies_without_credit_row'] = int(merged['_merge'].eq('left_only').sum())
    merged = merged.drop(columns=['movie_id', '_merge']).fillna('').reset_index(drop=True)
    report['movies_retained'] = len(merged)
    if len(merged) < 2:
        raise ValueError('At least two movies are required.')
    return merged, report


def build_tags(movies, report):
    """Malformed JSON becomes an empty list and is counted, not guessed."""
    movies = movies.copy()
    errors = Counter()
    def parse(value, field):
        if not value:
            return []
        try:
            result = json.loads(value)
            if not isinstance(result, list) or any(not isinstance(x, dict) for x in result):
                raise ValueError('Expected a list of objects')
            return result
        except (ValueError, TypeError):
            errors[field] += 1
            return []
    def names(items):
        return [x['name'].strip() for x in items if isinstance(x.get('name'), str) and x['name'].strip()]
    def atom(value):
        return re.sub(r'\W+', '', value.casefold())
    for field in ['genres', 'keywords', 'cast', 'crew']:
        movies[field + '_parsed'] = [parse(v, field) for v in movies[field]]
    movies['genre_names'] = movies.genres_parsed.map(names)
    movies['cast_names'] = movies.cast_parsed.map(lambda x: names(x[:3]))
    movies['director_names'] = movies.crew_parsed.map(lambda x: names([p for p in x if p.get('job') == 'Director']))
    def tags(row):
        structured = row['genre_names'] + names(row['keywords_parsed']) + row['cast_names'] + row['director_names']
        # Keep multiword entities together: science fiction -> sciencefiction.
        return ' '.join([str(row['overview']).casefold()] + [atom(x) for x in structured if atom(x)])
    movies['tags'] = movies.apply(tags, axis=1)
    movies['year'] = movies.release_date.str.extract(r'^(\d{4})', expand=False).fillna('Unknown year')
    report['malformed_json_by_field'] = {f: errors[f] for f in ['genres', 'keywords', 'cast', 'crew']}
    report['movies_without_usable_cast'] = int(movies.cast_names.map(len).eq(0).sum())
    report['movies_without_usable_director'] = int(movies.director_names.map(len).eq(0).sum())
    report['empty_tags'] = int(movies.tags.str.strip().eq('').sum())
    keep = ['id', 'title', 'year', 'overview', 'genre_names', 'cast_names', 'director_names', 'tags']
    return movies[keep].reset_index(drop=True)


def train_model(movies):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000,
                                 sublinear_tf=True, dtype=np.float32)
    vectors = vectorizer.fit_transform(movies.tags)
    similarity = cosine_similarity(vectors).astype(np.float32)
    np.clip(similarity, 0, 1, out=similarity)
    return vectorizer, vectors, similarity


def save_model(movies, similarity, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Store exact top-20 neighbours instead of the full quadratic matrix.
    # App requests are limited to 20 results; lower-ranked pairs are omitted.
    k = min(20, len(movies) - 1)
    indices = np.empty((len(movies), k), dtype=np.int32)
    scores = np.empty((len(movies), k), dtype=np.float32)
    for i, row in enumerate(similarity):
        order = np.argsort(-row, kind='stable')
        order = order[order != i][:k]
        indices[i] = order
        scores[i] = row[order]
    compact = {'format': 'top20-v1', 'movie_ids': movies.id.to_numpy(),
               'indices': indices, 'scores': scores,
               'diagonal': np.diag(similarity).astype(np.float32)}
    # Only load pickle files produced by this project or another trusted source.
    for name, value in [('model.pkl', movies), ('similarity.pkl', compact)]:
        with (directory / name).open('wb') as handle:
            pickle.dump(value, handle, protocol=4)


def load_model(directory):
    directory = Path(directory)
    try:
        with (directory / 'model.pkl').open('rb') as handle:
            movies = pickle.load(handle)
        with (directory / 'similarity.pkl').open('rb') as handle:
            similarity = pickle.load(handle)
    except FileNotFoundError as exc:
        raise RuntimeError('Model files are missing. Run python train.py or all notebook cells first.') from exc
    if isinstance(similarity, dict) and similarity.get('format') == 'top20-v1':
        compact = similarity
        if not np.array_equal(compact['movie_ids'], movies.id.to_numpy()):
            raise ValueError('Model movie IDs do not match. Rebuild both model files.')
        n = len(movies)
        indices, scores = compact['indices'], compact['scores']
        if indices.shape != (n, min(20, n - 1)) or scores.shape != indices.shape:
            raise ValueError('Invalid compact similarity shape.')
        if (indices < 0).any() or (indices >= n).any() or not np.isfinite(scores).all():
            raise ValueError('Invalid compact similarity values.')
        similarity = np.zeros((n, n), dtype=np.float32)
        similarity[np.arange(n)[:, None], indices] = scores
        # Include the reciprocal edges to preserve matrix symmetry.
        np.maximum(similarity, similarity.T, out=similarity)
        np.fill_diagonal(similarity, compact['diagonal'])
    if not isinstance(movies, pd.DataFrame) or not isinstance(similarity, np.ndarray):
        raise ValueError('Invalid model files. Rebuild with python train.py.')
    if similarity.shape != (len(movies), len(movies)) or movies.id.duplicated().any():
        raise ValueError('Model files do not match. Rebuild with python train.py.')
    return movies.reset_index(drop=True), similarity


def recommend_by_id(movie_id, movies, similarity, n=5):
    if not isinstance(n, int) or not 1 <= n <= 20:
        raise ValueError('Choose between 1 and 20 recommendations.')
    matches = np.flatnonzero(movies.id.to_numpy() == int(movie_id))
    if len(matches) != 1:
        raise ValueError('Please choose a movie from the list.')
    index = int(matches[0])
    scores = similarity[index]
    order = np.argsort(-scores, kind='stable')
    order = [int(i) for i in order if i != index and scores[i] > 0][:n]
    result = movies.iloc[order].copy()
    result['similarity'] = scores[order]
    return result


def recommend(title, movies, similarity, n=5):
    matches = movies.loc[movies.title.str.casefold().eq(str(title).strip().casefold())]
    if matches.empty:
        raise ValueError(f'Movie not found: {title}')
    if len(matches) > 1:
        raise ValueError('Multiple films share this title; use recommend_by_id with a movie ID.')
    return recommend_by_id(int(matches.iloc[0].id), movies, similarity, n)
