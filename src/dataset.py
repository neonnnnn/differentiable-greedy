import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class MovieLensDataset(Dataset):
    """Dataset class for MovieLens 100K dataset。"""

    DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    DATA_COLS = ["user_id", "movie_id", "rating", "timestamp"]
    MOVIE_COLS = [
        "movie_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
    ] + [
        "genre_unknown",
        "Action",
        "Adventure",
        "Animation",
        "Childrens",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]
    USER_COLS = ["user_id", "age", "gender", "occupation", "zip_code"]

    def __init__(
        self,
        data_dir,
        zip_path,
        n_instances=100,
        n_movies=100,
        n_users=500,
        random_state=42,
    ):
        self.data_dir = data_dir
        self.zip_path = zip_path
        self.extracted_path = os.path.join(data_dir, "ml-100k")
        self.random_state = np.random.RandomState(random_state)
        self.n_instances = n_instances
        self.n_movies = n_movies
        self.n_users = n_users

        self.features, self.theta = self._create_instances()

    def __len__(self):
        return self.n_instances

    def __getitem__(self, idx):
        return self.features[idx], self.theta[idx]

    def _download_and_extract(self):
        """Download and extract the dataset if not exists。"""
        if not os.path.exists(self.extracted_path):
            os.makedirs(self.data_dir, exist_ok=True)
            urllib.request.urlretrieve(MovieLensDataset.DATASET_URL, self.zip_path)
            with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
                zip_ref.extractall(self.data_dir)
        else:
            print("Dataset already exists.")

    def _load_dataframes(self):
        """load movielens data as a pandas' dataframe."""
        ratings = pd.read_csv(
            os.path.join(self.extracted_path, "u.data"),
            sep="\t",
            names=MovieLensDataset.DATA_COLS,
        )

        users = pd.read_csv(
            os.path.join(self.extracted_path, "u.user"),
            sep="|",
            names=MovieLensDataset.USER_COLS,
        )

        movies = pd.read_csv(
            os.path.join(self.extracted_path, "u.item"),
            sep="|",
            names=MovieLensDataset.MOVIE_COLS,
            encoding="latin-1",
        )
        return ratings, users, movies

    def _preprocess_features(self, users, movies):
        """Preprocess user and movie features."""
        users["age"] = users["age"] / users["age"].max()
        user_features = pd.get_dummies(
            users,
            columns=["occupation", "gender"],
            prefix=["occ", "gender"],
        )
        user_features = user_features.drop(columns=["user_id", "zip_code"])
        movie_features = movies.drop(  # using only genres
            columns=[
                "movie_id",
                "title",
                "release_date",
                "video_release_date",
                "imdb_url",
            ]
        )
        user_features.set_index(users["user_id"], inplace=True)
        movie_features.set_index(movies["movie_id"], inplace=True)
        return user_features, movie_features

    def _create_instances(self):
        """Create instances (pairs of V and T)."""
        self._download_and_extract()
        ratings, users, movies = self._load_dataframes()
        user_features, movie_features = self._preprocess_features(users, movies)

        all_movie_ids = movies["movie_id"].unique()
        all_user_ids = users["user_id"].unique()
        n_movies = len(all_movie_ids)
        n_users = len(all_user_ids)
        n_ratings = len(ratings)
        print(f"Num Movies: {n_movies}, Num Users: {n_users}, Num Ratings:{n_ratings}")
        # First, we sampled users and movies
        all_sampled_movie_ids = []
        all_sampled_user_ids = []
        for _ in range(self.n_instances):
            # randomly sampling n_movies movies and n_users users
            sampled_movie_ids = self.random_state.choice(
                all_movie_ids, self.n_movies, replace=False
            )
            sampled_user_ids = self.random_state.choice(
                all_user_ids, self.n_users, replace=False
            )
            all_sampled_movie_ids.append(sampled_movie_ids)
            all_sampled_user_ids.append(sampled_user_ids)

        # Then, create features and link probabilities
        features = []
        theta = []
        for i in tqdm(range(self.n_instances)):
            sampled_movie_ids = all_sampled_movie_ids[i]
            sampled_user_ids = all_sampled_user_ids[i]
            # extract ratings
            sub_ratings = ratings[
                ratings["movie_id"].isin(sampled_movie_ids)
                & ratings["user_id"].isin(sampled_user_ids)
            ]
            sub_movie_feats = movie_features.loc[sampled_movie_ids]
            sub_user_feats = user_features.loc[sampled_user_ids]

            # create link probability matrix based on ratings
            rating_map = {1: 0.02, 2: 0.04, 3: 0.06, 4: 0.08, 5: 0.1}
            # create movie-user rating matrix
            pivoted_ratings = sub_ratings.pivot(
                index="movie_id", columns="user_id", values="rating"
            )
            pivoted_ratings = pivoted_ratings.reindex(
                index=sampled_movie_ids, columns=sampled_user_ids
            )
            # determine link probability based on ratings
            theta_true = pivoted_ratings.map(lambda x: rating_map.get(x, 0.0))
            # create feature tensor
            # movie feature matrix: shape: (self.n_movies, 19)
            movie_feats_tensor = torch.tensor(
                sub_movie_feats.values.astype(np.float32), dtype=torch.float32
            )
            # user feature matrix: shape: (self.n_users, 24=1+2+21)
            user_feats_tensor = torch.tensor(
                sub_user_feats.values.astype(np.float32), dtype=torch.float32
            )
            # feature tensor: (n_movies, n_users, 43)
            feature_tensor = torch.cat(
                [
                    movie_feats_tensor.unsqueeze(1).expand(-1, self.n_users, -1),
                    user_feats_tensor.unsqueeze(0).expand(self.n_movies, -1, -1),
                ],
                dim=2,
            )
            theta_true = torch.tensor(theta_true.values, dtype=torch.float32)
            features.append(feature_tensor)
            theta.append(theta_true)

        return torch.stack(features), torch.stack(theta)
