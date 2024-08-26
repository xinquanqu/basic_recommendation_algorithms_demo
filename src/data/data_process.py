import math

import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset(object):
    def __init__(self, data_directory: str = "../data/raw/"):
        self.data_directory = data_directory

        self.read_dataset()

        self.interactions_full_df = self.basic_feature_enginerring()

        self.train_test_split()

        self.build_index()

    def read_dataset(self):
        # Load shared articles dataset
        articles_df = pd.read_csv(self.data_directory + "shared_articles.csv")
        self.articles_df = articles_df[articles_df["eventType"] == "CONTENT SHARED"]
        # Load interaction dataset
        self.interactions_df = pd.read_csv(
            self.data_directory + "users_interactions.csv"
        )

    def basic_feature_enginerring(self):
        # feature engineering
        # rule 1: keep users with at least 5 interactions
        users_interactions_count_df = (
            self.interactions_df.groupby(["personId", "contentId"])
            .size()
            .groupby("personId")
            .size()
        )
        users_with_enough_interactions_df = users_interactions_count_df[
            users_interactions_count_df >= 5
        ].reset_index()[["personId"]]
        interactions_from_selected_users_df = self.interactions_df.merge(
            users_with_enough_interactions_df,
            how="right",
            left_on="personId",
            right_on="personId",
        )

        # add eventstrength feature
        event_type_strength = {
            "VIEW": 1.0,
            "LIKE": 2.0,
            "BOOKMARK": 2.5,
            "FOLLOW": 3.0,
            "COMMENT CREATED": 4.0,
        }

        interactions_from_selected_users_df["eventStrength"] = (
            interactions_from_selected_users_df["eventType"].apply(
                lambda x: event_type_strength[x]
            )
        )

        # normalize eventStrenth on unique interactions
        smooth_user_preference = lambda x: math.log(1 + x, 2)

        return (
            interactions_from_selected_users_df.groupby(["personId", "contentId"])[
                "eventStrength"
            ]
            .sum()
            .apply(smooth_user_preference)
            .reset_index()
        )

    def train_test_split(self, test_ratio: float = 0.20, random_state=42) -> None:
        interactions_train_df, interactions_test_df = train_test_split(
            self.interactions_full_df,
            stratify=self.interactions_full_df["personId"],
            test_size=0.20,
            random_state=random_state,
        )
        self.interactions_train_df = interactions_train_df
        self.interactions_test_df = interactions_test_df

    def build_index(self) -> None:
        # Indexing
        # Indexing by personId to speed up the searches during evaluation
        self.interactions_full_indexed_df = self.interactions_full_df.set_index(
            "personId"
        )
        self.interactions_train_indexed_df = self.interactions_train_df.set_index(
            "personId"
        )
        self.interactions_test_indexed_df = self.interactions_test_df.set_index(
            "personId"
        )

    def __repr__(self) -> str:
        def df_info(df: pd.DataFrame, df_name: str) -> str:
            return f"{df_name}: \nshape: {df.shape} \n{df.dtypes}\n"

        return "\n".join(
            [
                df_info(df, df_name)
                for df, df_name in [
                    (self.articles_df, "articles_df"),
                    (self.interactions_df, "interactions_df"),
                    (self.interactions_full_df, "interactions_full_df"),
                    (self.interactions_train_df, "interactions_train_df"),
                    (self.interactions_test_df, "interactions_test_df"),
                    (self.interactions_full_indexed_df, "interactions_full_indexed_df"),
                    (
                        self.interactions_train_indexed_df,
                        "interactions_train_indexed_df",
                    ),
                    (self.interactions_test_indexed_df, "interactions_test_indexed_df"),
                ]
            ]
        )
