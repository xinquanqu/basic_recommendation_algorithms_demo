from data.data_process import Dataset
from evaluation.model_evaluator import ModelEvaluator
from models.content_based_recommender import ContentBasedRecommender
from models.popularity_recommender import PopularityRecommender
from models.hybrid_recommender import HybridRecommender
from models.collaborative_filtering_recommender import CFRecommender



def main():
    # Load the dataset
    dataset = Dataset()

    # Content-Based Model
    content_based_recommender_model = ContentBasedRecommender(dataset)

    # Popularity Model 
    popularity_recommender_model = PopularityRecommender(dataset)

    # Collaborative Filtering Model
    collaborative_filtering_recommender_model = CFRecommender(dataset)

    # Hybrid Model
    hybrid_recommender_model = HybridRecommender(content_based_recommender_model, collaborative_filtering_recommender_model, cb_ensemble_weight=1.0, cf_ensemble_weight=100.0)

    # Evaluator
    evaluator = ModelEvaluator(dataset)

    # Evaluate the models
    models = [content_based_recommender_model, popularity_recommender_model, collaborative_filtering_recommender_model, hybrid_recommender_model]

    evaluator.evaluate_models_report(models)

if __name__ == '__main__':
    main()