import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class RecipeRecommender:
    def __init__(self, file_path):
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Load the dataset
        self.df = pd.read_csv(file_path)

        # Clean and prepare data
        self.prepare_data()

        # Create TF-IDF vectorization for ingredients
        self.create_ingredient_vectorizer()

    def prepare_data(self):
        # Ensure the correct column is used for ingredients
        try:
            # Clean and process ingredients
            # drops empty rows
            self.df = self.df.dropna(subset=[self.df.columns[10]])
            #drop duplicates (ingredients yg sama persis)
            self.df = self.df.drop_duplicates(subset='ingredients')
            
            # Filter by ingredient count (ensure at least 3 ingredients)
            # ngedrop recipe yg ingredientsnya kurang dari 3
            self.df = self.df[self.df['ingredients'].str.count(',') >= 2]

            # Convert ingredients to list of strings
            self.df['ingredient_list'] = self.df.iloc[:, 10].apply(
                lambda x: ' '.join(eval(x) if isinstance(x, str) else x)
            )
            
        except Exception as e:
            st.error(f"Error in data preparation: {e}")
            raise

    def create_ingredient_vectorizer(self):
        # Create TF-IDF vectorizer for ingredients, memberi value ke TIAP ingredients semakin rare ingredients semakin tinggi valuene, dan bakal lebih diutamain di top recommendation
        self.vectorizer = TfidfVectorizer()
        #creates tthe vector for the 'already assigned a value' ingredients
        self.ingredient_vectors = self.vectorizer.fit_transform(
            self.df['ingredient_list']
        )

    def recommend_recipes(self, input_ingredients, top_n):
        # Validate input
        if not input_ingredients:
            return []

        # Convert input ingredients to a single string
        input_ingredient_str = ' '.join(input_ingredients)

        # Vectorize input ingredients
        #input jd vector
        input_vector = self.vectorizer.transform([input_ingredient_str])

        # Calculate cosine similarity, input sm recipe dihitung cos nya, cos 0-1, 0 being not similar 1 being exactly similar
        similarities = cosine_similarity(input_vector, self.ingredient_vectors)[0]

        # Get top N similar recipes
        top_indices = similarities.argsort()[-top_n:][::-1]

        # Prepare recommendation results with similarity scores
        recommendations = []
        for idx in top_indices:
            try:
                recommendations.append({
                    'recipe_name': self.df.iloc[idx, 0],  # Recipe name
                    'ingredients': eval(self.df.iloc[idx, 10]) if isinstance(self.df.iloc[idx, 10], str) else self.df.iloc[idx, 10],
                    'similarity_score': similarities[idx],
                    'nutrition_values': self.df.iloc[idx, 6],
                    'steps': self.df.iloc[idx, 8]
                })
            except Exception as e:
                st.warning(f"Could not process recipe: {e}")

        return recommendations

    def ingredient_overlap(self, input_ingredients, recipe_ingredients):
        # Calculate the percentage of input ingredients present in recipe
        input_set = set(input_ingredients)
        recipe_set = set(recipe_ingredients)

        overlap = len(input_set.intersection(recipe_set))
        return (overlap / len(input_set)) * 100 if input_set else 0

def main():
    
    # Set page title and layout
    st.set_page_config(page_title="Recipe Recommender", page_icon="🍽️", layout="wide")
    
    st.markdown(
    """
    <style>
   
    div[data-testid="stToolbar"] {
            visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

    # Create loading placeholder
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown("<h1 style='text-align: center; color: grey;'>🍽️</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: grey;'>Loading Delicious Recipes...</h3>", unsafe_allow_html=True)
    
     # Attempt to load recommender with error handling
    try:
        # Use relative path or full path to your CSV
        recommender = RecipeRecommender('RAW_recipes.csv')
    except FileNotFoundError:
        loading_placeholder.error("Recipe dataset not found. Please check the file path.")
        st.stop()    

    # Main title
    st.title("🍳 Smart Recipe Recommender")

    # Sidebar for instructions
    st.sidebar.title("How to Use")
    st.sidebar.info(
        "1. Enter ingredients you want to use\n"
        "2. Separate ingredients with commas\n"
        "3. Press Enter or click 'Recommend Recipes'\n"
        "Example: chicken, onion, garlic"
    )

    # Ingredients input with key to track changes
    ingredients_input = st.text_input(
        "Enter your ingredients (comma-separated)", 
        placeholder="e.g. chicken, rice, bell pepper",
        key="ingredients_key"  # key to track input changes
    )

    # Number of recommendations slider
    num_recommendations = st.slider(
        "Number of Recommendations", 
        min_value=1, 
        max_value=10, 
        value=5
    )

    # Flag to trigger recommendations
    recommend_clicked = st.button("Recommend Recipes")

    # Only clear loading placeholder after successful recommender creation
    loading_placeholder.empty()

    # Check if recommendations should be shown
    if recommend_clicked or st.session_state.get('ingredients_key', '') != '':
        if ingredients_input:
            try:
                # Process ingredients
                ingredients = [ing.strip() for ing in ingredients_input.split(',')]

                # Get recommendations
                recommendations = recommender.recommend_recipes(ingredients, top_n=num_recommendations)

                loading_placeholder.empty()

                # Display recommendations
                #st.empty();
                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"Recipe {i}: {rec['recipe_name']}"):
                        # Similarity and Overlap
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Similarity Score", f"{rec['similarity_score']:.2f}")
                        with col2:
                            overlap = recommender.ingredient_overlap(ingredients, rec['ingredients'])
                            st.metric("Ingredient Overlap", f"{overlap:.2f}%")

                        # Ingredients
                        st.subheader("Ingredients")
                        # Safely handle ingredients list
                        ing_list = rec['ingredients']
                        for ing in ing_list:
                            st.write(f"- {ing}")

                        # Instructions
                        st.subheader("Instructions")
                        # Safely handle steps list
                        steps_list = eval(rec['steps'])
                        for j, step in enumerate(steps_list, 1):
                            st.write(f"{j}. {step}")

                        # Nutritional Values
                        st.subheader("Nutritional Values (Average = 2000 Calories Daily Value)")
                        # Safely handle nutrition values
                        nutrition_values = eval(rec['nutrition_values']) 
                        nutrition_mapping = [
                            ("Calories", nutrition_values[0], ""),
                            ("Sugar", (nutrition_values[2]*50)/100, "g"),
                            ("Sodium", (nutrition_values[3] * 2300) / 100, "mg"),
                            ("Protein", (nutrition_values[4] * 50) / 100, "g"),
                            ("Saturated Fat", (nutrition_values[5] * 20) / 100, "g"),
                            ("Carbohydrates", (nutrition_values[6] * 275) / 100, "g")
                        ]

                        for name, value, unit in nutrition_mapping:
                            st.metric(name, f"{value:.2f} {unit}")

            except Exception as e:
                loading_placeholder.empty()
                st.error(f"An error occurred: {e}")
    else:
        loading_placeholder.empty()
        st.warning("Please enter your ingredients.")

if __name__ == "__main__":

    main()
