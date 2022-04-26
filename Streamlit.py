import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
# import SessionState

# sidebar
with st.sidebar.expander("How it works?", expanded=True):
    st.markdown("## How it works? :thought_balloon:")
    st.write(
        "For an overview of the ML methods used and how I created this app, the dataset and some helpful references are listed below."
    )
    blog1 = "https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv"
    blog2 = "https://towardsdatascience.com/building-a-recipe-recommendation-api-using-scikit-learn-nltk-docker-flask-and-heroku-bfc6c4bdd2d4"
    blog3 = "https://medium.com/analytics-vidhya/how-to-build-personalized-recommendation-from-scratch-recipes-from-food-com-c7da4507f98"
    st.markdown(
        f"1. [Food.com Recipes and Interactions]({blog1})"
    )
    st.markdown(
        f"2. [Building a Recipe Recommendation API using Scikit-Learn, NLTK, Docker, Flask, and Heroku]({blog2})"
    )
    st.markdown(
        f"3. [How to build personalized recommendation from scratch: recipes from Food.com]({blog3})"
    )


image = Image.open("header_img.jpeg").resize((800, 200))
st.image(image)

st.markdown("# :cooking: What's For Dinner? ")

st.markdown(
    "A Recipe Recommendation App Presented by Siyu Mao (Inspired by Jack Leitch) <a href='https://github.com/maomaolikesmatcha/Food-recipe-recommendation-system' > <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/600px-Octicons-mark-github.svg.png' width='20' height='20' > </a> ",
    unsafe_allow_html=True,
)
st.markdown(
    "### Given ingredients in my fridge, what different recipes can I can make? :sushi: or :pizza: or :spaghetti:?"
)
st.markdown(
    "Cooking is a hobby for some and a major problem for others. However, you can always use a helping hand to make it easy. My recommendation engine will look through 10,000+ recipes to find matches for you :mag: (っ◔◡◔)っ ❤  Try it out for yourself!"
)

st.text("") 


## dataset + def()
df = pd.read_csv('final_df1.csv', index_col=0)

# df[['calories','total fat (PDV)','sugar (PDV)','sugar (PDV)','protein (PDV)','saturated fat (PDV)','carbohydrates (PDV)']] = df.nutrition.str.split(",",expand=True) 
# df['calories'] = df['calories'].apply(lambda x: x.replace('[', ''))
# df['carbohydrates (PDV)'] = df['carbohydrates (PDV)'].apply(lambda x: x.replace(']', ''))
# df = df.drop(columns=['id', 'submitted', 'description', 'nutrition', 'contributor_id', 'n_steps', 'n_ingredients'])

# df = df.dropna()
# df = df[:10000]

# from nltk.stem import LancasterStemmer
# ls = LancasterStemmer()
# def treat_ingredients(input):
#     output = []
#     for ingredient in input:
#         ingredient_list = ingredient.split(' ')
#         output.append(" ".join(ingredient_list))
#     return "".join(output)

# df['ingredients'] = df['ingredients'].apply(lambda x: [ls.stem(w) for w in x])
# df['ingredients'] = df['ingredients'].apply(lambda x : treat_ingredients(x))

# #prettify title 
# df['name'] = df['name'].apply(lambda x : x.replace('  ', ' '))
# # df['name'] = df['name'].apply(lambda x : x.replace(r'^\sS$', '\'s'))
# df['name'] = df['name'].str.title()

# word vectors
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english', max_df = 0.7,min_df = 2)
ing_tfidf = tfidf.fit_transform(df['ingredients'])

# similarity 
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(ing_tfidf, ing_tfidf)


# initial rec
def get_recommendations(input):
    # create embessing for input text
    input = ingredients
    # create tokens with elements
    input = input.split(",")
    if len(input) == 1:
        results = df.loc[df.ingredients.str.contains(r'(?=.*{})'.format(input[0])) == True]
    elif len(input) == 2:
        results = df.loc[df.ingredients.str.contains(r'(?=.*{})(?=.*{})'.format(input[0], input[1])) == True]
    elif len(input) == 3:
        results = df.loc[df.ingredients.str.contains(r'(?=.*{})(?=.*{})(?=.*{})'.format(input[0], input[1], input[2])) == True]
    else:
        results = df.loc[df.ingredients.str.contains(r'(?=.*{})(?=.*{})(?=.*{})(?=.*{})'.format(input[0], input[1], input[2], input[3])) == True]
    return results.drop(columns=['tags']).head()

# more recipes
def more_recipes_with_similar_ingredients(input):
    results = get_recommendations(input)
    recipe_name = results.name.values[0]
    # get the initial recipe's index
    initial_recipe = results.loc[results['name'] == recipe_name]
    idx = initial_recipe.index[0] - 1

    # Get the pairwsie similarity scores of all ing with ing
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the ing based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of top 5 most similar ing
    sim_scores = sim_scores[1:6]

    # Get the ingredient indices for recipes
    ing_indices = [i[0] for i in sim_scores]

    # get dataframe 
    result = df.iloc[ing_indices].drop(columns='tags')
    # result = result.sort_values('calories', ascending=True)

    # print(f'The choosen recipe is {recipe_name}\n')
    # print('Recommended recipes are: ', *result.name.values, sep='\n')
    return result.head()

# randomized options if none of the recipes are interesting
def surprise_me(category):
    random = df.loc[df.tags.str.contains(category) == True].sample(3)
    return random.drop(columns=['tags'])


# streamlit main page  ----------------------------------------------------------------------------
ingredients = st.text_input(
    "Enter one to four ingredients you would like to cook with (seperate ingredients with a comma)",
    "beef, tomatoes, cheese, onion",)
result = st.button('Give me recommendations')
gif_runner1 = st.image("waiting1.gif")

if result:
    st.success("Successful! Your DELICIOUS RECIPES are here!")
    st.write(get_recommendations(ingredients).style.format(precision=0))
    gif_runner1.empty()


### more similar recipes ---------------------------------------------------------------------------
st.markdown('#### :sparkles: More similar recipes?')
st.write('Based on similar ingredients')
more = st.button('Give me more recipes')
gif_runner2 = st.image("waiting2.gif")

if more:
    st.success("Successful! Some NEW RECIPES await you!")
    st.write(more_recipes_with_similar_ingredients(ingredients).style.format(precision=0))
    gif_runner2.empty()

### Randomized recipes (done)  -------------------------------------------------------------------------
st.markdown('#### :sparkles: Feeling Lucky?')
category = st.selectbox(
     'Which kind of cuisine do you like?',
     ('asian', 'north-american', 'desserts', 'mexican', 'vegetarian', 'seasonal', 'easy', 'indian', 'italian', 'french', 'healthy', 'appetizers', 'snacks'))
st.write('You selected:', category)

random = st.button('Give me 3 random recipes')
gif_runner3 = st.image("waiting3.gif")
if random:
    st.success("Successful! Here are your 3 randam AWESOME RECIPES!")
    st.write(surprise_me(category).style.format(precision=0))
    gif_runner3.empty()


# option = st.selectbox(
#      'How would you like to cook?',
#      ('Give me recommendations!', 'Feeling Luck? (3 random recipes)'))

# st.write('You selected:', option) 




# session_state.execute_recsys = st.button("Give me recommendations!")
# session_state.execute_recsys = st.button("Feeling Luck? (3 random recipes)")

# if session_state.execute_recsys:
#     col1, col2, col3 = st.beta_columns([1, 6, 1])
#     with col2:
#         gif_runner = st.image("waiting.gif")
#     # recipe = rec_sys.RecSys(ingredients)
#     recipe = get_recs(ingredients)
#     gif_runner.empty()
#     session_state.recipe_df_clean = recipe.copy()

#     recipe_display = recipe[["name", "minutes", "steps", "ingredients", 
#     "calories", "total fat (PDV)", "sugar (PDV)", "sodium (PDV)", 
#     "protein (PDV)", "saturated fat (PDV)", "carbohydrates (PDV)"]]
#     session_state.recipe_display = recipe_display.to_html(escape=False)
#     session_state.recipes = recipe.recipe.values.tolist()
#     session_state.model_computed = True
#     session_state.execute_recsys = False

 

