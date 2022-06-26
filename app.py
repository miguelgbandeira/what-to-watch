import flask
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

app = flask.Flask(__name__, template_folder='templates')

data = pd.read_csv("CleanDataset.csv")
new_df = data[["tagline","id","original_title","cast","director","genres","keywords"]]


def rec_features(row):
    return row['keywords'] +" "+row['cast']+" "+row['genres']+" "+row['director']+" "+row['tagline']

new_df["rec_features"] = new_df.apply(rec_features,axis=1)

indices = pd.Series(new_df.index, index=new_df['original_title'])

def title_from_index(index):
    return new_df[new_df.index == index]["original_title"].values[0]

def index_from_title(title):
    title_list = new_df['original_title'].tolist()
    common = difflib.get_close_matches(title, title_list, 1)
    if len(common) == 0:
        return None
    else:
        titlesim = common[0]
        return new_df.loc[new_df['original_title'] == titlesim].index[0]

cv = CountVectorizer()
count_matrix = cv.fit_transform(new_df["rec_features"])

def get_recommendations(title):
    movie_index = index_from_title(title)
    cosine_sim = cosine_similarity(count_matrix)
    sim_scores = list(enumerate(cosine_sim[movie_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    return sim_scores

def list_to_df(list):
    data = []
    col = ['Title']
    for row in list:
        data.append(title_from_index(row[0]))
    df = pd.DataFrame(data, columns=col)
    return df

# Set up the main route
@app.route('/', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
            
    if flask.request.method == 'POST':
        m_name = flask.request.form['movie_name']
        m_name = m_name.title()
        res = get_recommendations(m_name)
        if len(res) == 0:
            return(flask.render_template('negative.html',name=m_name))
        else:
            result_final = list_to_df(res)
            names = []
            for i in range(len(result_final)):
                names.append(result_final.iloc[i][0])

        return flask.render_template('positive.html',movie_names=names,search_name=m_name)

if __name__ == '__main__':
    app.run()