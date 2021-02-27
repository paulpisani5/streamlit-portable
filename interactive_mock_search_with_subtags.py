# Streamlit - Interactive search with topic filter

import pandas as pd
import numpy as np

import streamlit as st
from streamlit.report_thread import get_report_ctx
import SessionState
from annotated_text import annotated_text

from sqlalchemy import create_engine
from datetime import datetime 
import os
import pickle

#import json

#from top2vec import Top2Vec
from sklearn.metrics.pairwise import cosine_similarity

from collections import defaultdict


#def extract_photo_url(json_field):
#    dict_field = json.loads(json_field)
#    if 'photo' in dict_field:
#        return dict_field['photo']
#    else:
#        return 'https://docs.microsoft.com/en-us/windows/win32/uxguide/images/mess-error-image4.png'


# def refresh_tables():

#     # engine.execute("""DROP TABLE sessions;""")
#     # engine.execute("""DROP TABLE sessions_2;""")
#     # engine.execute("""DROP TABLE submitted_responses;""")
#     # engine.execute("""DROP TABLE submitted_responses_topic;""")

#     engine.execute("""
#                 CREATE TABLE IF NOT EXISTS sessions (
#                 session_id VARCHAR (255) NOT NULL,
#                 created_on VARCHAR (255) NOT NULL,
#                 username VARCHAR (255) NOT NULL
#                 );
#     """)

#     engine.execute("""
#                 CREATE TABLE IF NOT EXISTS submitted_responses (
#                 session_id VARCHAR (255) NOT NULL,
#                 created_on VARCHAR (255) NOT NULL,
#                 topic_label VARCHAR (255) NOT NULL,
#                 topic_1 varchar (255) NOT NULL,
#                 topic_2 varchar (255) NOT NULL,
#                 topic_3 varchar (255) NOT NULL,
#                 other_topics varchar (255) NOT NULL
#                 );
#     """)

#     engine.execute("""
#                 CREATE TABLE IF NOT EXISTS submitted_responses_topic (
#                 session_id VARCHAR (255) NOT NULL,
#                 created_on VARCHAR (255) NOT NULL,
#                 topic_label VARCHAR (255) NOT NULL,
#                 topic_label_appropriateness VARCHAR (255) NOT NULL,
#                 improved_topic_label VARCHAR (255) NOT NULL,
#                 concerns_opportunities VARCHAR (255) NOT NULL
#                 );
#     """)

#     engine.execute("""
#                 CREATE TABLE IF NOT EXISTS activity_tag_review (
#                 session_id VARCHAR (255) NOT NULL,
#                 created_on VARCHAR (255) NOT NULL,
#                 activity_uid VARCHAR (255) NOT NULL,
#                 topic_0_relevance VARCHAR (255) NOT NULL,
#                 topic_1_relevance VARCHAR (255) NOT NULL,
#                 topic_2_relevance VARCHAR (255) NOT NULL,
#                 topic_3_relevance VARCHAR (255) NOT NULL,
#                 topic_4_relevance VARCHAR (255) NOT NULL
#                 );
#     """)


# def authenticate_user():
    
#     existing_session = engine.execute("""SELECT COUNT(*) FROM sessions WHERE session_id = '%s'""" % session_id).first()[0] > 0
#     if existing_session > 0: 
#         return True 

#     else: 
#         placeholder_1 = st.sidebar.empty()
#         placeholder_2 = st.sidebar.empty()
#         placeholder_3 = st.sidebar.empty()

#         username = placeholder_1.text_input("Enter username: ")
#         password = placeholder_2.text_input("Enter password: ")
#         sign_in = placeholder_3.button("Sign in")

#         if sign_in and ((username == 'paul' and password == 'test') or (username == 'kejia' and password == 'test2') or (username == 'christoph' and password == 'classified')):
#             today = str(datetime.now())
#             engine.execute("""
#                 INSERT INTO sessions(session_id, created_on, username) 
#                 VALUES ('%s', '%s', '%s')
#             """ % (session_id, today, username))

#             placeholder_1.empty()
#             placeholder_2.empty()
#             placeholder_3.empty()

#             return True

#         elif sign_in and ~((username == 'paul' and password == 'test') or (username == 'kejia' and password == 'test2')):
#             st.subheader('Login failed, please retry!')

#         else: 
#             return False


@st.cache
def load_topic_labels():
    df = pd.read_csv(topic_labels_path)
    df['full_label'] = df['topic_id'].astype('str') + ' - ' + df['label_lvl_1']
    return df


@st.cache
def load_topic_stats():
    df = pd.read_pickle(topic_stats_path)
    return df


def write_activities_file_in_chunks(df, chunksize=10000):
    """
    This is used as a preprocessing step to split activities file into chunks for github upload.
    """

    output_filepath = activities_path.split('.pkl')[0]+'_%d.pkl'

    for i in range(0,1+len(df)//chunksize):

        print ('Writing lines %d to %d at %s'%(i*chunksize,(i+1)*chunksize,output_filepath%i))

        # write chunk
        pd.to_pickle(df.iloc[i*chunksize:(i+1)*chunksize],output_filepath%i)


@st.cache(show_spinner=False)
def load_activities_in_chunks(nchunks=None):
    """
    Read activities file that has been split into chunks.
    Note: if we are space constrained, then choose a smaller number of chunks to load.  Each chunk (currently)
    is about 17 mb.
    """
    
    input_filepath = activities_path.split('.pkl')[0]+'_%d.pkl'
    
    chunk_count = 0
    df_list = []
    
    while (nchunks is None or chunk_count < nchunks):
        try:
            df_list.append(pd.read_pickle(input_filepath % chunk_count))
            chunk_count += 1
        except FileNotFoundError:
            break

    return pd.concat(df_list)


@st.cache(show_spinner=False,allow_output_mutation=True)
def load_activities():

    if load_activities_in_chunks_flag:
        return load_activities_in_chunks(nchunks=nchunks)

    activities = pd.read_pickle(activities_path)

    # get mapping from newer run of model to older run if we are using different activities date from model date.
    # reason is in this case we want to use model data from older run which will not correspond to same topic ids as newer activities version, 
    # so we need to map newer ids back to previous ids.
    if (topic_model_date != activities_date):
        topic_id_mapping = activities.query('topic_0_id_global_prev != -1')[['topic_0_id_global','topic_0_id_global_prev']].set_index('topic_0_id_global').drop_duplicates()
        for i in range(5):
            field = 'topic_%d_id_global' % i
            activities.loc[:,field] = topic_id_mapping.loc[activities[field].values]['topic_0_id_global_prev'].values

#     activities['photo_url'] = activities['details'].apply(lambda x: extract_photo_url(x))
    relevant_columns = [
        'uid',
        'title',
        'user_uid',
        'summary',
        'class_experience',
        'published_at',
        'total_enrollment',
        'total_first_time_enrollment',
        'total_bookings',
        'age_min',
        'age_max',
        'photo_url',
        'tokens',
        'topic_0_id_global',
        'topic_0_score_global',
        'topic_1_id_global',
        'topic_1_score_global',
        'topic_2_id_global',
        'topic_2_score_global',
        'topic_3_id_global',
        'topic_3_score_global',
        'topic_4_id_global',
        'topic_4_score_global'
    ]
    activities['total_enrollment'] = activities['total_enrollment'].replace(np.nan,0)
    activities['total_first_time_enrollment'] = activities['total_first_time_enrollment'].replace(np.nan,0)
    activities = activities[relevant_columns].copy()

    return activities


@st.cache(allow_output_mutation=True)
def load_model():

    # Basic model load
    model = Top2Vec.load(model_path)
    ids = np.load(ids_path)

    return model, ids


def search_vectors_by_vector(vectors, vector, num_res):
    ranks = [res[0] for res in
             cosine_similarity(vectors, vector.reshape(1, -1))]
    indexes = np.flip(np.argsort(ranks)[-num_res:])
    scores = np.array([round(ranks[res], 4) for res in indexes])

    return indexes, scores


def find_closest_topics_to_vector (vector, max_topics=10, score_threshold=0.2):
    """
    Input: vector
    Output: list of closest topics to this vector.  Use this to get closest topics to given input topic vector.
    """

    topic_ids, topic_scores = search_vectors_by_vector(topic_vectors,vector,max_topics)
        
    # return the first topic, plus any additional with score > threshold
    return [(t,s) for (t,s) in zip(topic_ids,topic_scores) if s>score_threshold or t==topic_ids[0]]


def get_activities_from_topic(topic_id, start=0, end=None):


    activities = load_activities()

    rank_1_match = activities[(activities['topic_0_id_global'] == topic_id)].copy() 
    rank_1_match['match_rank'] = 0
    rank_1_match['match_score'] = rank_1_match['topic_0_score_global']

    rank_2_match = activities[(activities['topic_1_id_global'] == topic_id)].copy() 
    rank_2_match['match_rank'] = 1
    rank_2_match['match_score'] = rank_2_match['topic_1_score_global']

    rank_3_match = activities[(activities['topic_2_id_global'] == topic_id)].copy() 
    rank_3_match['match_rank'] = 2
    rank_3_match['match_score'] = rank_3_match['topic_2_score_global']

    rank_4_match = activities[(activities['topic_3_id_global'] == topic_id)].copy() 
    rank_4_match['match_rank'] = 3
    rank_4_match['match_score'] = rank_4_match['topic_3_score_global']

    rank_5_match = activities[(activities['topic_4_id_global'] == topic_id)].copy() 
    rank_5_match['match_rank'] = 4
    rank_5_match['match_score'] = rank_5_match['topic_4_score_global']

    top_matches = pd.concat([
        rank_1_match,
        rank_2_match,
        rank_3_match,
        rank_4_match,
        rank_5_match
    ])

    top_matches.sort_values(by='match_score', ascending=False, inplace=True)

    if start >= len(top_matches):
        return []

    if end is not None:
        return top_matches[start:end]

    else:
        return top_matches[start:]


def get_activities_from_topic_list(topic_id_list, start=0, end=None):
    """
    Returns all activites that match ALL of the topics in topic_id_list
    """

    df_list = []

    for topic_id in topic_id_list:

        df_list.append(get_activities_from_topic(topic_id))

    # get top match score result for each uid (if we get same activity from multiple topics)
    # Note: the 'filter' call retains only activities that show up in each topic in topic list.  If we were to remove that it would be topic OR rather than AND.
    activities =  pd.concat(df_list).sort_values('match_score',ascending=False).groupby('uid').filter(lambda x: len(x) == len(topic_id_list))
    activities = activities.groupby('uid').first().reset_index().sort_values('match_score',ascending=False)

    if end is not None:
        return activities[start:end]

    else:
        return activities[start:]


@st.cache
def load_image(activity_url, photo_url):
    html = "<a href='{url}'><img src='{src}' width='250'></a>".format(url=activity_url, src=photo_url)
    return html


def get_color_from_score(score):
    """
    Input: similarity score
    Output: color string
    """

    tag_colors = ['#b3a6d5','#b7a3cb','#bb9ec0','#c09ab5','#c496a9','#c8919e','#cc8d94','#d18989','#d5857d','#d98072','#dd816c','#e18c71','#e49675','#e8a17a','#ebab7f','#efb684','#f2c189','#f6cc8e','#f9d692','#fde097','#f9e399','#f1e29b','#e9e09d','#e2df9e','#dbdda0','#d3dca1','#ccdaa3','#c4d9a4','#bcd7a6','#b5d6a7']

    max_score = 0.5
    min_score = 0.1
    step_size = (max_score-min_score)*1./len(tag_colors)

    if score <= min_score:
        return tag_colors[0]

    if score >= max_score:
        return tag_colors[-1]

    else:
        return tag_colors[int(np.floor((score-min_score)*1./step_size))]


def interactive_search_mock_with_subtags():
    """
    Display a basic search function.  Search function will include:
        - Choice of model
        - Topic multiselect (AND)
        - Topic subtag multiselect (AND)
    Results will display:
        - Class image, name, summary, tagged topics + scores, any subtags
    """

    st.header('Find Classes')

    # search bar
    text_query = st.text_input('Search query',value=session_state.text_query)

    # topic multiselect
    topic_labels = load_topic_labels()

    topic_filter, subtag_filter = st.beta_columns([1,1])

    selected_topics = topic_filter.multiselect('Filter by topic',topic_labels.full_label)


    with open(subtags_path, 'rb') as handle:
        drawing_subtags = pickle.load(handle)

    for uid in drawing_subtags:
        tags = drawing_subtags[uid]
        drawing_subtags[uid] = ['Drawing - ' + t for t in tags]

    # collect possible subtags
    subtags = set()
    for uid,tags in drawing_subtags.items():
        for t in tags:
            subtags.add(t)
 
    selected_subtags = set(subtag_filter.multiselect('Filter by subtags',list(subtags)))

    sort_by, sort_order, subsample, num_results, extra_space = st.beta_columns([1,1,1,1,2])

    selected_sort_by = sort_by.selectbox('Sort by',['Similarity Score','Enrollments','Bookings','Random'])

    selected_sort_order = sort_order.selectbox('Sort order',['Descending','Ascending'])

    selected_subsample = subsample.selectbox('Subsample results before sorting?',['Yes','No'])

    selected_num_results = num_results.number_input('Number of results to display?',min_value=1,max_value=200,value=20)


    st.write('')

    if st.button('Search'):

        # get activities that match topics
        if len(selected_topics) > 0:
            activities = get_activities_from_topic_list(topic_labels.set_index('full_label').loc[selected_topics].topic_id,start=0,end=None)


            with st.beta_expander('Show similar topics:'):

                # print similar topics
                for topic in selected_topics:

                    tid = topic_labels.set_index('full_label').loc[topic].topic_id

                    st.markdown("Topics similar to **%s**:" % topic_labels.loc[tid].label_lvl_1)
                    similar_topics = find_closest_topics_to_vector(topic_vectors[tid],max_topics=20)

                    tags = []

                    for (tid, score) in similar_topics:
                        label_lvl_1 = topic_labels.loc[tid].label_lvl_1
                        tags.append((str(label_lvl_1) + (' (%0.2lf)' % score),'TAG',get_color_from_score(score)))
                        tags.append(' ')

                    annotated_text(*tags)

            st.write('')


        else:
            activities = load_activities()

        if len(selected_subtags)>0:

            activities = activities[activities.uid.isin(drawing_subtags.keys())]

            # get activities that match subtags
            activities = activities[activities.apply(lambda x: selected_subtags.issubset(set(drawing_subtags[x.uid])),axis=1)]

        # filter to activities that match search query
        if len(text_query)>0:

            activities = activities[activities.apply(lambda x: text_query.lower() in (str(x.title) + ' ' + str(x.summary)).lower(),axis=1)]

        # filter down to random selected_num_results classes if we have more than that
        if selected_subsample == 'Yes':
            if len(activities)>selected_num_results:
                activities = activities.sample(selected_num_results)


        # true if 'Ascending' selected
        sort_ascending = (selected_sort_order == 'Ascending')

        # handle sort order 
        if selected_sort_by == 'Similarity Score':

            # breaking out here because this is only relevant if match_score is present
            if len(selected_topics)>0:
                activities = activities.sort_values('match_score',ascending=sort_ascending)

            else:
                activities = activities.sort_values('topic_0_score_global',ascending=sort_ascending)

        elif selected_sort_by == 'Enrollments':
            activities = activities.sort_values('total_enrollment',ascending=sort_ascending)

        elif selected_sort_by == 'Bookings':
            activities = activities.sort_values('total_bookings',ascending=sort_ascending)

        elif selected_sort_by == 'Random':
            activities = activities.sample(frac=1)

        # truncate to selected_num_results after sorting if not using subsample
        activities = activities.iloc[:selected_num_results]
            
        # class listings
        with st.beta_container():

            if len(activities) == 0:

                st.write('No classes found matching criteria.')

            for i in range(len(activities)):
                
                photo_col, text_col = st.beta_columns([1,4])

                row = activities.iloc[i]

                activity_url = 'http://www.outschool.com/classes/' + row.uid
                photo_url = row.photo_url

                with photo_col:
                    st.markdown(load_image(activity_url, photo_url), unsafe_allow_html=True)

                with text_col:

                    st.markdown('### %s'%row.title)
                    st.markdown('*%s*'%row.summary)

                    with st.beta_expander("Show Class Experience", expanded=False):
                        st.write(row.class_experience)

                    tags = []

                    for i in range(5):
                        
                        score = row['topic_%d_score_global'%i]
                        label_lvl_1 = topic_labels.loc[row['topic_%d_id_global'%i]].label_lvl_1 
                        tags.append((str(label_lvl_1) + (' (%0.2lf)' % score),'TAG',get_color_from_score(score)))
                        tags.append(' ')

                    #if len(selected_subtags)>0:

                    for tag in drawing_subtags[row.uid]:
                        tags.append((tag,'SUBTAG','#cce6ff'))
                        tags.append(' ')

                    annotated_text(*tags)


st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 2000px;
        padding-left: 10rem;
        padding-right: 10rem;
    }}
</style>
""",
        unsafe_allow_html=True,
    )


# path definitions and flags
# ------------------------------------

hosted = True # for hosted version, we won't load any large files (full activites / models etc)

if hosted:
    load_model_flag = False
    load_activities_in_chunks_flag = True
    nchunks = None

else:
    load_model_flag = True
    load_activities_in_chunks_flag = True
    nchunks = None

# path to activities file
activities_date = '02-23-21'
#activities_path = 'activities-%s-with-topic-labels.pkl' % activities_date
activities_path = 'activities.pkl'

# path to existing model 
topic_model_date = '11-24-20'
model_subject = 'all' # this is the global model
model_prefix = 'top2vec_activities_all_class_exp_all_deep_doc2vec_skipped_updated_%s'
model_path = (model_prefix % topic_model_date) + '.model'

topic_stats_path = 'topics_global_%s.pkl' % topic_model_date
topic_labels_path = 'topic_labels_w_description.csv'

# path to topic vectors (if split from model)
topic_vectors_path = (model_prefix % topic_model_date) + '_topic_vectors.npy'

# path to existing ids
ids_path = (model_prefix % topic_model_date) + '_ids.npy'

# sub tag assignments
subtags_path = 'activities-11-24-20-with-topic-labels_subtags.pkl'

# end path definitions and flags
# ------------------------------------

# model load
if load_model_flag:
    model, ids = load_model()

    topic_vectors = model.topic_vectors

    # save topic vectors if desired
    np.save(topic_vectors_path, model.topic_vectors)

else:
    topic_vectors = np.load(topic_vectors_path)


# # connect to local db
# engine = create_engine('postgresql://christoph:classified@localhost:5432/streamlit_topics')

# # st.set_page_config(page_title='Topic Explorer', page_icon=None, layout='wide', initial_sidebar_state='auto')

# session_id = get_report_ctx().session_id
# session_id = session_id.replace('-','_')

# initialize state variables
session_state = SessionState.get(text_query='')

# force reset state variables if desired
def reset_session_state():
    session_state.text_query=''

if st.sidebar.button('Start Over'):
    reset_session_state()

action = st.sidebar.selectbox('Select Module:',['Option 1'])

if action == 'Option 1':
    interactive_search_mock_with_subtags()

