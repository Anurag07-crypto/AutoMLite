import time
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split,RandomizedSearchCV
# -------models-------------------
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.svm import SVC,SVR
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
# --------------------------------
from sklearn.metrics import classification_report,accuracy_score,mean_absolute_error,mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# ------------------------------

# For Decision Tree Classifier
dt_param_grid = {
    "criterion": ["gini", "entropy", "log_loss"],  # measure of quality of split
    "max_depth": [None, 5, 10, 15, 20, 25],        # maximum depth of the tree
    "min_samples_split": [2, 5, 10, 20],           # min samples to split a node
    "min_samples_leaf": [1, 2, 4, 8],              # min samples in a leaf node
    "max_features": [None, "sqrt", "log2"],        # features to consider for split
    "splitter": ["best", "random"]                 # strategy used to choose split
}

# For Decision Tree Regressor
dt_reg_param_grid = {
    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
    "max_depth": [None, 5, 10, 15, 20, 25],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": [None, "sqrt", "log2"]
}

# For KNeighborsClassifier
knn_param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11, 15],         # number of neighbors
    "weights": ["uniform", "distance"],          # uniform = equal, distance = closer points weigh more
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],  # nearest neighbor search method
    "p": [1, 2]                                  # 1 = Manhattan, 2 = Euclidean
}

# For KNeighborsRegressor
knn_reg_param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11, 15],
    "weights": ["uniform", "distance"],
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    "p": [1, 2]
}

# For SVM Classifier
svm_param_grid = {
    "C": [0.1, 1, 10, 100],               # regularization strength
    "kernel": ["linear", "poly", "rbf", "sigmoid"],  # kernel type
    "degree": [2, 3, 4],                  # only for 'poly' kernel
    "gamma": ["scale", "auto"],           # kernel coefficient
    "coef0": [0.0, 0.1, 0.5]              # only for 'poly' and 'sigmoid'
}

# For SVM Regressor
svr_param_grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "degree": [2, 3, 4],
    "gamma": ["scale", "auto"],
    "epsilon": [0.01, 0.1, 0.5, 1.0],     # epsilon-tube within which no penalty is given
    "coef0": [0.0, 0.1, 0.5]
}

# Random Forest Classifier
rf_param_grid = {
    "n_estimators": [50, 100, 200, 300],        # number of trees
    "criterion": ["gini", "entropy", "log_loss"],  # quality of split
    "max_depth": [None, 5, 10, 20, 30],        # max depth of each tree
    "min_samples_split": [2, 5, 10],           # min samples required to split
    "min_samples_leaf": [1, 2, 4],             # min samples required at leaf
    "max_features": ["auto", "sqrt", "log2"],  # number of features to consider
    "bootstrap": [True, False]                 # whether to use bootstrap samples
}

# Random Forest Regressor
rf_reg_param_grid = {
    "n_estimators": [50, 100, 200, 300],
    "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
    "max_depth": [None, 5, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["auto", "sqrt", "log2"],
    "bootstrap": [True, False]
}



st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Wikimedia_Brand_Guidelines_Update_2022_Wikimedia_Logo_Brandmark.png/1200px-Wikimedia_Brand_Guidelines_Update_2022_Wikimedia_Logo_Brandmark.png")
st.title("Best Model Selector For Your Prediction")
st.subheader("In this model you can find best prediction model according to your file")
# ------------------file uploading----------------------
st.text("Upload your file")
file_up = st.file_uploader("Upload file in csv format",type=["csv"])
# ------------------data handling-----------------------
if file_up:
    st.toast("Your file successfully uploaded",icon="🦾")
    data = pd.read_csv(file_up)
    # --------------data preview------------------
    st.dataframe(data.head(10))
    column = data.columns
    rem = ["None"] + [col for col in column]
    remove_n = st.selectbox("Remove unnessecary column that reduce model efficency",rem)
    if remove_n!="None":
        data.drop(columns=[remove_n],axis=1,inplace=True)
        st.toast("Column is Removed",icon="🥰")
    select=st.selectbox("Select your target column",column)
    st.toast("Target column selected",icon="😎")

    if select:
            X = data.drop(columns=[select],axis=1)
            y = data[select]
            st.title("Select model")
            classify,regression = st.columns(2)
            with classify:
                st.image("https://cdn.botpenguin.com/assets/website/Classification_Algorithm_f6b45f3f99.png",width=100)
                cl = st.button("select_classification")
                if cl:
                    st.success("Classification is selcted")
            with regression:
                st.image("https://serokell.io/files/bn/bnug59b5.1_(29)_(1).jpg",width=100)
                re = st.button("select_regression")
                if re:
                    st.success("Regression is selcted")
            if cl:
                y = LabelEncoder().fit_transform(y)
            
# Training dataset

            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

            numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
            categorical_features = X_train.select_dtypes(include=["object","category"]).columns.tolist()

            # transformers
            numeric_transformes = Pipeline(steps=[
                ("Imputation",SimpleImputer(strategy="median")),
                ("scaling",StandardScaler())
            ])
        
            categorical_transformers = Pipeline(steps=[
                ("Imputation",SimpleImputer(strategy="most_frequent")),
                ("Encoding",OneHotEncoder(sparse_output=False,handle_unknown="ignore"))
            ])

            preprocess = ColumnTransformer(
                transformers=[
                    ("Number",numeric_transformes,numeric_features),
                    ("Category",categorical_transformers,categorical_features)
                ]
            )
            
            if cl:
                st.subheader("Select and See How different model work  different your dataset ")
                tab_1, tab_2, tab_3, tab_4, tab_5 = st.tabs(["Logistic Regression","Decision Tree Classifier","KNeighbour Classifier","SVC","Random Forest Classifier"])
                # model fitting
                acc_score={}
                # logistic
                with tab_1:
                    log_pipline = Pipeline(steps=[
                        ("Preprocess",preprocess),
                        ("Classify",LogisticRegression())
                    ])
                    log_pipline.fit(X_train,y_train)
                    log_pred = log_pipline.predict(X_test)
                    class_log = classification_report(y_test,log_pred)
                    tab_1.write(f"The classification report is \n{class_log}")
                    log_acc = accuracy_score(y_test,log_pred)
                    acc_score["Logistic Regression"]=log_acc
                with tab_2:
                    tree_class = Pipeline(steps=[
                        ("Preprocess",preprocess),
                        ("Classify",RandomizedSearchCV(DecisionTreeClassifier(),dt_param_grid,n_iter=100))
                    ])
                    
                    tree_class.fit(X_train,y_train)
                    tree_class_pred = tree_class.predict(X_test)
                    class_tree = classification_report(y_test,tree_class_pred)
                    tab_2.write(f"The classification report is \n{class_tree}")
                    tree_acc = accuracy_score(y_test,tree_class_pred)
                    acc_score["Decision Tree Classifier"]=tree_acc
                with tab_3:
                    neighbour_class = Pipeline(steps=[
                        ("Preprocess",preprocess),
                        ("Classify",RandomizedSearchCV(KNeighborsClassifier(),knn_param_grid,n_iter=100))
                    ])
                    neighbour_class.fit(X_train,y_train)
                    neighbour_class_pred = neighbour_class.predict(X_test)
                    class_neighbour = classification_report(y_test,neighbour_class_pred)
                    tab_3.write(f"The classification report is \n{class_neighbour}")
                    k_acc = accuracy_score(y_test,neighbour_class_pred)
                    acc_score["KNeighbour classifier"]=k_acc
                with tab_4:
                    SVC_class = Pipeline(steps=[
                        ("Preprocess",preprocess),
                        ("Classify",RandomizedSearchCV(SVC(),svm_param_grid,n_iter=100))
                    ])
                    SVC_class.fit(X_train,y_train)
                    SVC_class_pred = SVC_class.predict(X_test)
                    class_SVC = classification_report(y_test,SVC_class_pred)
                    tab_4.write(f"The classification report is \n{class_SVC}")
                    svc_acc = accuracy_score(y_test,SVC_class_pred)
                    acc_score["SVC"]=svc_acc
                with tab_5:
                    random_class = Pipeline(steps=[
                        ("Preprocess",preprocess),
                        ("Classify",RandomizedSearchCV(RandomForestClassifier(),rf_param_grid,n_iter=100))
                    ])
                    random_class.fit(X_train,y_train)
                    random_class_pred = random_class.predict(X_test)
                    class_random = classification_report(y_test,random_class_pred)
                    tab_5.write(f"The classification report is \n{class_random}")
                    random_acc = accuracy_score(y_test,random_class_pred)
                    acc_score["Random Forest classifier"]=random_acc
                best_model = max(acc_score,key=acc_score.get)
                mod_score = acc_score[best_model]

                progress_text = "Operation in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)

                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                time.sleep(1)
                my_bar.empty()
                st.markdown(f"The best Model is {best_model} with accuracy score - {mod_score*100} ")
                st.success("Model trained successfully")
            if re:
                st.subheader("Select and See How different model work  different your dataset ")
                tab_1_R, tab_2_R, tab_3_R, tab_4_R, tab_5_R = st.tabs(["Linear Regression","Decision Tree Regressior","KNeighbour Regressior","SVR","Random Forest Regressior"])
                # model fitting
                r_score={}
                # Linear Regression
                with tab_1_R:
                    log_pipline_R = Pipeline(steps=[
                        ("Preprocess",preprocess),
                        ("Regression",LinearRegression())
                    ])
                    log_pipline_R.fit(X_train,y_train)
                    log_pred_R = log_pipline_R.predict(X_test)
                    mse_log = mean_squared_error(y_test,log_pred_R)
                    mae_log = mean_absolute_error(y_test,log_pred_R)
                    log_r = r2_score(y_test,log_pred_R)
                    r_score["Linear regression"] = log_r
                    tab_1_R.write(f"The Mean squared error is {mse_log:2f}")
                    tab_1_R.write(f"The Mean Absolute Error is {mae_log:2f}")
                with tab_2_R:
                    tree_class = Pipeline(steps=[
                        ("Preprocess",preprocess),
                        ("Regression",RandomizedSearchCV(DecisionTreeRegressor(),dt_reg_param_grid,n_iter=100))
                    ])
                    
                    tree_class.fit(X_train,y_train)
                    tree_class_pred = tree_class.predict(X_test)
                    mse_tree = mean_squared_error(y_test,tree_class_pred)
                    mae_tree = mean_absolute_error(y_test,tree_class_pred)
                    tree_r = r2_score(y_test,tree_class_pred)
                    r_score["Decision Tree Regression"] = tree_r
                    st.write(f"The Mean squared error is {mse_tree:2f}")
                    tab_2_R.write(f"The Mean Absolute Error is {mae_tree:2f}")
                with tab_3_R:
                    neighbour_reg = Pipeline(steps=[
                        ("Preprocess",preprocess),
                        ("Regression",RandomizedSearchCV(KNeighborsRegressor(),knn_reg_param_grid,n_iter=100))
                    ])
                    neighbour_reg.fit(X_train,y_train)
                    neighbour_reg_pred = neighbour_reg.predict(X_test)
                    mse_knr = mean_squared_error(y_test,neighbour_reg_pred)
                    mae_knr = mean_absolute_error(y_test,neighbour_reg_pred)
                    knr_r = r2_score(y_test,neighbour_reg_pred)
                    r_score["KNeighbour Regression"] = knr_r
                    tab_3_R.write(f"The Mean squared error is {mse_knr:2f}")
                    tab_3_R.write(f"The Mean Absolute Error is {mae_knr:2f}")
                with tab_4_R:
                    SVR_reg = Pipeline(steps=[
                        ("Preprocess",preprocess),
                        ("Regression",RandomizedSearchCV(SVR(),svr_param_grid,n_iter=100))
                    ])
                    SVR_reg.fit(X_train,y_train)
                    SVR_reg_pred = SVR_reg.predict(X_test)
                    mse_SVR = mean_squared_error(y_test,SVR_reg_pred)
                    mae_SVR = mean_absolute_error(y_test,SVR_reg_pred)
                    svr_r = r2_score(y_test,SVR_reg_pred)
                    r_score["SVR"] = svr_r
                    tab_4_R.write(f"The Mean squared error is {mse_SVR:2f}")
                    tab_4_R.write(f"The Mean Absolute Error is {mae_SVR:2f}")
                with tab_5_R:
                    random_reg = Pipeline(steps=[
                        ("Preprocess",preprocess),
                        ("Regression",RandomizedSearchCV(RandomForestRegressor(),rf_reg_param_grid,n_iter=100))
                    ])
                    random_reg.fit(X_train,y_train)
                    random_reg_pred = random_reg.predict(X_test)
                    rand_r = r2_score(y_test,random_reg_pred)
                    r_score["random tree regressior"] = rand_r
                    mse_Random = mean_squared_error(y_test,random_reg_pred)
                    mae_Random = mean_absolute_error(y_test,random_reg_pred)
                    tab_5_R.write(f"The Mean squared error is {mse_Random:2f}")
                    tab_5_R.write(f"The Mean Absolute Error is {mae_Random:2f}")


                progress_text = "Operation in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)

                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                time.sleep(1)
                my_bar.empty()
                best_model_r = max(r_score,key=r_score.get)
                mod_score_r = r_score[best_model_r]
                st.markdown(f"The best Model is {best_model_r} with accuracy score - {mod_score_r*100} ")

                st.success("Model trained successfully")
