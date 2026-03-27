# ============================================================================================
# 1.Import Statements
# ============================================================================================


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier,plot_tree


# ============================================================================================
# 2. Reading Dataset from xlsx file
# ============================================================================================

data=pd.read_excel("dt_datasets.xlsx")

df=pd.DataFrame(data)
print(df.head())

print("\nClass Distribution:\n")
print(df["Decision"].value_counts())


# ============================================================================================
# 3. Features and Label separation
# ============================================================================================

x=df.drop("Decision",axis=1)
y=df["Decision"]


# ============================================================================================
# 4. Preprocessing Datasets
# ============================================================================================

cat_fea=["Weather","Parent","Money"]

preprocessor=ColumnTransformer(
    transformers=[
        ("cat",OneHotEncoder(handle_unknown="ignore"),cat_fea)
        ]
    )


# ============================================================================================
# 5. Binding Pipeline
# ============================================================================================

pipe=Pipeline(steps=[
    ("preprocessing",preprocessor),
    ("model",DecisionTreeClassifier(
        criterion="entropy",
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42)
     )
    ]
)


# ============================================================================================
# 6. Train on Full Dataset
# ============================================================================================

pipe.fit(x,y)

print("\nTraining Accuracy :",pipe.score(x,y))


# ============================================================================================
# 7. Clean Decision Tree Visualisation
# ============================================================================================

fea_name=pipe.named_steps["preprocessing"].get_feature_names_out()

plt.figure(figsize=(20,10),dpi=150)

plot_tree(
    pipe.named_steps["model"],
    feature_names=fea_name,
    class_names=pipe.named_steps["model"].classes_,
    filled=True,
    rounded=True,
    fontsize=10
    )

plt.title("Decision Tree (Teaching Version) ",fontsize=16)
plt.tight_layout()
plt.savefig("clean_decision_tree.png",dpi=300)
plt.show()


# ============================================================================================
# 8. User Input Prediction
# ============================================================================================

def pred():
    print("\n Enter User Details:")

    weather=input("Weather (Sunny/Windy/Rainy): ")
    parent= input("Parent (Yes/No): ")
    money =input("Money (Rich/Poor): ")
    
    user_df= pd.DataFrame({
        "Weather": [weather], "Parent": [parent], "Money": [money]
        })
    
    prediction =pipe.predict(user_df)[0]
    probability =pipe.predict_proba(user_df)

    print("\nPredicted Decision:", prediction)
    print("Class Probabilities:")

    for cls,prob in zip(pipe.named_steps["model"].classes_,probability[0]):
        print(f"{cls}: {round(prob*100,2)}%")


pred()
    