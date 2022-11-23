import os
import modal

LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()
    titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")
    titanic_df.dropna(inplace=True)
    titanic_df.drop(['PassengerId','Name', 'Ticket', 'Cabin'], axis='columns', inplace=True)
    titanic_df.Sex.replace({'male':0, 'female':1}, inplace=True)
    titanic_df.Embarked.replace({'S':0, 'C':1, 'Q':2}, inplace=True)
    print(titanic_df.head())
    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_modal",
        version=3,
        primary_key=["Pclass","Sex","Age", 'SibSp', 'Parch', "Fare",'Embarked'], 
        description="Titanic survival dataset")
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()