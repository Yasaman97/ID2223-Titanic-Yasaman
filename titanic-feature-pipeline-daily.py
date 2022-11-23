import os
import modal
    
BACKFILL=False
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def generate_passenger(Survived, Pclass_max, Pclass_min, Sex_max, Sex_min, Age_max, Age_min, SibSp_max, SibSp_min, Parch_max, Parch_min, Fare_max, Fare_min, Embarked_max, Embarked_min):
    """
    Returns a single passenger as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({ "Plass": [random.randint(Pclass_max, Pclass_min)],
                       "Sex": [random.randint(Sex_max, Sex_min)],
                       "Age": [random.uniform(Age_max, Age_min)],
                       "SibSp": [random.randint(SibSp_max, SibSp_min)],
                       "Parch": [random.randint(Parch_max, Parch_min)],
                       "Fare": [random.uniformt(Fare_max, Fare_min)],
                       "Embarked": [random.randint(Embarked_max, Embarked_min)],
                      })
    df['Surviveed'] = Survived
    return df


def get_random_passenger():
    """
    Returns a DataFrame containing one random passenger
    """
    import pandas as pd
    import random

    Survived_df = generate_passenger(1, 3, 1, 1, 0, 80, 0.42, 4, 0, 5, 0, 512.3292, 0, 2, 0)
    NotSurvived_df = generate_passenger(0, 3, 1, 1, 0, 74, 1, 8, 0, 6, 0, 263.0, 0, 2, 0)


    # randomly pick one of these 2 and write it to the featurestore
    pick_random = random.randint(1,2)
    if pick_random == 2:
        titanic_df = Survived_df
        print("Survived added")
    else:
        titanic_df = NotSurvived_df
        print("Not Survived added")

    return titanic_df



def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    if BACKFILL == True:
        titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")
    else:
        titanic_df = get_random_passenger()

    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_modal",
        version=1,
        primary_key=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"], 
        description="Titanic dataset")
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
