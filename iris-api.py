from fastapi import FastAPI, Query
import uvicorn
import pandas as pd
import pickle

app = FastAPI(title="IRIS Classifier")

@app.get("/")
def homepage():
    return {"message": "Iris Classifier"}

@app.get("/iris")
def iris_classifier(petal_length: float = Query(description="Petal Length",
                                               default=4.7,
                                               ge=1.0, le=6.9),
                    petal_width: float = Query(description="Petal Width",
                                               default=1.5,
                                               ge=0.1, le=2.5),
                    sepal_length: float = Query(description="Sepal Length",
                                               default=6.7,
                                               ge=4.3, le=7.9),
                    sepal_width: float = Query(description="Sepal Width",
                                               default=3.1,
                                               ge=2.0, le=4.4)):
    # preparing the dataframe
    column_names = ["sepal length (cm)", "sepal width (cm)",
                    "petal length (cm)", "petal width (cm)"]
    data = [[sepal_length, sepal_width,
            petal_length, petal_width]]
    df = pd.DataFrame(data=data, columns=column_names)

    # load pipeline
    pipe = pickle.load(open("iris-pipe.pkl", "rb"))
    predictions = pipe.predict(df)
    
    return {"class": predictions[0]}

if __name__ == "__main__":
    uvicorn.run("iris-api:app",
                host="0.0.0.0",
                port=8000,
                reload=True)