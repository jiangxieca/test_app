import streamlit as st
import requests

def main():
    st.title("Iris Classifier")

    with st.sidebar:
        petal_length = st.number_input("Petal Length", value=4.7, min_value=1.0, max_value=6.9, step=0.1)
        petal_width = st.number_input("Petal Width", value=1.5, min_value=0.1, max_value=2.5, step=0.1)
        sepal_length = st.number_input("Sepal Length", value=6.7, min_value=4.3, max_value=7.9, step=0.1)
        sepal_width = st.number_input("Sepal Width", value=3.1, min_value=2.0, max_value=4.4, step=0.1)
        classify = st.button("Classify")

    if classify:
        url=f"http://127.0.0.1:8000/iris?petal_length={petal_length}&petal_width={petal_width}&sepal_length={sepal_length}&sepal_width={sepal_width}"

        response = requests.get(url)
        if response.status_code == 200:
            prediction = response.json()
            st.write(prediction["class"])
        else:
            st.error("An error happened")
            st.error(response.json())
            
            
if __name__ == "__main__":
    main()
    