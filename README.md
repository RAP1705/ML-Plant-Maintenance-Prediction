### Project Steps:
>> 1. Create Custom ML Model with our Data feeding in Logistic regression Algo
>> 2. Expose ML Model as Fast API
>> 3. Consume the Fast API from Anywhere (Streamlit)



# Activate python Environment
python -m venv env-PM
.\env-PM\Scripts\activate

# To install all the required python Packages/modules
pip install -r requirements.txt

# Optional -  To install individual python Packages/Modules
pip install <module/package Name>

# Optional - To display any Install Python Package/Modules details
pip show <package_name>
pip list

# Generate ML Model and get the accuracy in terminal
go to the backend directory using  --> CD 
execute below command:
python ml_training.py


# Activate the FastAPIs
uvicorn vapp:app --reload

# Validate the APIs and Try Out
http://127.0.0.1:8000/docs


# To run Streamlit UI
Streamlit run <fileName>.py


