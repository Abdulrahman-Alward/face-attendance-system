@echo off
REM Launches the Streamlit app using the `cv2` conda env, where all the
REM project dependencies (dlib + CUDA, torch + CUDA, facenet-pytorch,
REM insightface, streamlit, ...) are installed.
"C:\Users\Award\anaconda3\envs\cv2\python.exe" -m streamlit run "%~dp0app.py"
