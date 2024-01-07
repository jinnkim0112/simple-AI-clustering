import streamlit as st
from clustering import clustering

st.title('Clustering with GPT')

if 'classes' not in st.session_state:
    st.session_state['classes'] = ['', '']

classes = st.session_state['classes']

st.header('How to use:')
st.subheader('1. click on the number of classes you want to classify')
if st.button("Add a class"):
    classes.append('')

st.subheader('2. then type in the classes')
for classInt in range(len(classes)):
    classes[classInt] = st.text_input('class'+ str(classInt), classes[classInt])
    
st.write(str(len(classes)) + ' classes' + ': '+ ', '.join(classes))

st.subheader('3. upload the file of the image that you have')
st.write('accepts only jpg at the time')

uploaded_file = st.file_uploader("Choose a jpg file")
if uploaded_file is not None:
    st.image(uploaded_file)

st.subheader('4. click on this button to let GPT classify')
if st.button("Classify", type="primary"):
    if uploaded_file is None:
        st.warning('please upload the image')
    else:
        with st.spinner('wait for it...'):
            result = clustering(uploaded_file, classes)
        st.header(result)