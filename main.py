import streamlit as st
import sentiment as se


# Streamlit app layout
def main():
    st.title('Social Media Sentiment Analysis App')
    st.write('Enter any Social Media statements to analyze sentiment:')
    user_input = st.text_area('Input your text here:', '')

    if st.button('Analyze'):
        if user_input:
            output = se.prediction(user_input)
            if output[0] == "positive":
                st.write("**Sentiment:** Positive 😀")
            elif output[0] == 'negative':
                st.write("**Sentiment:** Negative 😞")
            else:
                st.write("Something went wrong. Please type another sentence.")


if __name__ == '__main__':
    main()
